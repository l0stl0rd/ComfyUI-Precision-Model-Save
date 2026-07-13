import json
import logging
import os

import torch

import comfy.lora
import comfy.model_management
import comfy.utils
import folder_paths
from comfy.cli_args import args


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class PrecisionModelSave:
    """
    Materialises a ComfyUI MODEL's weight patches in float32, then saves the
    resulting standalone diffusion model in a chosen output dtype.

    This intentionally does not call comfy.sd.save_checkpoint(), so it avoids
    the normal ModelSave materialisation/casting path. It is intended as a
    diagnostic saver for live model merges.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "diffusion_models/precision_merge"},
                ),
                "save_dtype": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
                "strip_diffusion_model_prefix": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model/merging"

    @staticmethod
    def _materialise_weight(key, patch_description, output_dtype):
        """
        patch_description follows ModelPatcher.get_key_patches():
          [(base_weight, convert_func), patch_1, patch_2, ...]
        """
        if not patch_description:
            raise RuntimeError(f"No patch description for {key}")

        base_weight, convert_func = patch_description[0]

        # Work on CPU in FP32 so interpolation/extrapolation is not rounded to
        # the source checkpoint dtype before saving.
        if not isinstance(base_weight, torch.Tensor):
            raise TypeError(
                f"{key}: unsupported base weight type {type(base_weight).__name__}"
            )

        weight = base_weight.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        )

        if convert_func is not None:
            try:
                weight = convert_func(weight, inplace=True)
            except TypeError:
                weight = convert_func(weight)

        patches = patch_description[1:]
        if patches:
            # Current ComfyUI supports intermediate_dtype here. The fallback
            # keeps the node usable on somewhat older builds.
            try:
                weight = comfy.lora.calculate_weight(
                    patches,
                    weight,
                    key,
                    intermediate_dtype=torch.float32,
                )
            except TypeError:
                weight = comfy.lora.calculate_weight(patches, weight, key)

        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                f"{key}: materialised weight is {type(weight).__name__}, not Tensor"
            )

        return weight.detach().to(
            device="cpu", dtype=output_dtype, copy=False
        ).contiguous()

    def save(
        self,
        model,
        filename_prefix,
        save_dtype,
        strip_diffusion_model_prefix,
        prompt=None,
        extra_pnginfo=None,
    ):
        output_dtype = DTYPES[save_dtype]

        full_output_folder, filename, counter, subfolder, resolved_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        output_name = f"{filename}_{counter:05}_.safetensors"
        output_path = os.path.join(full_output_folder, output_name)

        logging.info(
            "[PrecisionModelSave] Materialising model patches in FP32; "
            "saving as %s to %s",
            save_dtype,
            output_path,
        )

        # This returns the base tensor plus every live merge/patch attached to
        # the MODEL object, rather than only model.model.state_dict().
        patch_map = model.get_key_patches("diffusion_model.")

        if not patch_map:
            raise RuntimeError(
                "No diffusion_model.* weights found. "
                "This node expects a ComfyUI MODEL/ModelPatcher."
            )

        state_dict = {}
        total = len(patch_map)

        for index, (internal_key, patch_description) in enumerate(
            patch_map.items(), start=1
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()

            save_key = internal_key
            if strip_diffusion_model_prefix and save_key.startswith(
                "diffusion_model."
            ):
                save_key = save_key[len("diffusion_model."):]

            state_dict[save_key] = self._materialise_weight(
                internal_key,
                patch_description,
                output_dtype,
            )

            if index == 1 or index % 100 == 0 or index == total:
                logging.info(
                    "[PrecisionModelSave] %d/%d tensors materialised",
                    index,
                    total,
                )

        metadata = {
            "format": "pt",
            "comfy_precision_model_save": "1",
            "materialise_dtype": "fp32",
            "saved_dtype": save_dtype,
            "stripped_diffusion_model_prefix": str(
                bool(strip_diffusion_model_prefix)
            ).lower(),
        }

        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata[key] = json.dumps(value)

        comfy.utils.save_torch_file(
            state_dict,
            output_path,
            metadata=metadata,
        )

        # Release references promptly; large FP32 dictionaries can consume a
        # lot of system RAM until Python's next collection.
        del state_dict

        logging.info("[PrecisionModelSave] Saved: %s", output_path)
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "PrecisionModelSave": PrecisionModelSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrecisionModelSave": "Precision Model Save (FP32 Materialise)",
}
