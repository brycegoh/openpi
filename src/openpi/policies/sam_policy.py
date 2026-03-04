import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_sam_example() -> dict:
    """Creates a random input example for the SAM policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/images/laptop": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/phone": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/side": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Insert cable into the connector.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SAMInputs(transforms.DataTransformFn):
    """
    Transform inputs for SAM (cable insertion) policy.
    Handles three camera views: laptop, phone, and side cameras.
    """

    # Action dimension for the robot
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        laptop_image = _parse_image(data.get("observation/images/laptop", data.get("laptop")))
        phone_image = _parse_image(data.get("observation/images/phone", data.get("phone")))
        side_image = _parse_image(data.get("observation/images/side", data.get("side")))

        # Create inputs dict with three camera views
        inputs = {
            "state": data["observation/state"] if "observation/state" in data else data.get("state"),
            "image": {
                "base_0_rgb": laptop_image,
                "left_wrist_0_rgb": phone_image,
                "right_wrist_0_rgb": side_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Include actions during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SAMOutputs(transforms.DataTransformFn):
    """
    Transform outputs from model back to SAM-specific format.
    """

    def __call__(self, data: dict) -> dict:
        # Return actions (typically 7-dim: 6 joints + gripper)
        return {"actions": np.asarray(data["actions"])}
