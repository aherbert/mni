"""Segmentation functions."""

from typing import Any

import numpy.typing as npt
import torch
from cellpose import models

from .utils import compact_mask


def segment(
    img: npt.NDArray[Any],
    model_type: str,
    diameter: float,
    device: str | None | torch.device = None,
) -> npt.NDArray[Any]:
    """Run greyscale segmentation.

    Args:
        img: Image to segment.
        model_type: Name of the model.
        diameter: Expected diameter of objects.
        device: Name of the torch device.
    """
    device = (
        device if isinstance(device, torch.device) else _get_device(device)
    )
    model = models.CellposeModel(
        device=device,
        pretrained_model=model_type,
    )

    array, _flows, _styles = model.eval(
        img,
        normalize=True,
        diameter=diameter,
    )
    return compact_mask(array)


def _get_device(device: str | None = None) -> torch.device:
    """Get a torch device given the available backends."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
