"""Segmentation functions."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from cellpose import models
from skimage.segmentation import clear_border
from skimage.util import map_array


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
        model_type=model_type,
    )
    # 0 = greyscale; [[2, 1]] = cells in green (2), nuclei in red (1)
    channels = [[0, 0]]

    array, _flows, _styles = model.eval(
        img,
        channels=channels,
        normalize=True,
        diameter=diameter,
    )
    return array


def _get_device(device: str | None = None) -> torch.device:
    """Get a torch device given the available backends."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def filter_segmentation(
    mask: npt.NDArray[Any], border: int = 5, relabel: bool = True
) -> tuple[npt.NDArray[Any], int]:
    """Removes border objects and filters small objects from segmentation mask.

    The size of the border can be specified. Use a negative size to skip removal of
    border objects.

    Objects are optionally relabeled to be continuous from 1.
    Note: Removal of border objects can result in missing labels from the
    original [0, N] label IDs.

    Args:
        mask: unfiltered segmentation mask
        border: width of the border examined (negative to disable)
        relabel: Set to True to relabel objects in [0, N]

    Returns:
        filtered segmentation mask, number of objects (N)
    """
    cleared: npt.NDArray[Any] = (
        mask if border < 0 else clear_border(mask, buffer_size=border)  # type: ignore[no-untyped-call]
    )
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    new_mask = cells_cleaned * mask
    n = np.sum(mask_sizes)
    if relabel:
        old_id = np.arange(len(sizes))[mask_sizes]
        new_mask = map_array(
            new_mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
        )
    return new_mask, n  # type: ignore[no-any-return]


def relabel(
    mask: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], list[int]]:
    """Relabels the mask to be continuous from 1.

    The old id array contains the original ID for each new label:

    original id = old_id[label - 1]

    Args:
        mask: unfiltered segmentation mask

    Returns:
        relabeled mask, old_id
    """
    sizes = np.bincount(mask.ravel())
    sizes[0] = 0
    mask_sizes = sizes != 0
    old_id = np.arange(len(sizes))[mask_sizes]
    n = len(old_id)
    new_mask = map_array(
        mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
    )
    return new_mask, old_id  # type: ignore[no-any-return]
