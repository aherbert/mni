"""Graphical user interface functions."""

from typing import Any

import napari
import napari.utils.colormaps
import numpy as np
import numpy.typing as npt
from vispy.color import Colormap


def show_analysis(
    image: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    spot_image1: npt.NDArray[Any],
    spot_image2: npt.NDArray[Any],
    channel_names: list[str] | None = None,
    visible_channels: list[int] | None = None,
) -> None:
    """Show the spot analysis images.

    Args:
        image: Image (CYX).
        label_image: Label image.
        spot_image1: First channel spots.
        spot_image2: Second channel spots.
        channel_names: Name of image channels.
        visible_channels: Image channels to display.
    """
    viewer = napari.Viewer()
    n = image.shape[0]
    if channel_names is None:
        channel_names = []
    while len(channel_names) < n:
        channel_names.append("Channel " + str(len(channel_names)))
    colors = _generate_color_map(channel_names)
    # Add image layers
    for i, im in enumerate(image):
        layer = viewer.add_image(im)
        layer.contrast_limits_range = (np.min(im), np.max(im))
        # This saturates the upper levels
        # layer.contrast_limits = np.percentile(im, (1, 99))
        layer.blending = "additive"
        layer.name = channel_names[i]
        layer.colormap = colors[i]
        layer.visible = i in visible_channels if visible_channels else True
    # Add labels
    labels = viewer.add_labels(label_image, name="Objects")
    labels.contour = 1
    _ = viewer.add_labels(spot_image1, name="Spots 1")
    labels = viewer.add_labels(spot_image2, name="Spots 2")
    # Avoid color clash if all spots overlap
    labels.colormap = napari.utils.colormaps.label_colormap(seed=0.12345)
    viewer.reset_view()
    napari.run()


def _generate_color_map(channel_names: list[str]) -> list[str | Colormap]:
    """Generate a list of color maps for the channels.

    Args:
        channel_names: Channel names.

    Returns:
        color maps
    """
    # Napari supports vispy or matplotlib colormap names

    # Determine the number of channels
    num_channels = len(channel_names)

    if num_channels == 1:
        return ["gray"]

    # Default channel color assignments
    special_channels: dict[str, str | Colormap] = {}

    # From CZI (Carl Zeiss) microscope image metadata
    for name, color in zip(
        ["AF647", "AF568", "AF488", "DAPI"],
        ["#FFFFFF", "#FF1800", "#00FF33", "#00A0FF"],
        strict=False,
    ):
        special_channels[name] = Colormap(["black", color])

    # Other color assignments. This list is used in reverse order amd repeated as required.
    # Supports using a Colormap. This requires the RBG value of the final color.
    remaining_colors = [
        Colormap(["black", "#ff2f92"]),  # strawberry
        Colormap(["black", "#8efa00"]),  # lime
        Colormap(["black", "#009193"]),  # teal
        Colormap(["black", "#00fdff"]),  # turquoise
        Colormap(["black", "#aa7942"]),  # brown
        Colormap(["black", "#ffc0cb"]),  # pink
        "bop orange",
        "bop blue",
        "bop purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "blue",
        "green",
        "red",
    ]
    # special case for grey/R/G/B
    if num_channels == 4:
        remaining_colors.append("gray")
    elif len(remaining_colors) < num_channels:
        # Do not run out of colours
        remaining_colors.insert(0, "gray")
        remaining_colors.extend(
            remaining_colors * (num_channels // len(remaining_colors))
        )

    return [
        special_channels[ch]
        if ch in special_channels
        else remaining_colors.pop()
        for ch in channel_names
    ]
