"""Graphical user interface functions."""

from typing import Any

import napari
import napari.utils.colormaps
import numpy as np
import numpy.typing as npt
import pandas as pd
from vispy.color import Colormap


def show_analysis(
    image: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    spot_image1: npt.NDArray[Any],
    spot_image2: npt.NDArray[Any],
    channel_names: list[str] | None = None,
    visible_channels: list[int] | None = None,
    label_df: pd.DataFrame | None = None,
    spot_df: pd.DataFrame | None = None,
) -> None:
    """Show the spot analysis images.

    Optional spots dataframe must contain a column named
    'channel' with values of either 1 or 2.

    Args:
        image: Image (CYX).
        label_image: Label image.
        spot_image1: First channel spots.
        spot_image2: Second channel spots.
        channel_names: Name of image channels.
        visible_channels: Image channels to display.
        label_df: DataFrame with 1 row per label.
        spot_df: DataFrame with 1 row per label for each channel.
    """
    viewer = napari.Viewer()
    n = image.shape[0]
    if channel_names is None:
        channel_names = ["Channel " + str(i) for i in range(n)]
    else:
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
    object_labels = viewer.add_labels(
        label_image,
        name="Objects",
        features=_to_features(label_df, np.max(label_image)),
    )
    object_labels.preserve_labels = True
    object_labels.contour = 1
    # Add spots
    labels1 = viewer.add_labels(
        spot_image1,
        name="Spots 1",
        features=_to_features(spot_df, np.max(spot_image1), channel=1),
    )
    labels1.preserve_labels = True
    labels2 = viewer.add_labels(
        spot_image2,
        name="Spots 2",
        features=_to_features(spot_df, np.max(spot_image2), channel=2),
    )
    labels2.preserve_labels = True
    # Avoid color clash if all spots overlap
    labels2.colormap = napari.utils.colormaps.label_colormap(seed=0.12345)

    viewer.reset_view()
    viewer.layers.selection.active = object_labels

    viewer.window.add_plugin_dock_widget("napari", "Features table widget")

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


def _to_features(
    df: pd.DataFrame | None, size: int, channel: int | None = None
) -> pd.DataFrame | None:
    """Convert data to a features table."""
    if df is None:
        return None
    if channel is not None and "channel" in df.columns:
        df = df[df["channel"] == channel].copy()
    if len(df) != size:
        return None
    # Insert an empty row for the background label
    df = pd.concat([pd.DataFrame([{col: 0 for col in df.columns}]), df])
    return df
