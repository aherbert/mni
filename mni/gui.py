"""Graphical user interface functions."""

from collections.abc import Callable
from typing import Any

import napari
import napari.utils.colormaps
import numpy as np
import numpy.typing as npt
import pandas as pd
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vispy.color import Colormap


class _AnalysisWidget(QWidget):  # type: ignore[misc]
    """Widget to add a button with an analysis function."""

    def __init__(
        self,
        text: str,
        analysis_fun: Callable[
            [
                npt.NDArray[Any],
                npt.NDArray[Any],
                npt.NDArray[Any],
            ],
            tuple[
                npt.NDArray[Any] | None,
                npt.NDArray[Any] | None,
                npt.NDArray[Any] | None,
                pd.DataFrame | None,
                pd.DataFrame | None,
            ],
        ],
    ) -> None:
        """Create the widget.

        Args:
            text: Text for the widget button.
            analysis_fun: Function to call.
        """
        super().__init__()
        self.analysis_fun = analysis_fun
        self.btn1 = QPushButton(text)
        self.btn1.clicked.connect(self.run)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn1)
        self.setLayout(self.layout)

    def run(self) -> None:
        """Repeat the analysis and update the features."""
        viewer = napari.current_viewer()
        o_layer = viewer.layers["Objects"]
        s_layer1 = viewer.layers["Spots 1"]
        s_layer2 = viewer.layers["Spots 2"]
        label_image = o_layer.data
        spot_image1 = s_layer1.data
        spot_image2 = s_layer2.data
        label_image_a, spot_image1_a, spot_image2_a, label_df, spot_df = (
            self.analysis_fun(label_image, spot_image1, spot_image2)
        )

        # Images may be updated
        if label_image_a is not None:
            o_layer.data = label_image_a
            label_image = label_image_a
        if spot_image1_a is not None:
            s_layer1.data = spot_image1_a
            spot_image1 = spot_image1_a
        if spot_image2_a is not None:
            s_layer2.data = spot_image2_a
            spot_image2 = spot_image2_a

        # Features must match the displayed image
        o_layer.features = _to_features(label_df, np.max(label_image))
        s_layer1.features = _to_features(
            spot_df, np.max(spot_image1), channel=1
        )
        if spot_image2 is not None:
            s_layer2.data = spot_image2
        s_layer2.features = _to_features(
            spot_df, np.max(spot_image2), channel=2
        )


def show_analysis(
    image: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    spot_image1: npt.NDArray[Any],
    spot_image2: npt.NDArray[Any],
    channel_names: list[str] | None = None,
    visible_channels: list[int] | None = None,
    label_df: pd.DataFrame | None = None,
    spot_df: pd.DataFrame | None = None,
    upper_limit: float = 100,
) -> None:
    """Show the spot analysis images.

    Optional spots dataframe must contain a column named
    'channel' with values of either 1 or 2.

    This method will block until the viewer is closed.

    Args:
        image: Image (CYX).
        label_image: Label image.
        spot_image1: First channel spots.
        spot_image2: Second channel spots.
        channel_names: Name of image channels.
        visible_channels: Image channels to display.
        label_df: DataFrame with 1 row per label.
        spot_df: DataFrame with 1 row per label for each channel.
        upper_limit: Upper limit (percentile) for contrast display range.
    """
    viewer = create_viewer(
        image,
        label_image,
        spot_image1,
        spot_image2,
        channel_names=channel_names,
        visible_channels=visible_channels,
        label_df=label_df,
        spot_df=spot_df,
        upper_limit=upper_limit,
    )
    show_viewer(viewer)


def create_viewer(
    image: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    spot_image1: npt.NDArray[Any],
    spot_image2: npt.NDArray[Any],
    channel_names: list[str] | None = None,
    visible_channels: list[int] | None = None,
    label_df: pd.DataFrame | None = None,
    spot_df: pd.DataFrame | None = None,
    upper_limit: float = 100,
) -> napari.Viewer:
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
        upper_limit: Upper limit (percentile) for contrast display range.
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
        im_min, im_max = np.min(im), np.max(im)
        layer.contrast_limits_range = (im_min, im_max)
        if upper_limit < 100:
            layer.contrast_limits = (im_min, np.percentile(im, upper_limit))
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
        # Avoid color clash if all spots overlap
        colormap=napari.utils.colormaps.label_colormap(seed=0.12345),
    )
    labels2.preserve_labels = True

    viewer.window.add_plugin_dock_widget("napari", "Features table widget")

    viewer.reset_view()
    viewer.layers.selection.active = object_labels

    return viewer


def add_analysis_function(
    viewer: napari.Viewer,
    analysis_fun: Callable[
        [
            npt.NDArray[Any],
            npt.NDArray[Any],
            npt.NDArray[Any],
        ],
        tuple[
            npt.NDArray[Any] | None,
            npt.NDArray[Any] | None,
            npt.NDArray[Any] | None,
            pd.DataFrame | None,
            pd.DataFrame | None,
        ],
    ],
) -> None:
    """Add the analysis function.

    The analysis function can update the mask images. If None
    is returned for a mask then the displayed mask is not changed.

    Args:
        viewer: Viewer.
        analysis_fun: Analysis function.
    """
    widget = _AnalysisWidget("Redo analysis", analysis_fun)
    viewer.window.add_dock_widget(widget, area="right")


def show_viewer(viewer: napari.Viewer) -> None:
    """Show the spot analysis viewer.

    This method will block until the viewer is closed.

    Args:
        viewer: Viewer.
    """
    # The viewer is not required; start the event loop
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
    # Order by label
    df.sort_values("label", inplace=True)
    # Insert an empty row for the background label
    df = pd.concat([pd.DataFrame([{col: 0 for col in df.columns}]), df])
    return df
