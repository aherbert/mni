#!/usr/bin/env python3
"""Program to show the spot analysis result images."""

import argparse


def main() -> None:
    """Program to show the spot analysis result images."""
    parser = argparse.ArgumentParser(
        description="""Program to show the spot analysis result images."""
    )
    _ = parser.add_argument("image", help="Image (TIFF)")
    _ = parser.add_argument("mask", help="Mask image (TIFF)")
    _ = parser.add_argument("label1", help="First channel spot labels (TIFF)")
    _ = parser.add_argument("label2", help="Second channel spot labels (TIFF)")

    _ = parser.add_argument(
        "--channel-names", nargs="+", default=[], help="Channel names"
    )
    _ = parser.add_argument(
        "--visible-channels",
        nargs="+",
        type=int,
        default=[],
        help="Visible channels",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    from tifffile import imread

    from mni.gui import show_analysis

    image = imread(args.image)
    label_image = imread(args.mask)
    label1 = imread(args.label1)
    label2 = imread(args.label2)

    show_analysis(
        image,
        label_image,
        label1,
        label2,
        channel_names=args.channel_names,
        visible_channels=args.visible_channels,
    )


if __name__ == "__main__":
    main()
