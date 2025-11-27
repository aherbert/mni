#!/usr/bin/env python3
"""Program to find micro-nuclei in a mask image."""

import argparse


def main() -> None:
    """Program to find micro-nuclei in a mask image."""
    parser = argparse.ArgumentParser(
        description="""Program to find micro-nuclei in a mask image."""
    )
    _ = parser.add_argument("mask", help="Mask image (TIFF)")

    group = parser.add_argument_group("Micro-nuclei Options")
    _ = group.add_argument(
        "--distance",
        default=20,
        type=int,
        help="Search distance for bleb (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--size",
        default=2000,
        type=int,
        help="Maximum micro-nucleus size (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--min-size",
        default=50,
        type=int,
        help="Minimum micro-nucleus size (pixels) (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    from tifffile import imread

    from mni.utils import find_micronuclei

    im = imread(args.mask)
    data = find_micronuclei(
        im, distance=args.distance, size=args.size, min_size=args.min_size
    )
    for x in data:
        print(x)


if __name__ == "__main__":
    main()
