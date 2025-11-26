#!/usr/bin/env python3
"""Program to threshold each object in an image."""

import argparse


def main() -> None:
    """Program to threshold each object in an image."""
    parser = argparse.ArgumentParser(
        description="""Program to threshold each object in an image."""
    )
    _ = parser.add_argument("image", help="Image (TIFF)")
    _ = parser.add_argument("mask", help="Mask image (TIFF)")

    group = parser.add_argument_group("Threshold Options")
    _ = group.add_argument(
        "--sigma",
        default=1.5,
        type=float,
        help="Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--method",
        default="mean_plus_std",
        choices=["mean_plus_std", "otsu", "yen", "minimum"],
        help="Thresholding method (default: %(default)s)",
    )
    _ = group.add_argument(
        "--std",
        default=4,
        type=float,
        help="Std.dev above the mean (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import os

    from scipy.ndimage import gaussian_filter
    from tifffile import imread, imwrite

    from mni.utils import object_threshold, threshold_method

    fun = threshold_method(args.method, std=args.std)

    im = imread(args.image)
    mask = imread(args.mask)

    if args.sigma > 0:
        im = gaussian_filter(im, args.sigma, mode="mirror")

    m = object_threshold(
        im,
        mask,
        fun,
    )
    base, suffix = os.path.splitext(args.image)
    fn = f"{base}.objects{suffix}"
    print(fn)
    imwrite(fn, m, compression="zlib")


if __name__ == "__main__":
    main()
