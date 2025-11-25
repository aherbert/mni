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
        "--std",
        default=3,
        type=float,
        help="Std.dev above the mean (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import math
    import os
    from typing import Any

    import numpy as np
    import numpy.typing as npt

    # import skimage.filters
    from tifffile import imread, imwrite

    from mni.utils import object_threshold

    def std_above_mean(h: npt.NDArray[Any]) -> int:
        values = np.arange(len(h))
        probs = h / np.sum(h)
        mean = np.sum(probs * values)
        sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
        t = np.clip(math.floor(mean + args.std * sd), 0, values[-1])
        return int(t)

    im = imread(args.image)
    mask = imread(args.mask)

    # TODO: apply Gaussian filter to image pixels

    m = object_threshold(
        im,
        mask,
        std_above_mean,
    )
    base, suffix = os.path.splitext(args.image)
    fn = f"{base}.objects{suffix}"
    print(fn)
    imwrite(fn, m, compression="zlib")


if __name__ == "__main__":
    main()
