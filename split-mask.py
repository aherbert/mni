#!/usr/bin/env python3
"""Program to split a mask image."""

import argparse


def main() -> None:
    """Program to split a mask image."""
    parser = argparse.ArgumentParser(
        description="""Program to split a mask image into an image per object."""
    )
    _ = parser.add_argument("mask", help="Mask image (TIFF)")
    _ = parser.add_argument(
        "--bytes",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use 1 byte per sample (uint 8); default is 1 bit per sample (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import os

    import numpy as np
    from tifffile import imread, imwrite

    im = imread(args.mask)
    base, suffix = os.path.splitext(args.mask)
    n = np.max(im)
    for n in range(1, np.max(im) + 1):
        m = im == n
        if np.any(m):
            fn = f"{base}.{n}{suffix}"
            print(fn)
            if args.bytes:
                m = m.astype(np.uint8) * 255
            imwrite(fn, m, photometric="minisblack", compression="zlib")


if __name__ == "__main__":
    main()
