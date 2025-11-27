#!/usr/bin/env python3
"""Program to analyse the spots between two channels within an object."""

import argparse


def main() -> None:
    """Program to analyse the spots between two channels within an object."""
    parser = argparse.ArgumentParser(
        description="""Program to analyse the spots between two channels within an object."""
    )
    _ = parser.add_argument("mask", help="Mask image (TIFF)")
    _ = parser.add_argument("image1", help="First image (TIFF)")
    _ = parser.add_argument("label1", help="First image labels (TIFF)")
    _ = parser.add_argument("image2", help="Second image (TIFF)")
    _ = parser.add_argument("label2", help="Second image labels (TIFF)")

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

    from mni.utils import (
        classify_objects,
        collate_groups,
        find_micronuclei,
        find_objects,
        spot_analysis,
        spot_summary,
    )

    label_image = imread(args.mask)
    im1 = imread(args.image1)
    label1 = imread(args.label1)
    im2 = imread(args.image2)
    label2 = imread(args.label2)

    # find micro-nuclei and bleb parents
    objects = find_objects(label_image)
    data = find_micronuclei(
        label_image,
        objects=objects,
        distance=args.distance,
        size=args.size,
        min_size=args.min_size,
    )

    # Create groups. This collates blebs with their parent.
    groups = collate_groups(data)
    class_names = classify_objects(data, args.size, 10.0)

    results = spot_analysis(
        label_image, objects, groups, im1, label1, im2, label2
    )

    for r in results:
        parent = int(r[3])
        cls = class_names.get(parent, "")
        r2 = r[:4] + (cls,) + r[4:]
        print(r2)

    summary = spot_summary(results, groups)

    for s in summary:
        parent = int(s[1])
        cls = class_names.get(parent, "")
        s2 = s[:2] + (cls,) + s[2:] + (s[-2] + s[-1],)
        print(s2)


if __name__ == "__main__":
    main()
