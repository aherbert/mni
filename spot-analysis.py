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

    group = parser.add_argument_group("Spot Options")
    _ = group.add_argument(
        "--neighbour-distance",
        default=20.0,
        type=float,
        help="Search distance for nearest neighbour (pixels) (default: %(default)s)",
    )

    group = parser.add_argument_group("Save Options")
    _ = group.add_argument(
        "--spot-fn",
        help="Spot filename",
    )
    _ = group.add_argument(
        "--summary-fn",
        help="Summary filename",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    from tifffile import imread

    from mni.utils import (
        classify_objects,
        collate_groups,
        find_micronuclei,
        find_objects,
        format_spot_results,
        format_summary_results,
        save_csv,
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
    class_names = classify_objects(data, args.size, args.distance)

    results = spot_analysis(
        label_image,
        objects,
        groups,
        im1,
        label1,
        im2,
        label2,
        neighbour_distance=args.neighbour_distance,
    )

    formatted = format_spot_results(results, class_names=class_names)
    if args.spot_fn:
        save_csv(args.spot_fn, formatted)

    for r in formatted:
        print(r)

    summary = spot_summary(results, groups)
    formatted2 = format_summary_results(summary, class_names=class_names)
    if args.summary_fn:
        save_csv(args.summary_fn, formatted2)

    for r in formatted2:
        print(r)


if __name__ == "__main__":
    main()
