#!/usr/bin/env python3
"""Program to analyse the spots between two channels within MNi objects."""

import argparse


def main() -> None:
    """Program to analyse the spots between two channels within MNi objects."""
    parser = argparse.ArgumentParser(
        description="""Program to analyse the spots between two channels within MNi objects."""
    )
    _ = parser.add_argument("image", help="Image (CYX) (TIFF)")
    _ = parser.add_argument(
        "--object-ch",
        default=3,
        type=int,
        help="Object channel (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--spot-ch1",
        default=1,
        type=int,
        help="Spot channel 1 (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--spot-ch2",
        default=2,
        type=int,
        help="Spot channel 2 (default: %(default)s)",
    )

    group = parser.add_argument_group("Object Options")
    _ = group.add_argument(
        "--model-type",
        default="cpsam",
        help="Name of default model (default: %(default)s)",
    )
    _ = group.add_argument(
        "--diameter",
        type=float,
        default=100,
        help="Expected nuclei diameter (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--device",
        type=str,
        help="Torch device name (default: auto-detect)",
    )
    _ = group.add_argument(
        "--border",
        default=20,
        type=int,
        help="Border to exclude objects (pixels; negative to disable) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--dilation",
        default=0,
        type=int,
        help="Dilation applied to objects (pixels; use to increase object size) (default: %(default)s)",
    )

    group = parser.add_argument_group("Threshold Options")
    _ = group.add_argument(
        "--sigma",
        default=1.5,
        type=float,
        help="Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--sigma2",
        default=30,
        type=float,
        help="Background Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--method",
        default="mean_plus_std_q",
        choices=["mean_plus_std", "mean_plus_std_q", "otsu", "yen", "minimum"],
        help="Thresholding method (default: %(default)s)",
    )
    _ = group.add_argument(
        "--std",
        default=7,
        type=float,
        help="Std.dev above the mean (default: %(default)s)",
    )
    _ = group.add_argument(
        "--quantile",
        default=0.75,
        type=float,
        help="Quantile for lowest value used in mean_plus_std_q (default: %(default)s)",
    )
    _ = group.add_argument(
        "--fill-holes",
        default=2,
        type=int,
        help="Remove contiguous holes smaller than the specified size (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--min-spot-size",
        default=4,
        type=int,
        help="Minimum spot size (pixels) (default: %(default)s)",
    )

    group = parser.add_argument_group("Micro-nuclei Options")
    _ = group.add_argument(
        "--distance",
        default=2,
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

    group = parser.add_argument_group("View Options")
    _ = group.add_argument(
        "--view",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show results in graphical viewer",
    )
    _ = group.add_argument(
        "--channel-names", nargs="+", default=[], help="Channel names"
    )
    _ = group.add_argument(
        "--visible-channels",
        nargs="+",
        type=int,
        default=[],
        help="Visible channels (default is the spot channels)",
    )
    _ = group.add_argument(
        "--upper-limit",
        default=99.999,
        type=float,
        help="Upper contrast limit (percentile) (default: %(default)s)",
    )

    group = parser.add_argument_group("Other Options")
    _ = group.add_argument(
        "--repeat",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Repeat from stage: 1=object segmentation; 2=spot identification; 3=analysis",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("Initialising")

    import json
    import os
    from collections import Counter
    from typing import Any

    import czifile
    import numpy as np
    import numpy.typing as npt
    from tifffile import imread, imwrite

    from mni.utils import (
        analyse_objects,
        classify_objects,
        collate_groups,
        filter_method,
        find_micronuclei,
        find_objects,
        format_spot_results,
        format_summary_results,
        object_threshold,
        save_csv,
        spot_analysis,
        spot_summary,
        threshold_method,
    )

    base, suffix = os.path.splitext(args.image)
    if args.image.endswith("czi"):
        image = czifile.imread(args.image)
        # CZI file may have CZXT format
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        suffix = ".tiff"
    else:
        image = imread(args.image)
    if image.ndim != 3 or np.argmin(image.shape) != 0:
        raise RuntimeError("Expected CYX image: " + str(image.shape))

    stage = args.repeat if args.repeat else 10

    label_fn = f"{base}.objects{suffix}"
    if stage <= 1 or not os.path.exists(label_fn):
        from mni.segmentation import segment
        from mni.utils import filter_segmentation

        label_image = segment(
            image[args.object_ch],
            args.model_type,
            args.diameter,
            device=args.device,
        )
        label_image, n_objects = filter_segmentation(
            label_image, border=args.border
        )

        # Dilation is saved into the mask
        if args.dilation > 0:
            logger.info("Applying dilation %d", args.dilation)
            import skimage

            footprint = skimage.morphology.disk(
                args.dilation, decomposition="sequence"
            )
            dilated = skimage.morphology.dilation(
                label_image, footprint=footprint
            )
            dilated[label_image != 0] = 0  # Do not overwrite neighbours
            label_image += dilated

        imwrite(label_fn, label_image, compression="zlib")
    else:
        label_image = imread(label_fn)
        n_objects = np.max(label_image)

    logger.info("Identified %d objects: %s", n_objects, label_fn)

    # Spot identification
    std = args.std
    if args.method == "mean_plus_std_q":
        # make std shift equivalent to get the same thresholding level if a normal distribution is truncated
        import scipy.stats

        m, v = scipy.stats.truncnorm(
            -10, scipy.stats.norm.ppf(args.quantile)
        ).stats("mv")
        std = (std - m) / np.sqrt(v)
        logger.info(
            "Adjusted threshold %s to %.3f using normal distribution truncated at cdf=%s",
            args.std,
            std,
            args.quantile,
        )
    fun = threshold_method(args.method, std=std, q=args.quantile)
    filter_fun = filter_method(args.sigma, args.sigma2)

    spot1_fn = f"{base}.spot1{suffix}"
    im1 = image[args.spot_ch1]
    if stage <= 2 or not os.path.exists(spot1_fn):
        label1 = object_threshold(
            filter_fun(im1),
            label_image,
            fun,
            fill_holes=args.fill_holes,
            min_size=args.min_spot_size,
        )
        imwrite(spot1_fn, label1, compression="zlib")
    else:
        label1 = imread(spot1_fn)
    logger.info(
        "Identified %d spots in channel %d: %s",
        np.max(label1),
        args.spot_ch1,
        spot1_fn,
    )

    spot2_fn = f"{base}.spot2{suffix}"
    im2 = image[args.spot_ch2]
    if stage <= 2 or not os.path.exists(spot2_fn):
        label2 = object_threshold(
            filter_fun(im2),
            label_image,
            fun,
            fill_holes=args.fill_holes,
            min_size=args.min_spot_size,
        )
        imwrite(spot2_fn, label2, compression="zlib")
    else:
        label2 = imread(spot2_fn)
    logger.info(
        "Identified %d spots in channel %d: %s",
        np.max(label2),
        args.spot_ch2,
        spot2_fn,
    )

    # Analysis cannot be loaded from previous results, just skip
    spot_fn = f"{base}.spots.csv"
    summary_fn = f"{base}.summary.csv"

    # Create an anlysis function to allow this to be repeated.
    # Settings used: distance, size, min_size, neighbour_distance.
    # These could be added as input for the function and updated through the GUI.
    def analysis_fun(
        label_image: npt.NDArray[Any],
        label1: npt.NDArray[Any],
        label2: npt.NDArray[Any],
    ) -> None:
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
        logger.info(
            "Classified objects: %s", dict(Counter(class_names.values()))
        )

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
        logger.info("Saving spot results: %s", spot_fn)
        save_csv(spot_fn, formatted)

        o_data = analyse_objects(image[args.object_ch], label_image, objects)
        # Collate to ID: (size, intensity, cx, cy)
        object_data = {
            x[0]: (x[1],) + y for x, y in zip(objects, o_data, strict=True)
        }

        summary = spot_summary(results, groups)
        formatted2 = format_summary_results(
            summary, class_names=class_names, object_data=object_data
        )
        logger.info("Saving summary results: %s", summary_fn)
        save_csv(summary_fn, formatted2)

    if stage <= 3 or not (
        os.path.exists(spot_fn) and os.path.exists(summary_fn)
    ):
        analysis_fun(label_image, label1, label2)
    else:
        logger.info("Existing spot results: %s", spot_fn)
        logger.info("Existing summary results: %s", summary_fn)

    # Save settings if all OK
    fn = f"{base}.settings.json"
    logger.info("Saving settings: %s", fn)
    with open(fn, "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.view:
        logger.info("Launching viewer")
        import pandas as pd
        from skimage.measure import label

        from mni.gui import add_analysis_function, create_viewer, show_viewer
        from mni.utils import compact_mask

        label_df = pd.read_csv(summary_fn)
        spot_df = pd.read_csv(spot_fn)

        visible_channels = (
            args.visible_channels
            if args.visible_channels
            else [args.spot_ch1, args.spot_ch2]
        )
        viewer = create_viewer(
            image,
            label_image,
            label1,
            label2,
            channel_names=args.channel_names,
            visible_channels=visible_channels,
            label_df=label_df,
            spot_df=spot_df,
            upper_limit=args.upper_limit,
        )

        # Allow recomputation of features
        def redo_analysis_fun(
            label_image: npt.NDArray[Any],
            label1: npt.NDArray[Any],
            label2: npt.NDArray[Any],
        ) -> tuple[
            npt.NDArray[Any] | None,
            npt.NDArray[Any] | None,
            npt.NDArray[Any] | None,
            pd.DataFrame | None,
            pd.DataFrame | None,
        ]:
            logger.info("Repeating analysis")
            # Manual editing may duplicate label IDs
            label_image, m = label(label_image, return_num=True)
            label1, m1 = label(label1, return_num=True)
            label2, m2 = label(label2, return_num=True)
            label_image = compact_mask(label_image, m=m)
            label1 = compact_mask(label1, m=m1)
            label2 = compact_mask(label2, m=m2)

            # Save new labels
            imwrite(label_fn, label_image, compression="zlib")
            imwrite(spot1_fn, label1, compression="zlib")
            imwrite(spot2_fn, label2, compression="zlib")

            analysis_fun(label_image, label1, label2)

            # Reload analysis
            label_df = pd.read_csv(summary_fn)
            spot_df = pd.read_csv(spot_fn)

            return label_image, label1, label2, label_df, spot_df

        add_analysis_function(viewer, redo_analysis_fun)

        show_viewer(viewer)

    logger.info("Done")


if __name__ == "__main__":
    main()
