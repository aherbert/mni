#!/usr/bin/env python3
"""Program to create reports on the MNi analysis results."""

import argparse


def main() -> None:
    """Program to create reports on the MNi analysis results."""
    parser = argparse.ArgumentParser(
        description="""Program to create reports on the MNi analysis results.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _ = parser.add_argument(
        "files",
        nargs="+",
        help="spot.csv, summary.csv, or result directory",
        metavar="csv/dir",
    )

    group = parser.add_argument_group("Report Options")
    _ = group.add_argument(
        "--report",
        nargs="+",
        type=int,
        default=[],
        metavar="n",
        help="""Report (default is all)

1: Class count
2: Class and spot count
3: Class and spot channel count
4: Class and spot overlaps above iou parameter
5: Class and spot neighbours below distance parameter
6: Class and spots above manders parameter
""",
    )
    _ = group.add_argument(
        "--distance",
        type=float,
        default=10.0,
        help="Neighbour distance (default: %(default)s)",
    )
    _ = group.add_argument(
        "--iou",
        type=float,
        default=0.1,
        help="Intersecion-over-union threshold (default: %(default)s)",
    )
    _ = group.add_argument(
        "--manders",
        type=float,
        default=0.2,
        help="Manders threshold threshold (default: %(default)s)",
    )
    _ = group.add_argument(
        "--out",
        help="Output directory (default is first argument directory, else '.')",
    )
    _ = group.add_argument(
        "--tablefmt",
        default="psql",
        choices=[
            "plain",
            "simple",
            "github",
            "grid",
            "simple_grid",
            "rounded_grid",
            "heavy_grid",
            "mixed_grid",
            "double_grid",
            "fancy_grid",
            "outline",
            "simple_outline",
            "rounded_outline",
            "heavy_outline",
            "mixed_outline",
            "double_outline",
            "fancy_outline",
            "pipe",
            "orgtbl",
            "asciidoc",
            "jira",
            "presto",
            "pretty",
            "psql",
            "rst",
            "mediawiki",
            "moinmoin",
            "youtrack",
            "html",
            "unsafehtml",
            "latex",
            "latex_raw",
            "latex_booktabs",
            "latex_longtable",
            "textile",
            "tsv",
        ],
        help="tabulate table format (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging
    import os

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("Reading data...")

    from mni.reports import create_report, load_data, number_of_reports

    summary_df, spot_df = load_data(args.files)
    logger.info(
        "Summary records = %d; Spot records = %d",
        len(summary_df),
        len(spot_df),
    )

    if args.out:
        out = args.out
    else:
        out = "."
        for path in args.files:
            if os.path.isdir(path):
                out = path
                break
    logger.info("Saving reports to %s", out)
    os.makedirs(out, exist_ok=True)

    report_ids = (
        set(args.report)
        if args.report
        else list(range(1, 1 + number_of_reports()))
    )
    for report_id in report_ids:
        df, title = create_report(
            summary_df,
            spot_df,
            report_id,
            out=out,
            distance=args.distance,
            iou=args.iou,
            manders=args.manders,
        )
        if df.empty:
            logger.warning("No data for report number: %d", report_id)
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info("Report %d: %s", report_id, title)
                print(df.to_markdown(index=False, tablefmt=args.tablefmt))

    logger.info("Done")


if __name__ == "__main__":
    main()
