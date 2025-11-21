#!/usr/bin/env python3
"""Program to segment images using cellpose."""

import argparse


def main() -> None:
    """Program to segment images using cellpose."""
    parser = argparse.ArgumentParser(
        description="""Program to segment images using cellpose"""
    )

    _args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Done")


if __name__ == "__main__":
    main()
