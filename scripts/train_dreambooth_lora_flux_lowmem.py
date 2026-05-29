#!/usr/bin/env python
"""Compatibility wrapper for the packaged FLUX low-memory trainer."""

from lora.trainers.flux_lowmem import main, parse_args


if __name__ == "__main__":
    main(parse_args())
