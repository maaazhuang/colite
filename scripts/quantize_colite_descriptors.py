#!/usr/bin/env python3
"""Quantize Colite float SIFT descriptors to COLMAP uint8/512 semantics."""

import argparse
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<IIQQQ")
KEYPOINT_BYTES = 5 * 4
MAGIC = 0x434C4654


def quantize(path: Path) -> tuple[int, int]:
    data = bytearray(path.read_bytes())
    magic, version, num_keypoints, rows, cols = HEADER.unpack_from(data)
    if magic != MAGIC or version != 1 or rows != num_keypoints or cols != 128:
        raise ValueError(f"Unsupported Colite feature file: {path}")
    offset = HEADER.size + num_keypoints * KEYPOINT_BYTES
    expected = offset + rows * cols * 4
    if len(data) != expected:
        raise ValueError(f"Unexpected feature file size: {path}")
    descriptors = np.frombuffer(data, dtype="<f4", count=rows * cols,
                                offset=offset)
    descriptors[:] = np.clip(np.rint(descriptors * 512.0), 0, 255) / 512.0
    path.write_bytes(data)
    return int(rows), int(cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.workspace / "features").glob("*.bin"))
    total = 0
    for path in files:
        rows, _ = quantize(path)
        total += rows
    print(f"files={len(files)} descriptors={total}")


if __name__ == "__main__":
    main()
