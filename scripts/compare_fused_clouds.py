#!/usr/bin/env python3
"""Compare ASCII/binary PLY point clouds with directed nearest-neighbor metrics."""

import argparse
import io
import json
import struct
from pathlib import Path

import cv2
import numpy as np


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "float32": "<f4",
    "float64": "<f8",
}


def read_ply(path: Path):
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        count = 0
        properties = []
        in_vertices = False
        while True:
            line = stream.readline().decode().strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                fields = line.split()
                in_vertices = fields[1] == "vertex"
                if in_vertices:
                    count = int(fields[2])
            elif line.startswith("property ") and in_vertices:
                fields = line.split()
                if fields[1] == "list":
                    properties.append(
                        ("list", fields[4], PLY_TYPES[fields[2]],
                         PLY_TYPES[fields[3]])
                    )
                else:
                    properties.append(
                        ("scalar", fields[2], PLY_TYPES[fields[1]], None)
                    )
            elif line == "end_header":
                break
        if fmt == "ascii":
            xyz = np.empty((count, 3), dtype=np.float64)
            for row_index in range(count):
                tokens = iter(stream.readline().decode().split())
                row = {}
                for kind, name, value_type, item_type in properties:
                    if kind == "scalar":
                        row[name] = float(next(tokens))
                    else:
                        list_count = int(next(tokens))
                        for _ in range(list_count):
                            next(tokens)
                xyz[row_index] = [row[axis] for axis in "xyz"]
        elif fmt == "binary_little_endian":
            payload = io.BytesIO(stream.read())
            scalar_prefix = []
            first_list = len(properties)
            for index, prop in enumerate(properties):
                if prop[0] == "list":
                    first_list = index
                    break
                scalar_prefix.append((prop[1], prop[2]))
            prefix_struct = struct.Struct(
                "<" + "".join(np.dtype(prop[2]).char
                              for prop in properties[:first_list])
            )
            xyz = np.empty((count, 3), dtype=np.float64)
            xyz_indices = [
                [prop[1] for prop in properties[:first_list]].index(axis)
                for axis in "xyz"
            ]
            for row_index in range(count):
                scalar_values = prefix_struct.unpack(payload.read(prefix_struct.size))
                xyz[row_index] = [scalar_values[index] for index in xyz_indices]
                for kind, name, count_type, item_type in properties[first_list:]:
                    if kind != "list":
                        raise ValueError(
                            "Scalar property after list property is unsupported"
                        )
                    count_dtype = np.dtype(count_type)
                    list_count = np.frombuffer(
                        payload.read(count_dtype.itemsize),
                        dtype=count_dtype, count=1
                    )[0]
                    payload.seek(
                        int(list_count) * np.dtype(item_type).itemsize,
                        io.SEEK_CUR,
                    )
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return xyz


def stats(values, diagonal):
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "median_over_reference_bbox_diagonal": float(np.median(values) / diagonal),
        "p95_over_reference_bbox_diagonal": float(np.quantile(values, 0.95) / diagonal),
    }


def subsample(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def nearest_distances(queries, points):
    if not len(queries) or not len(points):
        return np.empty(0, dtype=np.float64)
    index = cv2.flann_Index(points.astype(np.float32),
                            dict(algorithm=1, trees=8))
    _, squared = index.knnSearch(queries.astype(np.float32), 1,
                                 params=dict(checks=128))
    return np.sqrt(np.maximum(squared[:, 0], 0.0)).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_points", type=int, default=50000,
                        help="Deterministic samples per cloud; 0 uses all")
    args = parser.parse_args()
    candidate = read_ply(args.candidate)
    reference = read_ply(args.reference)
    candidate_min = candidate.min(axis=0)
    candidate_max = candidate.max(axis=0)
    reference_min = reference.min(axis=0)
    reference_max = reference.max(axis=0)
    diagonal = float(np.linalg.norm(reference_max - reference_min))
    candidate_sample = subsample(candidate, args.max_points)
    reference_sample = subsample(reference, args.max_points)
    candidate_to_reference = nearest_distances(candidate_sample,
                                               reference_sample)
    reference_to_candidate = nearest_distances(reference_sample,
                                               candidate_sample)
    result = {
        "candidate_points": int(len(candidate)),
        "reference_points": int(len(reference)),
        "candidate_samples": int(len(candidate_sample)),
        "reference_samples": int(len(reference_sample)),
        "candidate_bounds": {
            "min": candidate_min.tolist(),
            "max": candidate_max.tolist(),
            "diagonal": float(np.linalg.norm(candidate_max - candidate_min)),
        },
        "reference_bounds": {
            "min": reference_min.tolist(),
            "max": reference_max.tolist(),
            "diagonal": diagonal,
        },
        "reference_bbox_diagonal": diagonal,
        "candidate_to_reference": stats(candidate_to_reference, diagonal),
        "reference_to_candidate": stats(reference_to_candidate, diagonal),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
