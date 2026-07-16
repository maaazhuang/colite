#!/usr/bin/env python3
"""Compare ASCII/binary PLY point clouds with directed nearest-neighbor metrics."""

import argparse
import json
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
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
                    raise ValueError("List property in vertex element is unsupported")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
            elif line == "end_header":
                break
        if fmt == "ascii":
            rows = [stream.readline().decode().split() for _ in range(count)]
            values = np.asarray(rows, dtype=np.float64)
            names = [name for name, _ in properties]
            xyz = values[:, [names.index(axis) for axis in "xyz"]]
        elif fmt == "binary_little_endian":
            values = np.fromfile(stream, dtype=np.dtype(properties), count=count)
            xyz = np.column_stack([values[axis] for axis in "xyz"]).astype(np.float64)
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


def nearest_distances(queries, points, query_block=256, point_block=8192):
    result = np.full(len(queries), np.inf, dtype=np.float64)
    for query_begin in range(0, len(queries), query_block):
        query = queries[query_begin:query_begin + query_block]
        best = np.full(len(query), np.inf, dtype=np.float64)
        for point_begin in range(0, len(points), point_block):
            point = points[point_begin:point_begin + point_block]
            squared = np.sum((query[:, None, :] - point[None, :, :]) ** 2, axis=2)
            best = np.minimum(best, squared.min(axis=1))
        result[query_begin:query_begin + len(query)] = np.sqrt(best)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = read_ply(args.candidate)
    reference = read_ply(args.reference)
    diagonal = float(np.linalg.norm(reference.max(axis=0) - reference.min(axis=0)))
    candidate_to_reference = nearest_distances(candidate, reference)
    reference_to_candidate = nearest_distances(reference, candidate)
    result = {
        "candidate_points": int(len(candidate)),
        "reference_points": int(len(reference)),
        "reference_bbox_diagonal": diagonal,
        "candidate_to_reference": stats(candidate_to_reference, diagonal),
        "reference_to_candidate": stats(reference_to_candidate, diagonal),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
