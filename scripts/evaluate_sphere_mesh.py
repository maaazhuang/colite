#!/usr/bin/env python3
"""Evaluate a triangle PLY mesh against a unit-sphere ground truth."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_mesh_outputs import read_ply, topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--radius", type=float, default=1.0)
    args = parser.parse_args()

    reports = []
    for path in args.meshes:
        vertices, _, faces, _ = read_ply(path)
        triangles = vertices[np.asarray(faces, dtype=np.int64)]
        centroids = np.mean(triangles, axis=1)
        radial_error = np.abs(np.linalg.norm(centroids, axis=1) - args.radius)
        edge_lengths = np.linalg.norm(
            triangles[:, [0, 1, 2]] - triangles[:, [1, 2, 0]], axis=2
        )
        mesh_topology = topology(vertices, faces)
        reports.append({
            "path": str(path),
            "vertices": len(vertices),
            "faces": len(faces),
            "surface_area": mesh_topology["area_sum"],
            "nonmanifold_edges": mesh_topology["nonmanifold_edges"],
            "boundary_edges": mesh_topology["boundary_edges"],
            "centroid_radial_error_median": float(np.median(radial_error)),
            "centroid_radial_error_p95": float(np.percentile(radial_error, 95)),
            "max_edge_length": float(np.max(edge_lengths)),
        })
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
