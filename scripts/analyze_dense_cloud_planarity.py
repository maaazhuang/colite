#!/usr/bin/env python3
"""Measure deterministic local-plane residuals in an ASCII/binary PLY cloud."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_fused_clouds import read_ply, subsample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--neighbors", type=int, default=24)
    args = parser.parse_args()

    points = read_ply(args.input)
    samples = subsample(points, args.max_points)
    if len(points) < args.neighbors or args.neighbors < 3:
        raise ValueError("Point cloud is too small for the requested neighborhood")

    index = cv2.flann_Index(points.astype(np.float32),
                            dict(algorithm=1, trees=8))
    indices, _ = index.knnSearch(samples.astype(np.float32), args.neighbors,
                                 params=dict(checks=128))
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = np.maximum(eigenvalues.sum(axis=1), 1e-24)
    normalized_residual = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0) / scale)
    result = {
        "input": str(args.input),
        "points": int(len(points)),
        "samples": int(len(samples)),
        "neighbors": args.neighbors,
        "normalized_local_plane_residual": {
            "median": float(np.median(normalized_residual)),
            "p75": float(np.quantile(normalized_residual, 0.75)),
            "p90": float(np.quantile(normalized_residual, 0.90)),
            "p95": float(np.quantile(normalized_residual, 0.95)),
            "mean": float(np.mean(normalized_residual)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
