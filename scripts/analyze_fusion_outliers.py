#!/usr/bin/env python3
"""Trace robust fused-cloud outliers back to per-view seed depth maps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_fused_clouds import read_ply


def read_colite_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        magic = stream.readline().decode("ascii").split()
        if magic != ["COLITE_DEPTH_MAP_BINARY", "1"]:
            raise ValueError(f"unsupported Colite depth map: {path}")
        width, height, _depth_min, _depth_max = stream.readline().split()
        values = np.fromfile(
            stream, dtype="<f4", count=int(width) * int(height)
        )
    return values.reshape((int(height), int(width)))


def read_colmap_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = b""
        while header.count(b"&") < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"invalid COLMAP depth map: {path}")
            header += byte
        width, height, channels = map(int, header.decode().split("&")[:3])
        values = np.fromfile(
            stream, dtype="<f4", count=width * height * channels
        )
    return values.reshape((height, width, channels), order="F")[:, :, 0]


def read_image_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for line in lines[::2]:
        fields = line.split()
        names[int(fields[0])] = fields[9]
    return names


def read_sparse_ranges(path: Path) -> dict[int, tuple[float, float, int]]:
    values: dict[int, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            depth = float(row["depth"])
            if np.isfinite(depth) and depth > 0:
                values[int(row["image_id"])].append(depth)
    result = {}
    for image_id, depths in values.items():
        array = np.asarray(depths, dtype=np.float64)
        result[image_id] = (
            float(np.quantile(array, 0.01)),
            float(np.quantile(array, 0.99)),
            len(depths),
        )
    return result


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colite-ply", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--images-txt", required=True, type=Path)
    parser.add_argument("--sparse-depths", required=True, type=Path)
    parser.add_argument("--colite-depth-dir", required=True, type=Path)
    parser.add_argument("--colmap-depth-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--quantile", type=float, default=0.001)
    args = parser.parse_args()

    xyz = np.asarray(read_ply(args.colite_ply), dtype=np.float64)
    lower = np.quantile(xyz, args.quantile, axis=0)
    upper = np.quantile(xyz, 1.0 - args.quantile, axis=0)
    is_outlier = np.any((xyz < lower) | (xyz > upper), axis=1)

    image_names = read_image_names(args.images_txt)
    sparse_ranges = read_sparse_ranges(args.sparse_depths)
    depth_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    by_image: dict[int, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "outlier_count": 0,
            "inlier_count": 0,
            "outlier_colite_depth": [],
            "outlier_colmap_depth": [],
            "outlier_track_length": [],
            "inlier_colite_depth": [],
            "inlier_colmap_depth": [],
            "inlier_track_length": [],
        }
    )
    all_values = {
        "outlier_colite_depth": [],
        "outlier_colmap_depth": [],
        "outlier_track_length": [],
        "inlier_colite_depth": [],
        "inlier_colmap_depth": [],
        "inlier_track_length": [],
    }

    with args.tracks.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            point_idx = int(row["point_idx"])
            image_id = int(row["seed_image_id"])
            if image_id not in depth_cache:
                name = image_names[image_id]
                depth_cache[image_id] = (
                    read_colite_depth(args.colite_depth_dir / f"{name}.bin"),
                    read_colmap_depth(
                        args.colmap_depth_dir / f"{name}.geometric.bin"
                    ),
                )
            colite_depth, colmap_depth = depth_cache[image_id]
            y = int(row["seed_row"])
            x = int(row["seed_col"])
            candidate = float(colite_depth[y, x])
            reference = float(colmap_depth[y, x])
            reference = reference if np.isfinite(reference) and reference > 0 else np.nan
            prefix = "outlier" if is_outlier[point_idx] else "inlier"
            group = by_image[image_id]
            group[f"{prefix}_count"] += 1
            for target, value in (
                (f"{prefix}_colite_depth", candidate),
                (f"{prefix}_colmap_depth", reference),
                (f"{prefix}_track_length", float(row["num_observations"])),
            ):
                # Bound inlier diagnostics while retaining all rare outliers.
                if prefix == "outlier" or len(all_values[target]) < 200_000:
                    all_values[target].append(value)
                    group[target].append(value)

    rows = []
    for image_id in sorted(by_image):
        group = by_image[image_id]
        sparse = sparse_ranges.get(image_id, (np.nan, np.nan, 0))
        outlier_colmap = np.asarray(group["outlier_colmap_depth"])
        rows.append(
            {
                "image_id": image_id,
                "name": image_names[image_id],
                "outliers": group["outlier_count"],
                "inliers": group["inlier_count"],
                "outlier_colmap_valid": int(np.isfinite(outlier_colmap).sum()),
                "outlier_colite_depth_median": finite_summary(
                    group["outlier_colite_depth"]
                ).get("median", np.nan),
                "outlier_colmap_depth_median": finite_summary(
                    group["outlier_colmap_depth"]
                ).get("median", np.nan),
                "sparse_depth_p01": sparse[0],
                "sparse_depth_p99": sparse[1],
                "sparse_samples": sparse[2],
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "num_points": int(xyz.shape[0]),
        "outlier_quantile": args.quantile,
        "robust_lower_xyz": lower.tolist(),
        "robust_upper_xyz": upper.tolist(),
        "num_outliers": int(is_outlier.sum()),
        "outlier_ratio": float(is_outlier.mean()),
        "metrics": {
            key: finite_summary(values) for key, values in all_values.items()
        },
        "per_image_tsv": str(args.output_tsv),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
