#!/usr/bin/env python3
"""Compare two PINHOLE undistortion workspaces in a common camera frame."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_pinhole_camera(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera in {path}")
        return {
            "width": int(fields[2]),
            "height": int(fields[3]),
            "fx": float(fields[4]),
            "fy": float(fields[5]),
            "cx": float(fields[6]),
            "cy": float(fields[7]),
        }
    raise ValueError(f"No camera found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_images", type=Path, required=True)
    parser.add_argument("--reference_cameras", type=Path, required=True)
    parser.add_argument("--candidate_images", type=Path, required=True)
    parser.add_argument("--candidate_cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_pinhole_camera(args.reference_cameras)
    candidate = read_pinhole_camera(args.candidate_cameras)
    x, y = np.meshgrid(
        np.arange(reference["width"], dtype=np.float32),
        np.arange(reference["height"], dtype=np.float32),
    )
    normalized_x = (x - reference["cx"]) / reference["fx"]
    normalized_y = (y - reference["cy"]) / reference["fy"]
    map_x = candidate["fx"] * normalized_x + candidate["cx"]
    map_y = candidate["fy"] * normalized_y + candidate["cy"]
    common_mask = (
        (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= candidate["width"] - 1.0)
        & (map_y <= candidate["height"] - 1.0)
    )

    results = []
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    for reference_path in sorted(args.reference_images.iterdir()):
        if not reference_path.is_file():
            continue
        candidate_path = args.candidate_images / reference_path.name
        if not candidate_path.exists():
            continue
        reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        candidate_image = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
        if reference_image is None or candidate_image is None:
            continue
        if reference_image.shape[:2] != (
            reference["height"],
            reference["width"],
        ):
            raise ValueError(f"Unexpected reference size: {reference_path}")
        if candidate_image.shape[:2] != (
            candidate["height"],
            candidate["width"],
        ):
            raise ValueError(f"Unexpected candidate size: {candidate_path}")
        warped = cv2.remap(
            candidate_image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        difference = (
            warped.astype(np.float64) - reference_image.astype(np.float64)
        )[common_mask]
        squared = float(np.square(difference).sum())
        absolute = float(np.abs(difference).sum())
        count = int(difference.size)
        rmse = math.sqrt(squared / count)
        results.append(
            {
                "image": reference_path.name,
                "rmse": rmse,
                "mae": absolute / count,
                "psnr_db": 20.0 * math.log10(255.0 / max(rmse, 1e-12)),
            }
        )
        squared_error_sum += squared
        absolute_error_sum += absolute
        value_count += count

    if not results:
        raise RuntimeError("No common images found")
    aggregate_rmse = math.sqrt(squared_error_sum / value_count)
    report = {
        "reference_camera": reference,
        "candidate_camera": candidate,
        "common_pixel_ratio": float(common_mask.mean()),
        "num_images": len(results),
        "rmse": aggregate_rmse,
        "mae": absolute_error_sum / value_count,
        "psnr_db": 20.0
        * math.log10(255.0 / max(aggregate_rmse, 1e-12)),
        "images": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
