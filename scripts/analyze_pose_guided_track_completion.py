#!/usr/bin/env python3
"""Estimate conservative final-pose guided sparse-track completion."""

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import project, read_model


HEADER = struct.Struct("<IIQQQ")


def read_features(path: Path):
    data = path.read_bytes()
    magic, version, count, rows, cols = HEADER.unpack_from(data)
    if magic != 0x434C4654 or version != 1 or count != rows or cols != 128:
        raise ValueError(f"Unsupported feature file: {path}")
    keypoint_offset = HEADER.size
    descriptor_offset = keypoint_offset + count * 20
    keypoints = np.frombuffer(
        data, dtype="<f4", count=count * 5, offset=keypoint_offset
    ).reshape((count, 5))[:, :2].copy()
    descriptors = np.frombuffer(
        data, dtype="<f4", count=rows * cols, offset=descriptor_offset
    ).reshape((rows, cols)).copy()
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors /= np.maximum(norms, 1e-12)
    return keypoints, descriptors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--max_reprojection", type=float, default=4.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    cameras, images, points = read_model(args.model)
    feature_data = {
        image_id: read_features(args.workspace / "features" / f"{image_id}.bin")
        for image_id in images
    }
    assigned = {
        image_id: np.full(len(feature_data[image_id][0]), -1, dtype=np.int64)
        for image_id in images
    }
    for point_id, (_, track) in points.items():
        for image_id, point2d_idx in track:
            assigned[image_id][point2d_idx] = point_id

    cell_size = args.max_reprojection
    grids = {}
    for image_id, (keypoints, _) in feature_data.items():
        grid = defaultdict(list)
        for index, xy in enumerate(keypoints):
            if assigned[image_id][index] >= 0:
                continue
            grid[tuple(np.floor(xy / cell_size).astype(np.int64))].append(index)
        grids[image_id] = grid

    candidates = []
    projected_pairs = 0
    spatial_candidates = 0
    for point_id, (xyz, track) in points.items():
        if len(track) < 2:
            continue
        observed_images = {image_id for image_id, _ in track}
        track_descriptors = np.stack(
            [feature_data[image_id][1][point2d_idx]
             for image_id, point2d_idx in track]
        )
        for image_id, image in images.items():
            if image_id in observed_images:
                continue
            predicted = project(cameras[image[2]], image, xyz)
            if predicted is None:
                continue
            projected_pairs += 1
            cell = np.floor(predicted / cell_size).astype(np.int64)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for feature_idx in grids[image_id].get(
                        (int(cell[0] + dx), int(cell[1] + dy)), ()
                    ):
                        error = float(np.linalg.norm(
                            feature_data[image_id][0][feature_idx] - predicted
                        ))
                        if error > args.max_reprojection:
                            continue
                        spatial_candidates += 1
                        descriptor = feature_data[image_id][1][feature_idx]
                        angles = np.arccos(np.clip(
                            track_descriptors @ descriptor, -1.0, 1.0
                        ))
                        second = float(np.partition(angles, 1)[1])
                        best = float(np.min(angles))
                        candidates.append(
                            (second, best, error, image_id, feature_idx,
                             point_id, len(track))
                        )

    candidates.sort()
    result = {
        "points": len(points),
        "projected_point_image_pairs": projected_pairs,
        "spatial_candidates": spatial_candidates,
        "thresholds": {},
    }
    for threshold in (0.3, 0.301, 0.305, 0.31, 0.4, 0.5, 0.6, 0.7):
        claimed_observations = set()
        claimed_point_images = set()
        selected = []
        for candidate in candidates:
            second, _, _, image_id, feature_idx, point_id, _ = candidate
            if second > threshold:
                break
            observation = (image_id, feature_idx)
            point_image = (point_id, image_id)
            if observation in claimed_observations or point_image in claimed_point_images:
                continue
            claimed_observations.add(observation)
            claimed_point_images.add(point_image)
            selected.append(candidate)
        result["thresholds"][str(threshold)] = {
            "accepted": len(selected),
            "reprojection_p50": (
                float(np.median([item[2] for item in selected]))
                if selected else None
            ),
            "second_descriptor_angle_p50": (
                float(np.median([item[0] for item in selected]))
                if selected else None
            ),
            "source_track_length_p50": (
                float(np.median([item[6] for item in selected]))
                if selected else None
            ),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
