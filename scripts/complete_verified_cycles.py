#!/usr/bin/env python3
"""Add conservative descriptor/geometric verified edges backed by 2-hop cycles."""

import argparse
import json
import math
import struct
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import project, read_model


def read_features(path: Path):
    with path.open("rb") as stream:
        magic, version, num_keypoints, rows, cols = struct.unpack(
            "<IIQQQ", stream.read(32)
        )
        if magic != 0x434C4654 or version != 1:
            raise ValueError(f"Unsupported feature file: {path}")
        keypoints = np.frombuffer(
            stream.read(num_keypoints * 5 * 4), dtype="<f4"
        ).reshape((num_keypoints, 5))
        descriptors = np.frombuffer(
            stream.read(rows * cols * 4), dtype="<f4"
        ).reshape((rows, cols))
    if rows != num_keypoints:
        raise ValueError(f"Keypoint/descriptor count mismatch: {path}")
    return keypoints, descriptors


def pixel_ray(camera, pixel):
    model, params = camera
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = params[:3]
        distorted = (np.asarray(pixel, dtype=np.float64) - (cx, cy)) / f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        distorted = (
            (float(pixel[0]) - cx) / fx,
            (float(pixel[1]) - cy) / fy,
        )
        distorted = np.asarray(distorted, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported diagnostic camera model: {model}")

    normalized = distorted.copy()
    if model == "SIMPLE_RADIAL":
        k1 = float(params[3])
        for _ in range(12):
            radius2 = float(normalized @ normalized)
            scale = 1.0 + k1 * radius2
            if abs(scale) < 1e-12:
                return None
            updated = distorted / scale
            if np.linalg.norm(updated - normalized) < 1e-12:
                normalized = updated
                break
            normalized = updated
    ray = np.asarray([normalized[0], normalized[1], 1.0])
    return ray / np.linalg.norm(ray)


def triangulate_candidate(cameras, images, keypoints, first, second):
    image1 = images[first[0]]
    image2 = images[second[0]]
    camera1 = cameras[image1[2]]
    camera2 = cameras[image2[2]]
    ray1 = pixel_ray(camera1, keypoints[first[0]][first[1], :2])
    ray2 = pixel_ray(camera2, keypoints[second[0]][second[1], :2])
    if ray1 is None or ray2 is None:
        return None
    center1 = -image1[0].T @ image1[1]
    center2 = -image2[0].T @ image2[1]
    world_ray1 = image1[0].T @ ray1
    world_ray2 = image2[0].T @ ray2
    system = np.column_stack((world_ray1, -world_ray2))
    depths, _, _, _ = np.linalg.lstsq(system, center2 - center1, rcond=None)
    if depths[0] <= 0.0 or depths[1] <= 0.0:
        return None
    xyz = 0.5 * (
        center1 + depths[0] * world_ray1
        + center2 + depths[1] * world_ray2
    )
    predicted1 = project(camera1, image1, xyz)
    predicted2 = project(camera2, image2, xyz)
    if predicted1 is None or predicted2 is None:
        return None
    return max(
        float(np.linalg.norm(predicted1 - keypoints[first[0]][first[1], :2])),
        float(np.linalg.norm(predicted2 - keypoints[second[0]][second[1], :2])),
    )


def descriptor_angle(descriptors, first, second):
    descriptor1 = descriptors[first[0]][first[1]].astype(np.float64)
    descriptor2 = descriptors[second[0]][second[1]].astype(np.float64)
    denominator = np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2)
    if denominator <= 0.0:
        return math.inf
    cosine = float(np.clip(descriptor1 @ descriptor2 / denominator, -1.0, 1.0))
    return math.acos(cosine)


def canonical_edge(first, second):
    return (first, second) if first[0] < second[0] else (second, first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--max_center_degree", type=int, default=32)
    parser.add_argument("--max_descriptor_angle", type=float, default=0.7)
    parser.add_argument("--max_reprojection_error", type=float, default=2.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    verified = args.workspace / "matches" / "verified_matches.tsv"
    header = []
    rows = []
    adjacency = defaultdict(set)
    pair_metadata = {}
    used_first = defaultdict(set)
    used_second = defaultdict(set)
    existing = set()
    with verified.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                header.append(line)
                continue
            fields = line.split()
            pair = (int(fields[0]), int(fields[1]))
            first = (pair[0], int(fields[11]))
            second = (pair[1], int(fields[12]))
            rows.append(line)
            pair_metadata.setdefault(pair, fields[:11])
            used_first[pair].add(first[1])
            used_second[pair].add(second[1])
            edge = canonical_edge(first, second)
            existing.add(edge)
            adjacency[first].add(second)
            adjacency[second].add(first)

    support = defaultdict(set)
    skipped_high_degree_centers = 0
    for center, neighbors in adjacency.items():
        if len(neighbors) > args.max_center_degree:
            skipped_high_degree_centers += 1
            continue
        for first, second in combinations(sorted(neighbors), 2):
            if first[0] == second[0]:
                continue
            first, second = canonical_edge(first, second)
            pair = (first[0], second[0])
            edge = (first, second)
            if pair not in pair_metadata or edge in existing:
                continue
            if first[1] in used_first[pair] or second[1] in used_second[pair]:
                continue
            support[edge].add(center)

    image_ids = sorted({image_id for edge in support for image_id, _ in edge})
    keypoints = {}
    descriptors = {}
    for image_id in image_ids:
        keypoints[image_id], descriptors[image_id] = read_features(
            args.workspace / "features" / f"{image_id}.bin"
        )
    cameras, images, _ = read_model(args.model)

    candidates = []
    rejected_descriptor = 0
    rejected_geometry = 0
    for (first, second), centers in support.items():
        if len(centers) < args.min_support:
            continue
        distance = descriptor_angle(descriptors, first, second)
        if distance > args.max_descriptor_angle:
            rejected_descriptor += 1
            continue
        reprojection_error = triangulate_candidate(
            cameras, images, keypoints, first, second
        )
        if (reprojection_error is None or
                reprojection_error > args.max_reprojection_error):
            rejected_geometry += 1
            continue
        candidates.append((
            -len(centers), reprojection_error, distance, first, second
        ))
    candidates.sort()

    additions = []
    for negative_support, error, distance, first, second in candidates:
        pair = (first[0], second[0])
        if first[1] in used_first[pair] or second[1] in used_second[pair]:
            continue
        used_first[pair].add(first[1])
        used_second[pair].add(second[1])
        additions.append((
            pair_metadata[pair], first[1], second[1], -negative_support,
            error, distance
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.writelines(header)
        stream.writelines(rows)
        for metadata, idx1, idx2, _, _, _ in additions:
            stream.write("\t".join(metadata + [str(idx1), str(idx2)]) + "\n")

    result = {
        "input_edges": len(rows),
        "two_hop_candidates": len(support),
        "skipped_high_degree_centers": skipped_high_degree_centers,
        "descriptor_rejections": rejected_descriptor,
        "geometry_rejections": rejected_geometry,
        "eligible_candidates": len(candidates),
        "added_edges": len(additions),
        "output_edges": len(rows) + len(additions),
        "added_support_histogram": {
            str(value): sum(1 for item in additions if item[3] == value)
            for value in sorted({item[3] for item in additions})
        },
        "max_added_reprojection_error": (
            max(item[4] for item in additions) if additions else None
        ),
        "max_added_descriptor_angle": (
            max(item[5] for item in additions) if additions else None
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
