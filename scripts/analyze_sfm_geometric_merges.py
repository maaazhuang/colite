#!/usr/bin/env python3
"""Find conservative geometry-only sparse-point merge opportunities."""

import argparse
import itertools
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def quaternion_rotation(values: list[float]) -> np.ndarray:
    w, x, y, z = values
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_model(path: Path):
    cameras = {}
    with (path / "cameras.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            cameras[int(fields[0])] = (
                fields[1],
                np.asarray([float(value) for value in fields[4:]], dtype=np.float64),
            )

    images = {}
    with (path / "images.txt").open(encoding="utf-8") as stream:
        while True:
            header = stream.readline()
            if not header:
                break
            if not header.strip() or header.startswith("#"):
                continue
            fields = header.split()
            image_id = int(fields[0])
            rotation = quaternion_rotation([float(value) for value in fields[1:5]])
            translation = np.asarray([float(value) for value in fields[5:8]])
            camera_id = int(fields[8])
            points = stream.readline().split()
            xy = np.asarray(
                [[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 3)],
                dtype=np.float64,
            )
            images[image_id] = (rotation, translation, camera_id, xy)

    points = {}
    with (path / "points3D.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            point_id = int(fields[0])
            track = [
                (int(fields[i]), int(fields[i + 1]))
                for i in range(8, len(fields), 2)
            ]
            points[point_id] = (np.asarray(fields[1:4], dtype=np.float64), track)
    return cameras, images, points


def project(camera, image, xyz: np.ndarray) -> np.ndarray | None:
    model, params = camera
    rotation, translation, _, _ = image
    local = rotation @ xyz + translation
    if local[2] <= 0:
        return None
    x, y = local[:2] / local[2]
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        scale = 1.0 + k1 * (x * x + y * y)
        return np.asarray([f * scale * x + cx, f * scale * y + cy])
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return np.asarray([f * x + cx, f * y + cy])
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return np.asarray([fx * x + cx, fy * y + cy])
    raise ValueError(f"Unsupported diagnostic camera model: {model}")


def bounded_track_distance(adjacency, first_track, second_track, max_depth):
    targets = set(second_track)
    visited = set(first_track)
    frontier = deque((observation, 0) for observation in first_track)
    while frontier:
        observation, depth = frontier.popleft()
        if observation in targets:
            return depth
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(observation, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max_error", type=float, default=4.0)
    parser.add_argument("--radius_pixels", type=float, default=4.0)
    parser.add_argument("--verified", type=Path)
    parser.add_argument("--max_path_depth", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    cameras, images, points = read_model(args.model)
    adjacency = defaultdict(list)
    if args.verified:
        with args.verified.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                first = (int(fields[0]), int(fields[11]))
                second = (int(fields[1]), int(fields[12]))
                adjacency[first].append(second)
                adjacency[second].append(first)

    depths = []
    focals = []
    for xyz, track in points.values():
        for image_id, _ in track[:1]:
            image = images[image_id]
            depths.append(float((image[0] @ xyz + image[1])[2]))
            params = cameras[image[2]][1]
            focals.append(float(params[0]))
    radius = float(np.median(depths) / np.median(focals) * args.radius_pixels)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for point_id, (xyz, _) in points.items():
        cell = tuple(np.floor(xyz / radius).astype(np.int64))
        grid.setdefault(cell, []).append(point_id)

    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    tested = 0
    disjoint = 0
    accepted = []
    for point_id, (xyz, track) in points.items():
        cell = tuple(np.floor(xyz / radius).astype(np.int64))
        image_ids = {image_id for image_id, _ in track}
        for offset in offsets:
            neighbor = tuple(cell[i] + offset[i] for i in range(3))
            for other_id in grid.get(neighbor, []):
                if other_id <= point_id:
                    continue
                other_xyz, other_track = points[other_id]
                distance = float(np.linalg.norm(xyz - other_xyz))
                if distance > radius:
                    continue
                tested += 1
                if image_ids.intersection(image_id for image_id, _ in other_track):
                    continue
                disjoint += 1
                merged = (
                    len(track) * xyz + len(other_track) * other_xyz
                ) / (len(track) + len(other_track))
                max_error = 0.0
                valid = True
                for image_id, point2d_idx in track + other_track:
                    image = images[image_id]
                    predicted = project(cameras[image[2]], image, merged)
                    if predicted is None:
                        valid = False
                        break
                    error = float(np.linalg.norm(predicted - image[3][point2d_idx]))
                    max_error = max(max_error, error)
                    if max_error > args.max_error:
                        valid = False
                        break
                if valid:
                    path_distance = (
                        bounded_track_distance(
                            adjacency, track, other_track,
                            args.max_path_depth
                        )
                        if args.verified
                        else None
                    )
                    accepted.append(
                        (max_error, distance, point_id, other_id,
                         len(track), len(other_track), path_distance)
                    )
    accepted.sort()
    claimed = set()
    greedy = []
    for candidate in accepted:
        if candidate[2] in claimed or candidate[3] in claimed:
            continue
        claimed.update(candidate[2:4])
        greedy.append(candidate)
    greedy_by_path = {}
    if args.verified:
        for max_depth in range(1, args.max_path_depth + 1):
            claimed_by_path = set()
            selected = []
            for candidate in accepted:
                if candidate[6] is None or candidate[6] > max_depth:
                    continue
                if candidate[2] in claimed_by_path or candidate[3] in claimed_by_path:
                    continue
                claimed_by_path.update(candidate[2:4])
                selected.append(candidate)
            greedy_by_path[str(max_depth)] = len(selected)
    result = {
        "points": len(points),
        "median_depth": float(np.median(depths)),
        "median_focal": float(np.median(focals)),
        "search_radius": radius,
        "nearby_pairs": tested,
        "disjoint_view_pairs": disjoint,
        "accepted_pairs": len(accepted),
        "greedy_non_overlapping_pairs": len(greedy),
        "greedy_observations_preserved": int(
            sum(candidate[4] + candidate[5] for candidate in greedy)
        ),
        "accepted_max_error_p50": (
            float(np.median([candidate[0] for candidate in accepted]))
            if accepted else None
        ),
        "accepted_path_distance_histogram": (
            {
                str(depth): sum(1 for candidate in accepted
                                if candidate[6] == depth)
                for depth in range(1, args.max_path_depth + 1)
            }
            if args.verified else None
        ),
        "accepted_without_bounded_path": (
            sum(1 for candidate in accepted if candidate[6] is None)
            if args.verified else None
        ),
        "greedy_merges_by_max_path_depth": (
            greedy_by_path if args.verified else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
