#!/usr/bin/env python3
"""Evaluate forward-backward depth-map consistency on a COLMAP text model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def quaternion_matrix(values):
    w, x, y, z = np.asarray(values, dtype=np.float64)
    scale = np.linalg.norm((w, x, y, z))
    w, x, y, z = (w / scale, x / scale, y / scale, z / scale)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_model(path):
    cameras = {}
    for line in (path / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "PINHOLE":
            raise ValueError(f"Only PINHOLE cameras are supported, got {fields[1]}")
        cameras[int(fields[0])] = {
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": np.asarray(fields[4:8], dtype=np.float64),
        }
    images = {}
    lines = [
        line for line in (path / "images.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        image_id = int(fields[0])
        images[image_id] = {
            "rotation": quaternion_matrix(fields[1:5]),
            "translation": np.asarray(fields[5:8], dtype=np.float64),
            "camera": cameras[int(fields[8])],
            "name": fields[9],
        }
    return images


def read_colite(path):
    with path.open("rb") as stream:
        if stream.readline().decode().split() != ["COLITE_DEPTH_MAP_BINARY", "1"]:
            raise ValueError(f"Unsupported Colite map: {path}")
        width, height, _, _ = stream.readline().decode().split()
        return np.frombuffer(stream.read(), dtype="<f4").reshape(
            int(height), int(width)
        )


def read_colmap(path):
    with path.open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"Invalid COLMAP map: {path}")
            header.extend(byte)
        width, height, channels = map(int, header.decode().split("&")[:3])
        data = np.frombuffer(stream.read(), dtype="<f4")
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2)[
        :, :, 0
    ]


def scaled_intrinsics(image, shape):
    height, width = shape
    camera = image["camera"]
    fx, fy, cx, cy = camera["params"]
    sx = width / camera["width"]
    sy = height / camera["height"]
    return fx * sx, fy * sy, cx * sx, cy * sy


def bilinear(depth, x, y):
    height, width = depth.shape
    valid_xy = (
        np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        & (x <= width - 1) & (y <= height - 1)
    )
    x_safe = np.clip(np.nan_to_num(x), 0, width - 1)
    y_safe = np.clip(np.nan_to_num(y), 0, height - 1)
    x0 = np.floor(x_safe).astype(np.int32)
    y0 = np.floor(y_safe).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x_safe - x0
    wy = y_safe - y0
    values = (
        (1 - wx) * (1 - wy) * depth[y0, x0]
        + wx * (1 - wy) * depth[y0, x1]
        + (1 - wx) * wy * depth[y1, x0]
        + wx * wy * depth[y1, x1]
    )
    values[~valid_xy] = 0
    values[~np.isfinite(values)] = 0
    return values


def consistency_count(reference, source, reference_depth, source_depth, threshold):
    height, width = reference_depth.shape
    ref_fx, ref_fy, ref_cx, ref_cy = scaled_intrinsics(reference, (height, width))
    src_fx, src_fy, src_cx, src_cy = scaled_intrinsics(source, source_depth.shape)
    rows, cols = np.indices((height, width), dtype=np.float64)
    valid_ref = np.isfinite(reference_depth) & (reference_depth > 0)
    z_ref = reference_depth.astype(np.float64)
    points_ref = np.stack(
        ((cols - ref_cx) / ref_fx * z_ref,
         (rows - ref_cy) / ref_fy * z_ref,
         z_ref),
        axis=0,
    ).reshape(3, -1)
    rotation_ref = reference["rotation"]
    translation_ref = reference["translation"][:, None]
    world = rotation_ref.T @ (points_ref - translation_ref)
    points_src = source["rotation"] @ world + source["translation"][:, None]
    src_z = points_src[2]
    src_x = src_fx * points_src[0] / src_z + src_cx
    src_y = src_fy * points_src[1] / src_z + src_cy
    sampled_depth = bilinear(source_depth, src_x, src_y)
    points_src_back = np.stack(
        ((src_x - src_cx) / src_fx * sampled_depth,
         (src_y - src_cy) / src_fy * sampled_depth,
         sampled_depth),
        axis=0,
    )
    world_back = source["rotation"].T @ (
        points_src_back - source["translation"][:, None]
    )
    points_ref_back = rotation_ref @ world_back + translation_ref
    back_z = points_ref_back[2]
    back_x = ref_fx * points_ref_back[0] / back_z + ref_cx
    back_y = ref_fy * points_ref_back[1] / back_z + ref_cy
    error = np.hypot(back_x - cols.ravel(), back_y - rows.ravel())
    valid = (
        valid_ref.ravel() & (src_z > 0) & (sampled_depth > 0) & (back_z > 0)
        & np.isfinite(error) & (error <= threshold)
    )
    return valid.reshape(height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--source_images_tsv", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--format", choices=("colite", "colmap"), required=True)
    parser.add_argument("--colmap_suffix", default="geometric")
    parser.add_argument("--max_reprojection_error", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = read_model(args.model_path)
    sources = {}
    with args.source_images_tsv.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sources.setdefault(int(row["target_image_id"]), []).append(
                int(row["source_image_id"])
            )
    reader = read_colite if args.format == "colite" else read_colmap
    maps = {}
    for image_id, image in images.items():
        suffix = ".bin" if args.format == "colite" else f".{args.colmap_suffix}.bin"
        maps[image_id] = reader(args.depth_dir / f"{image['name']}{suffix}")

    total_pixels = 0
    total_valid = 0
    total_one = 0
    total_two = 0
    per_image = []
    for image_id in sorted(images):
        depth = maps[image_id]
        counts = np.zeros(depth.shape, dtype=np.uint8)
        for source_id in sources.get(image_id, []):
            counts += consistency_count(
                images[image_id], images[source_id], depth, maps[source_id],
                args.max_reprojection_error,
            )
        valid = np.isfinite(depth) & (depth > 0)
        pixels = depth.size
        valid_count = int(valid.sum())
        one = int((valid & (counts >= 1)).sum())
        two = int((valid & (counts >= 2)).sum())
        total_pixels += pixels
        total_valid += valid_count
        total_one += one
        total_two += two
        per_image.append({
            "image_id": image_id,
            "name": images[image_id]["name"],
            "valid_ratio": valid_count / pixels,
            "consistent_1_ratio": one / pixels,
            "consistent_2_ratio": two / pixels,
        })
    result = {
        "format": args.format,
        "num_images": len(per_image),
        "max_reprojection_error": args.max_reprojection_error,
        "valid_ratio": total_valid / total_pixels,
        "consistent_1_ratio": total_one / total_pixels,
        "consistent_2_ratio": total_two / total_pixels,
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
