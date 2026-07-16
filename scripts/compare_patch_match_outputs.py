#!/usr/bin/env python3
"""Compare COLITE and COLMAP depth maps on a shared COLMAP text model."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_colite(path: Path):
    with path.open("rb") as stream:
        magic = stream.readline().decode().split()
        if magic != ["COLITE_DEPTH_MAP_BINARY", "1"]:
            raise ValueError(f"Unsupported COLITE map: {path}")
        width, height, depth_min, depth_max = stream.readline().decode().split()
        data = np.frombuffer(stream.read(), dtype="<f4")
    return data.reshape(int(height), int(width)), float(depth_min), float(depth_max)


def read_colmap(path: Path):
    with path.open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"Invalid COLMAP map: {path}")
            header.extend(byte)
        width, height, channels = map(int, header.decode().split("&")[:3])
        data = np.frombuffer(stream.read(), dtype="<f4")
    array = data.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    return array[:, :, 0]


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def read_model(model: Path):
    cameras = {}
    for line in (model / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            cameras[int(fields[0])] = (int(fields[2]), int(fields[3]))
    points = {}
    for line in (model / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
    images = {}
    lines = [line for line in (model / "images.txt").read_text().splitlines()
             if not line.startswith("#")]
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        samples = lines[index + 1].split() if index + 1 < len(lines) else []
        observations = []
        for offset in range(0, len(samples), 3):
            point_id = int(samples[offset + 2])
            if point_id >= 0 and point_id in points:
                observations.append((float(samples[offset]), float(samples[offset + 1]), point_id))
        images[fields[9]] = {
            "q": np.asarray(fields[1:5], dtype=np.float64),
            "t": np.asarray(fields[5:8], dtype=np.float64),
            "camera": cameras[int(fields[8])],
            "observations": observations,
        }
    return images, points


def quantiles(values):
    if not values:
        return {"count": 0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colite_depth_dir", type=Path, required=True)
    parser.add_argument("--colmap_depth_dir", type=Path, required=True)
    parser.add_argument("--colmap_suffix", default="photometric",
                        choices=("photometric", "geometric"))
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images, points = read_model(args.model_path)
    direct_errors = []
    sparse_errors = {"colite": [], "colmap": []}
    valid = {"colite": 0, "colmap": 0, "common": 0, "pixels": 0}
    per_image = []
    for name, image in sorted(images.items()):
        colite_path = args.colite_depth_dir / f"{name}.bin"
        colmap_path = args.colmap_depth_dir / f"{name}.{args.colmap_suffix}.bin"
        if not colite_path.exists() or not colmap_path.exists():
            continue
        colite, _, _ = read_colite(colite_path)
        colmap = read_colmap(colmap_path)
        if colite.shape != colmap.shape:
            raise ValueError(f"Map shape mismatch for {name}: {colite.shape} vs {colmap.shape}")
        colite_valid = np.isfinite(colite) & (colite > 0)
        colmap_valid = np.isfinite(colmap) & (colmap > 0)
        common = colite_valid & colmap_valid
        valid["pixels"] += colite.size
        valid["colite"] += int(colite_valid.sum())
        valid["colmap"] += int(colmap_valid.sum())
        valid["common"] += int(common.sum())
        if common.any():
            direct_errors.extend((np.abs(colite[common] - colmap[common]) /
                                  np.maximum(colmap[common], 1e-12)).tolist())

        rotation = quaternion_matrix(image["q"])
        camera_width, camera_height = image["camera"]
        height, width = colite.shape
        for x, y, point_id in image["observations"]:
            depth = float((rotation @ points[point_id] + image["t"])[2])
            if depth <= 0:
                continue
            col = min(width - 1, max(0, int(round((x + 0.5) * width / camera_width - 0.5))))
            row = min(height - 1, max(0, int(round((y + 0.5) * height / camera_height - 0.5))))
            for key, depth_map in (("colite", colite), ("colmap", colmap)):
                estimate = float(depth_map[row, col])
                if np.isfinite(estimate) and estimate > 0:
                    sparse_errors[key].append(abs(estimate - depth) / depth)
        per_image.append({
            "name": name,
            "shape": [height, width],
            "colite_valid_ratio": float(colite_valid.mean()),
            "colmap_valid_ratio": float(colmap_valid.mean()),
            "common_valid_ratio": float(common.mean()),
        })

    result = {
        "num_images": len(per_image),
        "valid_ratio": {key: valid[key] / valid["pixels"] for key in ("colite", "colmap", "common")},
        "common_depth_relative_difference": quantiles(direct_errors),
        "sparse_point_relative_depth_error": {
            "colite": quantiles(sparse_errors["colite"]),
            "colmap": quantiles(sparse_errors["colmap"]),
        },
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
