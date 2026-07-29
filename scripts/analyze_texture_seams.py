#!/usr/bin/env python3
"""Measure color discontinuity across texture-atlas seams in a textured OBJ."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_obj(path):
    uvs = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("vt "):
            uvs.append(tuple(map(float, line.split()[1:3])))
        elif line.startswith("f "):
            face = []
            for item in line.split()[1:]:
                fields = item.split("/")
                face.append((int(fields[0]) - 1, int(fields[1]) - 1))
            if len(face) == 3:
                faces.append(face)
    return np.asarray(uvs, dtype=np.float64), faces


def bilinear_sample(image, uv):
    height, width = image.shape[:2]
    x = np.clip(uv[0] * (width - 1), 0.0, width - 1.0)
    y = np.clip((1.0 - uv[1]) * (height - 1), 0.0, height - 1.0)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    return (
        (1.0 - dy) * ((1.0 - dx) * image[y0, x0] + dx * image[y0, x1])
        + dy * ((1.0 - dx) * image[y1, x0] + dx * image[y1, x1])
    )


def inside_edge_samples(face, edge_vertices, uvs):
    uv_by_vertex = {vertex: uvs[uv] for vertex, uv in face}
    opposite_vertex = next(
        vertex for vertex, _ in face if vertex not in edge_vertices)
    first = uv_by_vertex[edge_vertices[0]]
    second = uv_by_vertex[edge_vertices[1]]
    opposite = uv_by_vertex[opposite_vertex]
    return [
        0.94 * first + 0.03 * second + 0.03 * opposite,
        0.485 * first + 0.485 * second + 0.03 * opposite,
        0.03 * first + 0.94 * second + 0.03 * opposite,
    ]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def analyze(obj_path, texture_path):
    uvs, faces = read_obj(obj_path)
    image = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read texture: {texture_path}")
    image = image.astype(np.float64)

    edge_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for edge in range(3):
            first = face[edge][0]
            second = face[(edge + 1) % 3][0]
            edge_faces[tuple(sorted((first, second)))].append(face_id)

    seam_edges = 0
    same_chart_edges = 0
    color_errors = []
    for edge_vertices, adjacent in edge_faces.items():
        if len(adjacent) != 2:
            continue
        left = faces[adjacent[0]]
        right = faces[adjacent[1]]
        left_uv = {vertex: uvs[uv] for vertex, uv in left}
        right_uv = {vertex: uvs[uv] for vertex, uv in right}
        if all(np.linalg.norm(left_uv[vertex] - right_uv[vertex]) <= 1e-8
               for vertex in edge_vertices):
            same_chart_edges += 1
            continue
        seam_edges += 1
        left_samples = inside_edge_samples(left, edge_vertices, uvs)
        right_samples = inside_edge_samples(right, edge_vertices, uvs)
        for left_sample, right_sample in zip(left_samples, right_samples):
            color_errors.append(float(np.linalg.norm(
                bilinear_sample(image, left_sample)
                - bilinear_sample(image, right_sample))))

    return {
        "obj": str(obj_path),
        "texture": str(texture_path),
        "faces": len(faces),
        "same_chart_manifold_edges": same_chart_edges,
        "seam_edges": seam_edges,
        "samples_per_seam": 3,
        "seam_color_error_bgr_l2": summarize(color_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.obj, args.texture)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
