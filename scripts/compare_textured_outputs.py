#!/usr/bin/env python3
"""Compare COLITE OBJ and COLMAP ASCII PLY texture-mapping outputs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def uv_stats(uvs, faces):
    areas = []
    for face in faces:
        if len(face) != 3:
            continue
        points = np.asarray([uvs[index] for index in face], dtype=np.float64)
        areas.append(abs(float(np.cross(points[1] - points[0],
                                        points[2] - points[0]))) * 0.5)
    values = np.asarray(areas, dtype=np.float64)
    valid = values > 1e-12
    return ({"faces": int(len(values)), "textured_faces": int(valid.sum()),
             "textured_face_ratio": float(valid.mean()) if len(valid) else 0.0,
             "uv_area_sum": float(values.sum())},
            set(np.flatnonzero(valid).tolist()))


def read_obj(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    uvs = [tuple(map(float, line.split()[1:3]))
           for line in lines if line.startswith("vt ")]
    faces = []
    for line in lines:
        if line.startswith("f "):
            faces.append([int(item.split("/")[1]) - 1
                          for item in line.split()[1:]])
    return uv_stats(uvs, faces)


def read_colmap_ply(path):
    lines = path.read_text(encoding="ascii").splitlines()
    vertex_count = 0
    face_count = 0
    for line in lines:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[2])
        elif line.startswith("element face "):
            face_count = int(line.split()[2])
        elif line == "end_header":
            data_start = lines.index(line) + 1
            break
    uvs = []
    faces = []
    for line in lines[data_start + vertex_count:data_start + vertex_count + face_count]:
        values = line.split()
        num_vertices = int(values[0])
        offset = 1 + num_vertices
        num_texcoords = int(values[offset])
        texcoords = list(map(float, values[offset + 1:offset + 1 + num_texcoords]))
        base = len(uvs)
        uvs.extend((texcoords[index], texcoords[index + 1])
                   for index in range(0, len(texcoords), 2))
        faces.append(list(range(base, base + num_vertices)))
    return uv_stats(uvs, faces)


def texture_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read texture: {path}")
    nonblack = np.max(image, axis=2) > 0
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "pixels": int(image.shape[0] * image.shape[1]),
            "nonblack_ratio": float(nonblack.mean()),
            "mean_bgr": image.mean(axis=(0, 1)).tolist(),
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colite_dir", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    colite_mesh, colite_faces = read_obj(args.colite_dir / "mesh.obj")
    colmap_mesh, colmap_faces = read_colmap_ply(args.colmap_dir / "mesh.ply")
    intersection = colite_faces & colmap_faces
    union = colite_faces | colmap_faces
    result = {
        "colite": {"mesh": colite_mesh,
                   "texture": texture_stats(args.colite_dir / "texture.png")},
        "colmap": {"mesh": colmap_mesh,
                   "texture": texture_stats(args.colmap_dir / "texture.png")},
        "textured_face_overlap": {
            "intersection": len(intersection), "union": len(union),
            "jaccard": len(intersection) / len(union) if union else 1.0,
            "colite_only": len(colite_faces - colmap_faces),
            "colmap_only": len(colmap_faces - colite_faces),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
