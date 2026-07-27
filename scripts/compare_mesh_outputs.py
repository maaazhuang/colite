#!/usr/bin/env python3
"""Compare two triangle PLY meshes with deterministic geometry/topology metrics."""

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PLY_TYPES = {
    "char": ("b", np.int8), "uchar": ("B", np.uint8),
    "int8": ("b", np.int8), "uint8": ("B", np.uint8),
    "short": ("h", np.int16), "ushort": ("H", np.uint16),
    "int16": ("h", np.int16), "uint16": ("H", np.uint16),
    "int": ("i", np.int32), "uint": ("I", np.uint32),
    "int32": ("i", np.int32), "uint32": ("I", np.uint32),
    "float": ("f", np.float32), "float32": ("f", np.float32),
    "double": ("d", np.float64), "float64": ("d", np.float64),
}


def read_ply(path: Path):
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        elements = []
        current = None
        while True:
            line = stream.readline().decode("ascii").strip()
            fields = line.split()
            if not fields or fields[0] in ("comment", "obj_info"):
                continue
            if fields[0] == "format":
                fmt = fields[1]
            elif fields[0] == "element":
                current = {"name": fields[1], "count": int(fields[2]),
                           "properties": []}
                elements.append(current)
            elif fields[0] == "property":
                current["properties"].append(fields[1:])
            elif fields[0] == "end_header":
                break

        vertex_element = next(item for item in elements if item["name"] == "vertex")
        face_element = next((item for item in elements if item["name"] == "face"),
                            {"count": 0, "properties": []})
        vertices = []
        normals = []
        faces = []
        if fmt == "ascii":
            names = [prop[-1] for prop in vertex_element["properties"]]
            for _ in range(vertex_element["count"]):
                values = stream.readline().decode("ascii").split()
                row = dict(zip(names, map(float, values)))
                vertices.append((row["x"], row["y"], row["z"]))
                if all(name in row for name in ("nx", "ny", "nz")):
                    normals.append((row["nx"], row["ny"], row["nz"]))
            for _ in range(face_element["count"]):
                values = list(map(int, stream.readline().decode("ascii").split()))
                faces.append(tuple(values[1:1 + values[0]]))
        elif fmt == "binary_little_endian":
            scalar_props = vertex_element["properties"]
            if any(prop[0] == "list" for prop in scalar_props):
                raise ValueError("List-valued vertex properties are unsupported")
            vertex_fmt = "<" + "".join(PLY_TYPES[prop[0]][0] for prop in scalar_props)
            vertex_size = struct.calcsize(vertex_fmt)
            names = [prop[1] for prop in scalar_props]
            for _ in range(vertex_element["count"]):
                values = struct.unpack(vertex_fmt, stream.read(vertex_size))
                row = dict(zip(names, values))
                vertices.append((row["x"], row["y"], row["z"]))
                if all(name in row for name in ("nx", "ny", "nz")):
                    normals.append((row["nx"], row["ny"], row["nz"]))
            if face_element["count"]:
                list_prop = next(prop for prop in face_element["properties"]
                                 if prop[0] == "list")
                count_fmt = "<" + PLY_TYPES[list_prop[1]][0]
                item_fmt_char = PLY_TYPES[list_prop[2]][0]
                for _ in range(face_element["count"]):
                    count = struct.unpack(count_fmt,
                                          stream.read(struct.calcsize(count_fmt)))[0]
                    item_fmt = "<" + item_fmt_char * count
                    faces.append(struct.unpack(item_fmt,
                                               stream.read(struct.calcsize(item_fmt))))
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return (np.asarray(vertices, dtype=np.float64),
            np.asarray(normals, dtype=np.float64), faces, fmt)


def topology(vertices, faces):
    triangles = [face for face in faces if len(face) == 3]
    edges = Counter()
    degenerate = 0
    areas = []
    for face in triangles:
        a, b, c = face
        edges[tuple(sorted((a, b)))] += 1
        edges[tuple(sorted((b, c)))] += 1
        edges[tuple(sorted((c, a)))] += 1
        area = 0.5 * np.linalg.norm(np.cross(vertices[b] - vertices[a],
                                             vertices[c] - vertices[a]))
        areas.append(float(area))
        degenerate += area <= 1e-15
    return {
        "vertices": int(len(vertices)), "faces": int(len(faces)),
        "triangles": int(len(triangles)), "edges": int(len(edges)),
        "boundary_edges": int(sum(value == 1 for value in edges.values())),
        "nonmanifold_edges": int(sum(value > 2 for value in edges.values())),
        "degenerate_triangles": int(degenerate),
        "euler_characteristic": int(len(vertices) - len(edges) + len(triangles)),
        "area_sum": float(sum(areas)),
    }


def geometric_face_keys(vertices, faces, decimals=6):
    """Canonicalize triangle faces by coordinates rather than PLY indices."""
    keys = set()
    for face in faces:
        if len(face) != 3:
            continue
        corners = [tuple(np.round(vertices[index], decimals)) for index in face]
        keys.add(tuple(sorted(corners)))
    return keys


def subsample(values, max_count):
    if len(values) <= max_count:
        return values
    indices = np.linspace(0, len(values) - 1, max_count, dtype=np.int64)
    return values[indices]


def nearest_distances(source, target):
    if not len(source) or not len(target):
        return np.empty(0, dtype=np.float64)
    index = cv2.flann_Index(target.astype(np.float32),
                            dict(algorithm=1, trees=8))
    _, squared = index.knnSearch(source.astype(np.float32), 1,
                                 params=dict(checks=128))
    return np.sqrt(np.maximum(squared[:, 0], 0.0)).astype(np.float64)


def quantiles(values):
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)), "mean": float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=20000)
    args = parser.parse_args()
    candidate, candidate_normals, candidate_faces, candidate_fmt = read_ply(args.candidate)
    reference, reference_normals, reference_faces, reference_fmt = read_ply(args.reference)
    candidate_sample = subsample(candidate, args.max_samples)
    reference_sample = subsample(reference, args.max_samples)
    candidate_min = candidate.min(axis=0)
    candidate_max = candidate.max(axis=0)
    reference_min = reference.min(axis=0)
    reference_max = reference.max(axis=0)
    diagonal = float(np.linalg.norm(reference_max - reference_min))
    candidate_face_keys = geometric_face_keys(candidate, candidate_faces)
    reference_face_keys = geometric_face_keys(reference, reference_faces)
    common_face_keys = candidate_face_keys & reference_face_keys
    union_face_keys = candidate_face_keys | reference_face_keys
    result = {
        "candidate_format": candidate_fmt, "reference_format": reference_fmt,
        "candidate": topology(candidate, candidate_faces),
        "reference": topology(reference, reference_faces),
        "geometric_face_overlap": {
            "common": len(common_face_keys),
            "candidate_only": len(candidate_face_keys - reference_face_keys),
            "reference_only": len(reference_face_keys - candidate_face_keys),
            "jaccard": (len(common_face_keys) / len(union_face_keys)
                        if union_face_keys else 1.0),
        },
        "candidate_bounds": {
            "min": candidate_min.tolist(),
            "max": candidate_max.tolist(),
            "diagonal": float(np.linalg.norm(candidate_max - candidate_min)),
        },
        "reference_bounds": {
            "min": reference_min.tolist(),
            "max": reference_max.tolist(),
            "diagonal": diagonal,
        },
        "reference_bbox_diagonal": diagonal,
        "candidate_to_reference": quantiles(nearest_distances(candidate_sample,
                                                                reference_sample)),
        "reference_to_candidate": quantiles(nearest_distances(reference_sample,
                                                                candidate_sample)),
        "candidate_has_normals": len(candidate_normals) == len(candidate),
        "reference_has_normals": len(reference_normals) == len(reference),
    }
    for key in ("candidate_to_reference", "reference_to_candidate"):
        if result[key].get("count") and diagonal > 0:
            result[key]["median_over_bbox"] = result[key]["median"] / diagonal
            result[key]["p95_over_bbox"] = result[key]["p95"] / diagonal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
