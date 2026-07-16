#!/usr/bin/env python3
"""Create a deterministic dense-workspace fixture for meshing/texturing A/B."""

import argparse
import math
import pathlib
import struct


def normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return tuple(x / length for x in v)


def quaternion_from_rotation(r):
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (0.25 * s, (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    axis = max(range(3), key=lambda i: r[i][i])
    j, k = (axis + 1) % 3, (axis + 2) % 3
    s = math.sqrt(1.0 + r[axis][axis] - r[j][j] - r[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[axis + 1] = 0.25 * s
    q[0] = (r[k][j] - r[j][k]) / s
    q[j + 1] = (r[j][axis] + r[axis][j]) / s
    q[k + 1] = (r[k][axis] + r[axis][k]) / s
    return tuple(q)


def camera_pose(center):
    z = normalize(tuple(-x for x in center))
    down = (0.0, 1.0, 0.0)
    if abs(sum(a * b for a, b in zip(z, down))) > 0.95:
        down = (0.0, 0.0, 1.0)
    x = normalize((down[1] * z[2] - down[2] * z[1],
                   down[2] * z[0] - down[0] * z[2],
                   down[0] * z[1] - down[1] * z[0]))
    y = (z[1] * x[2] - z[2] * x[1],
         z[2] * x[0] - z[0] * x[2],
         z[0] * x[1] - z[1] * x[0])
    rotation = (x, y, z)
    translation = tuple(-sum(rotation[row][col] * center[col]
                             for col in range(3)) for row in range(3))
    return quaternion_from_rotation(rotation), translation


def make_octa_sphere(levels=3):
    vertices = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(levels):
        midpoint = {}
        new_faces = []
        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in midpoint:
                midpoint[key] = len(vertices)
                vertices.append(normalize(tuple(vertices[a][i] + vertices[b][i]
                                                for i in range(3))))
            return midpoint[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces.extend(((a, ab, ca), (ab, b, bc),
                              (ca, bc, c), (ab, bc, ca)))
        faces = new_faces
    return [normalize(v) for v in vertices], faces


def write_ply(path, vertices, faces):
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property float nx\nproperty float ny\nproperty float nz\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertices:
            nx, ny, nz = normalize((x, y, z))
            stream.write(f"{x} {y} {z} {nx} {ny} {nz} "
                         f"{int((x+1)*127.5)} {int((y+1)*127.5)} {int((z+1)*127.5)}\n")
        for face in faces:
            stream.write("3 " + " ".join(map(str, face)) + "\n")


def write_ppm(path, view_id, size=256):
    with path.open("wb") as stream:
        stream.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
        for y in range(size):
            for x in range(size):
                stream.write(bytes(((x + view_id * 29) % 256,
                                    (y + view_id * 47) % 256,
                                    (160 + view_id * 17) % 256)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radial_jitter", type=float, default=0.0,
        help="Deterministically perturb radii to avoid co-spherical Delaunay degeneracy")
    args = parser.parse_args()
    root = pathlib.Path(args.output).resolve()
    images = root / "images"
    sparse = root / "sparse"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    vertices, faces = make_octa_sphere()
    if args.radial_jitter:
        vertices = [
            tuple(coordinate * (1.0 + args.radial_jitter *
                  math.sin(12.9898 * index + 78.233)) for coordinate in point)
            for index, point in enumerate(vertices)
        ]
    write_ply(root / "mesh.ply", vertices, faces)
    write_ply(root / "fused.ply", vertices, [])

    centers = [(0, 0, -4), (4, 0, 0), (0, 0, 4), (-4, 0, 0), (0, -4, 0)]
    with (sparse / "cameras.txt").open("w", encoding="ascii") as cameras:
        for index in range(len(centers)):
            cameras.write(f"{index + 1} PINHOLE 256 256 220 220 128 128\n")
    with (sparse / "images.txt").open("w", encoding="ascii") as image_file:
        for index, center in enumerate(centers):
            quaternion, translation = camera_pose(center)
            name = f"view-{index}.ppm"
            image_file.write(f"{index + 1} {' '.join(map(str, quaternion))} "
                             f"{' '.join(map(str, translation))} {index + 1} {name}\n\n")
            write_ppm(images / name, index)
    (sparse / "points3D.txt").write_text("", encoding="ascii")

    with (root / "fused.ply.vis").open("wb") as visibility:
        visibility.write(struct.pack("<Q", len(vertices)))
        for point in vertices:
            visible = []
            for index, center in enumerate(centers):
                direction = tuple(center[i] - point[i] for i in range(3))
                if sum(point[i] * direction[i] for i in range(3)) > 0:
                    visible.append(index)
            visibility.write(struct.pack("<I", len(visible)))
            for index in visible:
                visibility.write(struct.pack("<I", index))
    print(root)


if __name__ == "__main__":
    main()
