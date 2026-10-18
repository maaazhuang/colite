#!/usr/bin/env python3
"""Verify sparse-model interoperability with an actual COLMAP executable.

Uses only the Python standard library. Pass executables built for the same OS.
The output directory contains models, process logs and a JSON validation report.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess


# COLMAP camera model IDs and ordered parameter counts.
MODELS = [
    ("SIMPLE_PINHOLE", 3), ("PINHOLE", 4), ("SIMPLE_RADIAL", 4),
    ("RADIAL", 5), ("OPENCV", 8), ("OPENCV_FISHEYE", 8),
    ("FULL_OPENCV", 12), ("FOV", 5), ("SIMPLE_RADIAL_FISHEYE", 4),
    ("RADIAL_FISHEYE", 5), ("THIN_PRISM_FISHEYE", 12),
    ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
]
INVALID_ID = (1 << 64) - 1


def read_binary(path):
    """Decode the documented little-endian format independently of COLITE."""
    def unpack(file, fmt):
        size = struct.calcsize("<" + fmt)
        data = file.read(size)
        if len(data) != size:
            raise AssertionError("Truncated binary model")
        return struct.unpack("<" + fmt, data)

    def finished(file):
        assert not file.read(1), "Unexpected trailing model bytes"

    cameras, images, points = {}, {}, {}
    with (path / "cameras.bin").open("rb") as file:
        for _ in range(unpack(file, "Q")[0]):
            identifier, model, width, height = unpack(file, "IiQQ")
            name, count = MODELS[model]
            cameras[identifier] = [name, width, height, *unpack(file, "d" * count)]
        finished(file)
    with (path / "images.bin").open("rb") as file:
        for _ in range(unpack(file, "Q")[0]):
            header = unpack(file, "I7dI")
            name = bytearray()
            while True:
                char = file.read(1)
                assert char, "Unterminated image name"
                if char == b"\0":
                    break
                name.extend(char)
            observations = []
            for _ in range(unpack(file, "Q")[0]):
                x, y, identifier = unpack(file, "ddQ")
                observations.append([x, y, -1 if identifier == INVALID_ID else identifier])
            images[header[0]] = [*header[1:], name.decode("utf-8"), observations]
        finished(file)
    with (path / "points3D.bin").open("rb") as file:
        for _ in range(unpack(file, "Q")[0]):
            header = unpack(file, "Q3d3Bd")
            track = [list(unpack(file, "II")) for _ in range(unpack(file, "Q")[0])]
            points[header[0]] = [*header[1:], track]
        finished(file)
    return [cameras, images, points]


def read_text(path):
    cameras, images, points = {}, {}, {}
    for line in (path / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        cameras[int(fields[0])] = [fields[1], int(fields[2]), int(fields[3]),
                                   *map(float, fields[4:])]
    lines = iter((path / "images.txt").read_text().splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        observations = next(lines).split()
        assert len(observations) % 3 == 0
        images[int(fields[0])] = [*map(float, fields[1:8]), int(fields[8]), fields[9],
            [[float(observations[i]), float(observations[i + 1]), int(observations[i + 2])]
             for i in range(0, len(observations), 3)]]
    for line in (path / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        assert (len(fields) - 8) % 2 == 0
        points[int(fields[0])] = [*map(float, fields[1:4]), *map(int, fields[4:7]),
            float(fields[7]), [[int(fields[i]), int(fields[i + 1])]
                              for i in range(8, len(fields), 2)]]
    return [cameras, images, points]


def equal(expected, actual, location="model"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), location + " IDs differ"
        for key in expected:
            equal(expected[key], actual[key], location + "/" + str(key))
    elif isinstance(expected, list):
        assert len(expected) == len(actual), location + " lengths differ"
        for index, (left, right) in enumerate(zip(expected, actual)):
            equal(left, right, location + "/" + str(index))
    elif isinstance(expected, float):
        assert math.isclose(expected, actual, rel_tol=1e-13, abs_tol=1e-13), location
    else:
        assert expected == actual, location + " values differ"


def fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    cameras = []
    for identifier, (name, count) in enumerate(MODELS, 1):
        base = [500, 320, 240] if identifier in (1, 3, 4, 9, 10) else [500, 510, 320, 240]
        params = base + [0.01] * (count - len(base))
        cameras.append(f"{identifier} {name} 640 480 " + " ".join(map(str, params)))
    (path / "cameras.txt").write_text("\n".join(cameras) + "\n", encoding="utf-8")
    # IDs exceed uint32, but stay within COLMAP 4.0's signed TXT parser range.
    (path / "images.txt").write_text(
        "7 0.5 0.5 0.5 0.5 1.25 -2.5 3.75 2 folder/image7.jpg\n"
        "10.5 20.25 4294967313 30.5 40.25 -1\n"
        "9 1 0 0 0 -1 0 0 2 image9.jpg\n15 25 4294967313\n"
        "11 1 0 0 0 0 -1 0 2 empty.jpg\n\n", encoding="utf-8")
    (path / "points3D.txt").write_text(
        "4294967313 1.25 -2.5 7.75 17 101 249 0.125 7 0 9 0\n"
        "4294967314 0 0 5 0 255 127 -1\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colite", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--input_path", type=Path, help="Also verify an existing TXT/BIN model")
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    logs, checks, cases = [], [], {}
    (args.output_path / "report.json").unlink(missing_ok=True)

    def run(command):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
        logs.append({"command": list(map(str, command)), "returncode": result.returncode,
                     "stdout": result.stdout, "stderr": result.stderr})
        (args.output_path / "processes.json").write_text(json.dumps(logs, indent=2))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)

    def verify(source, label):
        expected = read_binary(source) if (source / "cameras.bin").exists() else read_text(source)
        cases[label] = {"source": str(source), "cameras": len(expected[0]),
                        "registered_images": len(expected[1]), "points3d": len(expected[2])}
        exported = args.output_path / (label + "_colite")
        run([args.colite, "model_converter", "--input_path", source,
             "--output_path", exported, "--output_type", "both"])
        equal(expected, read_text(exported))
        equal(expected, read_binary(exported))
        checks.append(label + ": COLITE TXT/BIN preserve all sparse-model fields")
        for fmt in ("TXT", "BIN"):
            # Isolate each format so COLMAP cannot prefer BIN during the TXT check.
            isolated = args.output_path / (label + "_input_" + fmt.lower())
            run([args.colite, "model_converter", "--input_path", exported,
                 "--Colite.input_format", fmt.lower(), "--output_path", isolated,
                 "--output_type", fmt.lower()])
            official = args.output_path / (label + "_colmap_" + fmt.lower())
            official.mkdir(parents=True, exist_ok=True)
            run([args.colmap, "model_converter", "--input_path", isolated,
                 "--output_path", official, "--output_type", fmt])
            equal(expected, read_text(official) if fmt == "TXT" else read_binary(official))
            returned = args.output_path / (label + "_returned_" + fmt.lower())
            run([args.colite, "model_converter", "--input_path", official,
                 "--output_path", returned, "--output_type", "both"])
            equal(expected, read_text(returned))
            equal(expected, read_binary(returned))
            checks.append(label + ": COLITE -> COLMAP -> COLITE via " + fmt)
        run([args.colite, "compare_colmap", "--reference_path", exported,
             "--candidate_path", exported, "--reference_format", "txt",
             "--candidate_format", "bin", "--output_path", args.output_path / (label + "_comparison")])
        checks.append(label + ": compare_colmap reads TXT and BIN explicitly")

    source = args.output_path / "fixture"
    fixture(source)
    verify(source, "fixture")
    if args.input_path:
        verify(args.input_path, "existing")
    report = {"success": True, "checks": checks, "cases": cases,
              "camera_models": [name for name, _ in MODELS],
              "scope": "cameras/images/points3D; rig/frame metadata is not preserved"}
    (args.output_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
