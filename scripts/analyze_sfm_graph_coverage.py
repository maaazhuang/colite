#!/usr/bin/env python3
"""Measure how a COLMAP text model covers Colite verified correspondences."""

import argparse
import json
from pathlib import Path


def read_observation_points(images_path: Path) -> dict[tuple[int, int], int]:
    observations: dict[tuple[int, int], int] = {}
    with images_path.open("r", encoding="utf-8") as stream:
        while True:
            header = stream.readline()
            if not header:
                break
            if not header.strip() or header.startswith("#"):
                continue
            image_id = int(header.split(maxsplit=1)[0])
            points = stream.readline().split()
            if len(points) % 3:
                raise ValueError("Invalid images.txt POINTS2D row")
            for offset in range(0, len(points), 3):
                observations[(image_id, offset // 3)] = int(points[offset + 2])
    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    observations = read_observation_points(args.model / "images.txt")
    counts = {"neither": 0, "one": 0, "same": 0, "different": 0}
    different_pairs: set[tuple[int, int]] = set()
    with args.verified.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            point1 = observations.get((int(fields[0]), int(fields[11])), -1)
            point2 = observations.get((int(fields[1]), int(fields[12])), -1)
            if point1 < 0 and point2 < 0:
                counts["neither"] += 1
            elif point1 < 0 or point2 < 0:
                counts["one"] += 1
            elif point1 == point2:
                counts["same"] += 1
            else:
                counts["different"] += 1
                different_pairs.add(tuple(sorted((point1, point2))))
    result = {
        **counts,
        "verified_edges": sum(counts.values()),
        "unique_different_point_pairs": len(different_pairs),
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
