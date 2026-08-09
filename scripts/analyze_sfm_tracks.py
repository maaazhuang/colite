#!/usr/bin/env python3
"""统计 COLMAP 文本模型 points3D.txt 的轨迹长度分布。"""

import argparse
import json
from pathlib import Path


def analyze_points3d(path: Path) -> dict:
    track_lengths = []
    unique_view_lengths = []
    duplicate_observations = 0

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 8 or (len(fields) - 8) % 2 != 0:
                raise ValueError(f"无效的 points3D.txt 行: {line[:120]}")

            image_ids = [int(fields[index]) for index in range(8, len(fields), 2)]
            track_length = len(image_ids)
            unique_view_length = len(set(image_ids))
            track_lengths.append(track_length)
            unique_view_lengths.append(unique_view_length)
            duplicate_observations += track_length - unique_view_length

    def distribution(lengths: list[int]) -> dict:
        point_count = len(lengths)
        observation_count = sum(lengths)

        def bucket(predicate) -> dict:
            selected = [length for length in lengths if predicate(length)]
            selected_observations = sum(selected)
            return {
                "points": len(selected),
                "point_ratio": len(selected) / point_count if point_count else 0.0,
                "observations": selected_observations,
                "observation_ratio": (
                    selected_observations / observation_count
                    if observation_count
                    else 0.0
                ),
            }

        return {
            "points": point_count,
            "observations": observation_count,
            "mean_track_length": observation_count / point_count if point_count else 0.0,
            "track_1": bucket(lambda length: length == 1),
            "track_2": bucket(lambda length: length == 2),
            "track_3": bucket(lambda length: length == 3),
            "track_ge_4": bucket(lambda length: length >= 4),
        }

    return {
        "path": str(path),
        "raw_track": distribution(track_lengths),
        "unique_view_track": distribution(unique_view_lengths),
        "duplicate_same_image_observations": duplicate_observations,
    }


def print_summary(label: str, result: dict) -> None:
    print(f"[{label}] {result['path']}")
    for name in ("raw_track", "unique_view_track"):
        stats = result[name]
        print(
            f"  {name}: points={stats['points']} observations={stats['observations']} "
            f"mean={stats['mean_track_length']:.6f}"
        )
        for bucket_name in ("track_2", "track_3", "track_ge_4"):
            bucket = stats[bucket_name]
            print(
                f"    {bucket_name}: points={bucket['points']} "
                f"({bucket['point_ratio']:.2%}), observations={bucket['observations']} "
                f"({bucket['observation_ratio']:.2%})"
            )
    print(
        "  duplicate_same_image_observations="
        f"{result['duplicate_same_image_observations']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=POINTS3D_TXT",
        help="可重复指定，例如 colite=output/.../points3D.txt",
    )
    parser.add_argument("--json", type=Path, help="可选 JSON 输出路径")
    args = parser.parse_args()

    results = {}
    for model in args.model:
        if "=" not in model:
            parser.error(f"--model 缺少 LABEL=PATH 格式: {model}")
        label, raw_path = model.split("=", 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / "points3D.txt"
        results[label] = analyze_points3d(path)
        print_summary(label, results[label])

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
