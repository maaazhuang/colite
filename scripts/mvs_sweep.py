#!/usr/bin/env python3
"""Run and summarize small COLITE MVS parameter sweeps.

The script intentionally stays dependency-free so it can be used from WSL,
CTest, or a plain project checkout. It supports two workflows:

  summarize: collect metrics from existing recon_mvs output directories.
  run: execute recon_mvs for a grid of geometric refinement parameters, then
       write a summary TSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PAIR_SUM_COLUMNS = [
    "num_accepted",
    "num_rejected_depth",
    "num_rejected_mutual_consistency",
    "num_rejected_invalid",
    "num_rejected_normal",
    "num_rejected_visited",
    "num_rejected_duplicate",
]

PATCH_MATCH_SUM_COLUMNS = [
    "perturbation_geometric_candidate_rejections",
    "perturbation_geometric_candidate_acceptances",
]

PATCH_MATCH_AVG_COLUMNS = [
    "perturbation_mean_abs_depth_update",
    "perturbation_mean_normal_angle_degrees",
]

PATCH_MATCH_MAX_COLUMNS = [
    "perturbation_max_abs_depth_update",
    "perturbation_max_normal_angle_degrees",
]

FUSION_ERROR_METRICS = [
    "relative_depth_error",
    "reprojection_error_pixels",
    "normal_angle_degrees",
    "view_angle_degrees",
]

FUSION_ERROR_PERCENTILES = ["p50", "p95", "p99"]

SUMMARY_COLUMNS = [
    "case",
    "output_path",
    "patch_match_requested_backend",
    "patch_match_actual_backend",
    "patch_match_geometric_consistency_weight",
    "patch_match_geometric_max_relative_depth_error",
    "normal_refinement_max_relative_depth_difference",
    "photometric_filtered_pixels",
    "normal_refined_pixels",
    "geometric_consistent_pixels",
    "geometric_filtered_pixels",
    "fused_points",
    "fused_observations",
    "num_tracks",
    "num_single_observation_tracks",
    "num_multi_observation_tracks",
    "num_observations",
    "mean_observations",
    "multi_observation_track_ratio",
    *[f"pair_{name}_sum" for name in PAIR_SUM_COLUMNS],
    *[f"{name}_sum" for name in PATCH_MATCH_SUM_COLUMNS],
    *[f"{name}_avg" for name in PATCH_MATCH_AVG_COLUMNS],
    *[f"{name}_max" for name in PATCH_MATCH_MAX_COLUMNS],
    *[
        f"fusion_{metric}_{percentile}"
        for metric in FUSION_ERROR_METRICS
        for percentile in FUSION_ERROR_PERCENTILES
    ],
]


@dataclass(frozen=True)
class SweepCase:
    name: str
    geometric_consistency_weight: float
    geometric_max_relative_depth_error: float
    normal_refinement_max_relative_depth_difference: float


def format_number(value: float) -> str:
    return f"{value:.12g}"


def format_token(value: float) -> str:
    text = format_number(value)
    return text.replace("-", "m").replace(".", "p")


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"Expected at least one numeric value in '{text}'.")
    return values


def read_key_value_file(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def numeric_sum(rows: Iterable[Dict[str, str]], column: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(column, "")
        if value != "":
            total += float(value)
    return total


def numeric_average(rows: Sequence[Dict[str, str]], column: str) -> str:
    values = [float(row[column]) for row in rows if row.get(column, "") != ""]
    if not values:
        return ""
    return format_number(sum(values) / len(values))


def numeric_max(rows: Sequence[Dict[str, str]], column: str) -> str:
    values = [float(row[column]) for row in rows if row.get(column, "") != ""]
    if not values:
        return ""
    return format_number(max(values))


def read_case_metadata(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "sweep_case.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(key): format_number(float(value)) for key, value in data.items()}


def read_fusion_error_percentiles(run_dir: Path) -> Dict[str, str]:
    rows = read_tsv_rows(run_dir / "dense" / "fusion_error_distribution.tsv")
    result: Dict[str, str] = {}
    for row in rows:
        if row.get("record") != "percentile":
            continue
        metric = row.get("metric", "")
        label = row.get("label", "")
        value = row.get("value", "")
        if (
            metric in FUSION_ERROR_METRICS
            and label in FUSION_ERROR_PERCENTILES
            and value != ""
        ):
            result[f"fusion_{metric}_{label}"] = value
    return result


def summarize_run(run_dir: Path) -> Dict[str, str]:
    run_dir = Path(run_dir)
    summary = read_key_value_file(run_dir / "summary.txt")
    fusion = read_key_value_file(run_dir / "dense" / "fusion_summary.txt")
    pair_rows = read_tsv_rows(run_dir / "dense" / "fusion_image_pairs.tsv")
    patch_rows = read_tsv_rows(run_dir / "stereo" / "patch_match.tsv")

    row = {column: "" for column in SUMMARY_COLUMNS}
    row["case"] = run_dir.name
    row["output_path"] = str(run_dir)

    for key, value in read_case_metadata(run_dir).items():
        if key in row:
            row[key] = value

    for key in [
        "photometric_filtered_pixels",
        "normal_refined_pixels",
        "geometric_consistent_pixels",
        "geometric_filtered_pixels",
        "fused_points",
        "fused_observations",
    ]:
        row[key] = summary.get(key, "")

    for key in [
        "num_tracks",
        "num_single_observation_tracks",
        "num_multi_observation_tracks",
        "num_observations",
        "mean_observations",
        "multi_observation_track_ratio",
    ]:
        row[key] = fusion.get(key, "")

    if patch_rows:
        row["patch_match_requested_backend"] = patch_rows[0].get(
            "requested_backend", ""
        )
        actual_backends = {
            row_data.get("actual_backend", "")
            for row_data in patch_rows
            if row_data.get("actual_backend", "") != ""
        }
        if len(actual_backends) == 1:
            row["patch_match_actual_backend"] = next(iter(actual_backends))
        elif len(actual_backends) > 1:
            row["patch_match_actual_backend"] = "mixed"

    for column in PAIR_SUM_COLUMNS:
        row[f"pair_{column}_sum"] = format_number(numeric_sum(pair_rows, column))

    for column in PATCH_MATCH_SUM_COLUMNS:
        row[f"{column}_sum"] = format_number(numeric_sum(patch_rows, column))

    for column in PATCH_MATCH_AVG_COLUMNS:
        row[f"{column}_avg"] = numeric_average(patch_rows, column)

    for column in PATCH_MATCH_MAX_COLUMNS:
        row[f"{column}_max"] = numeric_max(patch_rows, column)

    for key, value in read_fusion_error_percentiles(run_dir).items():
        row[key] = value

    return row


def write_summary(run_dirs: Sequence[Path], output_path: Path) -> None:
    rows = [summarize_run(Path(run_dir)) for run_dir in run_dirs]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_sweep_cases(
    weights: Sequence[float],
    relative_depth_errors: Sequence[float],
    normal_depth_thresholds: Sequence[float],
    prefix: str,
) -> List[SweepCase]:
    cases: List[SweepCase] = []
    for weight, depth_error, normal_threshold in itertools.product(
        weights, relative_depth_errors, normal_depth_thresholds
    ):
        name = (
            f"{prefix}-w{format_token(weight)}"
            f"-gd{format_token(depth_error)}"
            f"-nd{format_token(normal_threshold)}"
        )
        cases.append(
            SweepCase(
                name=name,
                geometric_consistency_weight=weight,
                geometric_max_relative_depth_error=depth_error,
                normal_refinement_max_relative_depth_difference=normal_threshold,
            )
        )
    return cases


def write_case_metadata(case: SweepCase, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    data = {
        "patch_match_geometric_consistency_weight":
            case.geometric_consistency_weight,
        "patch_match_geometric_max_relative_depth_error":
            case.geometric_max_relative_depth_error,
        "normal_refinement_max_relative_depth_difference":
            case.normal_refinement_max_relative_depth_difference,
    }
    with (output_path / "sweep_case.json").open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def build_recon_mvs_command(args: argparse.Namespace,
                            case: SweepCase,
                            output_path: Path) -> List[str]:
    return [
        str(args.recon_mvs),
        "--model_path",
        str(args.model_path),
        "--image_path",
        str(args.image_path),
        "--output_path",
        str(output_path),
        "--run_patch_match",
        "--max_image_size",
        str(args.max_image_size),
        "--patch_match_iterations",
        str(args.patch_match_iterations),
        "--patch_match_pyramid_levels",
        str(args.patch_match_pyramid_levels),
        "--patch_match_window_radius",
        str(args.patch_match_window_radius),
        "--patch_match_perturbation_samples",
        str(args.patch_match_perturbation_samples),
        "--patch_match_geometric_refinement_iterations",
        str(args.patch_match_geometric_refinement_iterations),
        "--patch_match_geometric_consistency_weight",
        format_number(case.geometric_consistency_weight),
        "--patch_match_geometric_max_relative_depth_error",
        format_number(case.geometric_max_relative_depth_error),
        "--filter_photometric_max_cost",
        format_number(args.filter_photometric_max_cost),
        "--refine_normals_from_depth",
        "--normal_refinement_max_relative_depth_difference",
        format_number(case.normal_refinement_max_relative_depth_difference),
        "--fuse_stereo",
        "--fusion_min_num_observations",
        str(args.fusion_min_num_observations),
        "--fusion_max_normal_error",
        format_number(args.fusion_max_normal_error),
        "--fusion_max_view_angle",
        format_number(args.fusion_max_view_angle),
        "--fusion_source_pixel_search_radius",
        str(args.fusion_source_pixel_search_radius),
        "--filter_geometric_min_num_consistent_sources",
        str(args.filter_geometric_min_num_consistent_sources),
        "--filter_dense_cloud",
        "--dense_cloud_min_track_observations",
        str(args.dense_cloud_min_track_observations),
        "--mesh_dense_cloud",
        "--mesh_max_edge_length",
        format_number(args.mesh_max_edge_length),
        "--mesh_max_normal_angle",
        format_number(args.mesh_max_normal_angle),
        "--num_threads",
        str(args.num_threads),
    ]


def write_commands(commands: Sequence[Sequence[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["case", "command"])
        for command in commands:
            case_name = Path(command[command.index("--output_path") + 1]).name
            writer.writerow([case_name, " ".join(command)])


def run_sweep(args: argparse.Namespace) -> None:
    cases = build_sweep_cases(
        weights=parse_float_list(args.weights),
        relative_depth_errors=parse_float_list(args.relative_depth_errors),
        normal_depth_thresholds=parse_float_list(args.normal_depth_thresholds),
        prefix=args.case_prefix,
    )

    output_root = Path(args.output_root)
    run_dirs: List[Path] = []
    commands: List[List[str]] = []
    for case in cases:
        output_path = output_root / case.name
        write_case_metadata(case, output_path)
        command = build_recon_mvs_command(args, case, output_path)
        commands.append(command)
        run_dirs.append(output_path)
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.commands_output:
        write_commands(commands, Path(args.commands_output))
    if args.summary_output:
        write_summary(run_dirs, Path(args.summary_output))


def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recon-mvs", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--commands-output", type=Path)
    parser.add_argument("--case-prefix", default="colite-mvs-sweep")
    parser.add_argument("--weights", default="0.05")
    parser.add_argument("--relative-depth-errors", default="0.01")
    parser.add_argument("--normal-depth-thresholds", default="0.005")
    parser.add_argument("--max-image-size", type=int, default=32)
    parser.add_argument("--patch-match-iterations", type=int, default=1)
    parser.add_argument("--patch-match-pyramid-levels", type=int, default=1)
    parser.add_argument("--patch-match-window-radius", type=int, default=1)
    parser.add_argument("--patch-match-perturbation-samples", type=int,
                        default=3)
    parser.add_argument("--patch-match-geometric-refinement-iterations",
                        type=int, default=1)
    parser.add_argument("--filter-photometric-max-cost", type=float,
                        default=0.3)
    parser.add_argument("--fusion-min-num-observations", type=int, default=1)
    parser.add_argument("--fusion-max-normal-error", type=float, default=10.0)
    parser.add_argument("--fusion-max-view-angle", type=float, default=180.0)
    parser.add_argument("--fusion-source-pixel-search-radius", type=int,
                        default=1)
    parser.add_argument("--filter-geometric-min-num-consistent-sources",
                        type=int, default=1)
    parser.add_argument("--dense-cloud-min-track-observations", type=int,
                        default=1)
    parser.add_argument("--mesh-max-edge-length", type=float, default=0.5)
    parser.add_argument("--mesh-max-normal-angle", type=float, default=30.0)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser(
        "summarize", help="summarize existing recon_mvs output directories"
    )
    summarize.add_argument("--runs", nargs="+", required=True, type=Path)
    summarize.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser(
        "run", help="run a geometric-refinement parameter sweep"
    )
    add_common_run_arguments(run)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summarize":
        write_summary(args.runs, args.output)
        return 0
    if args.command == "run":
        run_sweep(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
