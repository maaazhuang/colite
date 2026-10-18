#!/usr/bin/env python3
"""Reproducible staged SfM A/B; fresh workspaces, shared-input controls, real logs.

Requires Linux, numpy and the two locally built executables. Output directories
are never deleted. Existing successful commands are reused only with --resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import time

import numpy as np

from analyze_sfm_geometric_merges import read_model


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_stats(path):
    cameras, images, points = read_model(path)
    lengths = np.asarray([len(track) for _, track in points.values()])
    grouped = {image_id: [] for image_id in images}
    duplicates = 0
    for xyz, track in points.values():
        duplicates += len(track) - len({image_id for image_id, _ in track})
        for image_id, idx in track:
            grouped[image_id].append((xyz, idx))
    errors = []
    behind = 0
    for image_id, observations in grouped.items():
        if not observations:
            continue
        rotation, translation, camera_id, xy = images[image_id]
        xyz = np.asarray([point for point, _ in observations])
        indices = [idx for _, idx in observations]
        local = xyz @ rotation.T + translation
        behind += int(np.count_nonzero(local[:, 2] <= 0))
        norm = local[:, :2] / local[:, 2:3]
        model, params = cameras[camera_id]
        if model == "SIMPLE_RADIAL":
            f, cx, cy, k = params
            projected = f * norm * (1 + k * np.sum(norm * norm, axis=1)[:, None]) + [cx, cy]
        elif model == "PINHOLE":
            fx, fy, cx, cy = params
            projected = norm * [fx, fy] + [cx, cy]
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params
            projected = f * norm + [cx, cy]
        else:
            raise ValueError(f"Unsupported metric projection: {model}")
        errors.extend(np.linalg.norm(projected - xy[indices], axis=1).tolist())
    errors = np.asarray(errors)
    return dict(images=len(images), cameras=len(cameras), points=len(points),
                observations=int(lengths.sum()), mean_track_length=float(lengths.mean()) if len(lengths) else 0,
                tracks_ge4=int(np.count_nonzero(lengths >= 4)), duplicate_views=duplicates,
                behind_camera=behind, observation_error_mean=float(errors.mean()) if len(errors) else None,
                observation_error_p50=float(np.percentile(errors, 50)) if len(errors) else None,
                observation_error_p95=float(np.percentile(errors, 95)) if len(errors) else None,
                observation_error_p99=float(np.percentile(errors, 99)) if len(errors) else None,
                observation_error_max=float(errors.max()) if len(errors) else None,
                observations_above4px=int(np.count_nonzero(errors > 4)),
                observation_rmse=float(np.sqrt(np.mean(errors * errors))) if len(errors) else None)


def database_stats(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        counts = [row[0] for row in db.execute("SELECT rows FROM keypoints")]
        return dict(images=len(counts), features=sum(counts),
                    raw_matches=db.execute("SELECT COALESCE(SUM(rows),0) FROM matches").fetchone()[0],
                    verified_inliers=db.execute("SELECT COALESCE(SUM(rows),0) FROM two_view_geometries").fetchone()[0],
                    verified_pairs=db.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows>=15").fetchone()[0])


def workspace_stats(path):
    counts = []
    for feature_file in (path / "features").glob("*.bin"):
        with feature_file.open("rb") as stream:
            magic, version = struct.unpack("<II", stream.read(8))
            if magic != 0x434C4654 or version not in (1, 2):
                raise ValueError(f"Unsupported feature file: {feature_file}")
            if version == 2:
                stream.read(16)
            counts.append(struct.unpack("<Q", stream.read(8))[0])
    result = {"images": sum(1 for line in (path / "images.tsv").read_text().splitlines()
                            if line.strip() and not line.startswith("#")), "features": sum(counts)}
    if (path / "matches/summary.txt").exists():
        stats = dict(line.split(maxsplit=1) for line in (path / "matches/summary.txt").read_text().splitlines())
        result["verified_inliers"] = int(stats["verified_inliers"])
        result["verified_pairs"] = int(stats["verified_pairs"])
    for name in ("raw_matches", "verified_matches"):
        pairs, rows = set(), 0
        with (path / "matches" / f"{name}.tsv").open() as stream:
            for line in stream:
                if line.strip() and not line.startswith("#"):
                    fields = line.split()
                    pairs.add(tuple(fields[:2]))
                    rows += 1
        result[name] = rows
        result[name + "_pairs"] = len(pairs)
    return result


def perturb_model(source, target):
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "cameras.txt", target / "cameras.txt")
    _, images, points = read_model(source)
    centers = np.asarray([-rotation.T @ translation for rotation, translation, _, _ in images.values()])
    baseline = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    rng = np.random.default_rng(0)
    with (source / "images.txt").open() as src, (target / "images.txt").open("w") as dst:
        for line in src:
            if not line.strip() or line.startswith("#"):
                dst.write(line)
                continue
            fields = line.split()
            fields[5:8] = [format(float(v) + float(d), ".17g") for v, d in
                           zip(fields[5:8], rng.normal(0, baseline * 0.0002, 3))]
            dst.write(" ".join(fields) + "\n")
            dst.write(next(src))
    with (source / "points3D.txt").open() as src, (target / "points3D.txt").open("w") as dst:
        for line in src:
            if not line.strip() or line.startswith("#"):
                dst.write(line)
                continue
            fields = line.split()
            fields[1:4] = [format(float(v) + float(d), ".17g") for v, d in
                           zip(fields[1:4], rng.normal(0, baseline * 0.001, 3))]
            dst.write(" ".join(fields) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["building-18", "house"])
    parser.add_argument("--colite", type=Path)
    parser.add_argument("--colmap", type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--colite-guided-ratio", type=float, default=0.8,
                        help="Production candidate ratio; controlled matching always uses COLMAP's 0.8.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-run", type=Path,
                        help="Reuse an already measured COLMAP run; its timings remain explicitly attributed.")
    parser.add_argument("--phases", nargs="+", choices=["frontend", "controlled", "ba"],
                        default=["frontend", "controlled", "ba"])
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    reference_input_root = args.reference_run.resolve() if args.reference_run else None
    visited_references = set()
    while reference_input_root:
        if reference_input_root in visited_references:
            raise RuntimeError("Reference run chain contains a cycle")
        visited_references.add(reference_input_root)
        manifest = json.loads((reference_input_root / "report.json").read_text())
        inherited = manifest.get("reference_run")
        if not inherited:
            break
        reference_input_root = Path(inherited).resolve()
    colite = str((args.colite or root / "build-wsl-colite-cuda/src/colite/exe/colite").resolve())
    colmap = str((args.colmap or root / "build-wsl-colmap4-cuda132-reference/src/colmap/exe/colmap").resolve())
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError("Output must be empty unless --resume is explicitly specified")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", OPENBLAS_NUM_THREADS="1")
    report = {"device": args.device, "cpu_count": os.cpu_count(), "datasets": {},
              "colite_guided_ratio": args.colite_guided_ratio,
              "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
              "reference_input_root": str(reference_input_root) if reference_input_root else None,
              "notes": ["Wall time includes process startup and model/workspace I/O; no stage cache timing.",
                        "Same feature budget is not identical output count: COLMAP retains multiple orientations.",
                        "COLMAP is a reference, not ground truth; reprojection statistics are recomputed per observation.",
                        "BA uses identical perturbed input, 50 iterations and production loss/solver defaults.",
                        "Controlled mapper uses database relative poses, never final model poses."]}
    environment = {"executables": {exe: hashlib.sha256(Path(exe).read_bytes()).hexdigest()
                                    for exe in [colite, colmap]},
                   "gpu": subprocess.getoutput("nvidia-smi"), "cpu": subprocess.getoutput("lscpu")}
    environment_path = output / "environment.json"
    if args.resume and environment_path.exists():
        previous_environment = json.loads(environment_path.read_text())
        if previous_environment["executables"] != environment["executables"]:
            raise RuntimeError("Executables changed; use a fresh output directory")
    save(environment_path, environment)

    def run(stage, argv):
        stage.mkdir(parents=True, exist_ok=True)
        record = stage / "run.json"
        argv = [str(v) for v in argv]
        if record.exists() and args.resume:
            previous = json.loads(record.read_text())
            if previous["argv"] == argv and previous["exit_code"] == 0:
                return previous
            raise RuntimeError(f"Existing stage differs or failed: {stage}")
        started = time.perf_counter()
        print("RUN", stage, flush=True)
        with (stage / "stdout.log").open("w") as stdout, (stage / "stderr.log").open("w") as stderr:
            process = subprocess.run(["/usr/bin/time", "-v", "-o", str(stage / "resources.txt"), *argv],
                                     cwd=root, env=env, stdout=stdout, stderr=stderr)
        result = dict(argv=argv, wall_seconds=time.perf_counter() - started, exit_code=process.returncode)
        save(record, result)
        if process.returncode:
            raise RuntimeError(f"Failed {stage}: {(stage / 'stderr.log').read_text()[-2500:]}")
        print("DONE", stage.name, round(result["wall_seconds"], 3), flush=True)
        return result

    for dataset in args.datasets:
        folder = output / dataset
        image_path = root / "data" / dataset
        reference_folder = reference_input_root / dataset if reference_input_root else folder
        reference_text = reference_folder / "colmap/txt"
        db = reference_folder / "colmap/database.db"
        ws = folder / "colite/workspace"
        timings = {}
        dataset_report = {"timings": timings, "models": {}}
        report["datasets"][dataset] = dataset_report
        gpu = "1" if args.device == "gpu" else "0"
        if "frontend" in args.phases:
            for engine in (["colmap", "colite"] if dataset == "building-18" else ["colite", "colmap"]):
                if engine == "colmap" and args.reference_run:
                    reference_report = json.loads((args.reference_run / "report.json").read_text())
                    if reference_report["device"] != args.device:
                        raise ValueError("Reference device differs from this run")
                    reference = reference_report["datasets"][dataset]
                    for name in ("feature", "match", "mapper"):
                        timings["colmap_" + name] = reference["timings"]["colmap_" + name]
                    dataset_report["colmap_features_matches"] = database_stats(db)
                    dataset_report["models"]["colmap"] = model_stats(reference_text)
                    continue
                if engine == "colmap":
                    db.parent.mkdir(parents=True, exist_ok=True)
                    feature_args = [colmap, "feature_extractor", "--database_path", db, "--image_path", image_path,
                        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
                        "--FeatureExtraction.use_gpu", gpu, "--FeatureExtraction.num_threads", "8",
                        "--FeatureExtraction.max_image_size", "-1", "--SiftExtraction.max_num_features", "8192"]
                    match_args = [colmap, "exhaustive_matcher", "--database_path", db,
                        "--FeatureMatching.use_gpu", gpu, "--FeatureMatching.guided_matching", "1",
                        "--TwoViewGeometry.compute_relative_pose", "1", "--TwoViewGeometry.random_seed", "0"]
                    if args.device == "cpu":
                        match_args += ["--SiftMatching.cpu_brute_force_matcher", "1"]
                    model_path = folder / "colmap/sparse"
                    model_path.mkdir(parents=True, exist_ok=True)
                    mapper_args = [colmap, "mapper", "--database_path", db, "--image_path", image_path,
                                   "--output_path", model_path]
                else:
                    feature_args = [colite, "feature_extractor", "--Colite.workspace_path", ws, "--image_path", image_path,
                        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
                        "--FeatureExtraction.num_threads", "8", "--SiftExtraction.max_num_features", "8192",
                        "--Colite.feature_extractor", "cuda_vlfeat" if args.device == "gpu" else "vlsift"]
                    match_args = [colite, "exhaustive_matcher", "--Colite.workspace_path", ws,
                        "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.use_gpu", "1" if args.device == "gpu" else "0", "--FeatureMatching.guided_matching", "1",
                        "--Colite.guided_match_ratio", str(args.colite_guided_ratio)]
                    mapper_args = [colite, "mapper", "--Colite.workspace_path", ws]
                for name, argv in [("feature", feature_args), ("match", match_args), ("mapper", mapper_args)]:
                    timings[engine + "_" + name] = run(folder / engine / name, argv)["wall_seconds"]
                if engine == "colmap":
                    text = reference_text
                    text.mkdir(parents=True, exist_ok=True)
                    run(folder / engine / "convert", [colmap, "model_converter", "--input_path", model_path / "0",
                        "--output_path", text, "--output_type", "TXT"])
                    dataset_report["colmap_features_matches"] = database_stats(db)
                else:
                    text = ws / "sparse"
                    dataset_report["colite_features_matches"] = workspace_stats(ws)
                dataset_report["models"][engine] = model_stats(text)
                save(output / "report.json", report)
            run(folder / "compare", [colite, "compare_colmap", "--reference_path", reference_text,
                "--candidate_path", ws / "sparse", "--output_path", folder / "pose_compare"])
        if "controlled" in args.phases:
            shared = folder / "controlled/workspace"
            run(folder / "controlled/import", [sys.executable, root / "scripts/convert_colmap_database_to_colite_workspace.py",
                "--database", db, "--image_path", image_path, "--workspace", shared,
                "--include_descriptors", "--compute_pair_angles"])
            timings["colite_shared_mapper"] = run(folder / "controlled/mapper", [colite, "mapper", "--Colite.workspace_path", shared])["wall_seconds"]
            dataset_report["models"]["colite_shared_mapper"] = model_stats(shared / "sparse")
            run(folder / "controlled/compare", [colite, "compare_colmap", "--reference_path", reference_text,
                "--candidate_path", shared / "sparse", "--output_path", folder / "controlled/pose_compare"])
            timings["colite_shared_match"] = run(folder / "controlled/match", [colite, "exhaustive_matcher",
                "--Colite.workspace_path", shared, "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.use_gpu", "1" if args.device == "gpu" else "0", "--FeatureMatching.guided_matching", "1",
                "--Colite.guided_match_ratio", "0.8"])["wall_seconds"]
            dataset_report["shared_matches"] = workspace_stats(shared)
            run(folder / "controlled/raw_compare", [sys.executable, root / "scripts/compare_colmap_colite_raw_matches.py",
                "--database", db, "--colite", shared / "matches/raw_matches.tsv", "--json", folder / "controlled/raw_overlap.json"])
            run(folder / "controlled/verified_compare", [sys.executable, root / "scripts/compare_colmap_colite_raw_matches.py",
                "--database", db, "--colite", shared / "matches/verified_matches.tsv", "--verified",
                "--json", folder / "controlled/verified_overlap.json"])
            save(output / "report.json", report)
        if "ba" in args.phases:
            seed = folder / "ba/seed"
            if not seed.exists():
                perturb_model(reference_text, seed)
            dataset_report["models"]["ba_seed"] = model_stats(seed)
            for engine, executable in [("colmap", colmap), ("colite", colite)]:
                result_path = folder / "ba" / engine
                result_path.mkdir(parents=True, exist_ok=True)
                timings[engine + "_ba"] = run(folder / "ba" / (engine + "_run"), [executable, "bundle_adjuster",
                    "--input_path", seed, "--output_path", result_path, "--BundleAdjustmentCeres.max_num_iterations", "50"])["wall_seconds"]
                if engine == "colmap":
                    text = folder / "ba/colmap_txt"
                    text.mkdir(parents=True, exist_ok=True)
                    run(folder / "ba/convert", [colmap, "model_converter", "--input_path", result_path,
                        "--output_path", text, "--output_type", "TXT"])
                else:
                    text = result_path
                dataset_report["models"][engine + "_ba"] = model_stats(text)
                save(output / "report.json", report)
    save(output / "report.json", report)


if __name__ == "__main__":
    main()
