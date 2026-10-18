#!/usr/bin/env python3
"""Compare two extractors with the same image size, matcher and mapper.

Uses existing images and file workspaces only. Requires Linux and numpy for
the existing independent sparse-model statistics helper.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

from benchmark_sfm import model_stats, workspace_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--before-workspace", type=Path,
                        help="Reuse immutable baseline features/matches and rerun only the common mapper")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a fresh directory")
    args.output.mkdir(parents=True)
    before, after = str(args.before.resolve()), str(args.after.resolve())
    report = {"max_image_size": args.max_image_size,
              "common_matcher_mapper": after,
              "sha256": {exe: hashlib.sha256(Path(exe).read_bytes()).hexdigest() for exe in (before, after)},
              "variants": {}}
    for label, extractor in (("before", before), ("after", after)):
        workspace = (args.output / label / "workspace").resolve()
        commands = [
            ("extract", [extractor, "feature_extractor", "--image_path", str(args.images.resolve()),
                "--Colite.workspace_path", str(workspace), "--Colite.feature_extractor", "cuda_vlfeat",
                "--FeatureExtraction.max_image_size", str(args.max_image_size),
                "--FeatureExtraction.num_threads", "8", "--SiftExtraction.max_num_features", "8192",
                "--ImageReader.single_camera", "1"]),
            ("match", [after, "exhaustive_matcher", "--Colite.workspace_path", str(workspace),
                "--FeatureMatching.use_gpu", "1", "--FeatureMatching.guided_matching", "1"]),
            ("mapper", [after, "mapper", "--Colite.workspace_path", str(workspace)])]
        result = {"runs": []}
        if label == "before" and args.before_workspace:
            source = args.before_workspace.resolve()
            workspace.mkdir(parents=True)
            sources = [source / "cameras.txt", source / "images.tsv",
                       *sorted((source / "features").glob("*.bin")),
                       source / "matches/raw_matches.tsv", source / "matches/verified_matches.tsv"]
            for item in sources:
                target = workspace / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(item, target)
                except OSError:
                    shutil.copyfile(item, target)
            result["reused_frontend"] = str(source)
            commands = commands[-1:]
        report["variants"][label] = result
        for stage, argv in commands:
            root = args.output / label / stage
            root.mkdir(parents=True)
            started = time.perf_counter()
            print("RUN", label, stage, flush=True)
            with (root / "stdout.log").open("w") as stdout, (root / "stderr.log").open("w") as stderr:
                process = subprocess.run(argv, stdout=stdout, stderr=stderr)
            result["runs"].append({"stage": stage, "argv": argv, "returncode": process.returncode,
                                   "wall_seconds": time.perf_counter() - started})
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if process.returncode:
                raise RuntimeError(f"{label}/{stage} failed; inspect {root}")
        result["frontend"] = workspace_stats(workspace)
        result["model"] = model_stats(workspace / "sparse")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DONE", label, result["frontend"], result["model"]["images"], flush=True)


if __name__ == "__main__":
    main()
