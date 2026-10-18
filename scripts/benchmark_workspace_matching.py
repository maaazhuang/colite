#!/usr/bin/env python3
"""Repeat matching on immutable existing workspace features, without a database."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colite", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--matcher-options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        raise ValueError("Use a fresh output directory and a positive repeat count")
    args.output.mkdir(parents=True)
    report = {"executable": str(args.colite.resolve()),
              "sha256": hashlib.sha256(args.colite.read_bytes()).hexdigest(),
              "source": str(args.workspace.resolve()), "runs": []}
    options = args.matcher_options or ["--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", "1", "--Colite.guided_match_ratio", "0.8"]
    for index in range(args.repeat):
        root = args.output / str(index)
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        sources = [args.workspace / "cameras.txt", args.workspace / "images.tsv",
                   *sorted((args.workspace / "features").glob("*.bin"))]
        for source in sources:
            target = workspace / source.relative_to(args.workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        command = [str(args.colite.resolve()), "exhaustive_matcher",
                   "--Colite.workspace_path", str(workspace.resolve()), *options]
        print("RUN", index, flush=True)
        started = time.perf_counter()
        with (root / "stdout.log").open("w") as stdout, (root / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        record = {"argv": command, "returncode": process.returncode,
                  "wall_seconds": time.perf_counter() - started}
        if process.returncode:
            raise RuntimeError(f"Matching failed; inspect {root / 'stderr.log'}")
        record["summary"] = dict(line.split(maxsplit=1) for line in
            (workspace / "matches/summary.txt").read_text().splitlines())
        for name in ("raw_matches.tsv", "verified_matches.tsv"):
            record[name + "_sha256"] = hashlib.sha256((workspace / "matches" / name).read_bytes()).hexdigest()
        report["runs"].append(record)
        report["median_seconds"] = statistics.median(run["wall_seconds"] for run in report["runs"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DONE", index, round(record["wall_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
