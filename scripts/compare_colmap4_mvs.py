#!/usr/bin/env python3
"""Run isolated COLITE/COLMAP 4.0 MVS tools on identical inputs.

The script intentionally compares process-level behavior and stable artifact
statistics. Geometry distances can be added through --evaluator (for example a
project-specific MeshLab/Open3D wrapper) without coupling COLITE to Python mesh
packages.
"""

import argparse
import json
import pathlib
import struct
import subprocess
import time


def run(command):
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def ply_header(path):
    result = {"path": str(path), "bytes": path.stat().st_size}
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("format "):
                result["format"] = line.split()[1]
            elif line.startswith("element vertex "):
                result["vertices"] = int(line.split()[2])
            elif line.startswith("element face "):
                result["faces"] = int(line.split()[2])
            elif line == "end_header":
                break
    return result


def png_header(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return {"path": str(path), "bytes": path.stat().st_size,
            "width": width, "height": height}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colite", required=True)
    parser.add_argument("--colmap", required=True,
                        help="COLMAP 4.0 executable, not the 3.11 reference")
    parser.add_argument("--workspace", required=True,
                        help="Dense COLMAP workspace with fused.ply and sparse/")
    parser.add_argument("--mesh", help="Common PLY for simplifier/texturer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stages", default="poisson,delaunay,simplifier,texturer")
    parser.add_argument("--target-ratio", type=float, default=0.5)
    args = parser.parse_args()

    workspace = pathlib.Path(args.workspace).resolve()
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    report = {"workspace": str(workspace), "runs": {}, "artifacts": {}}
    fused = workspace / "fused.ply"

    if "poisson" in stages:
        for implementation, executable in (("colite", args.colite),
                                             ("colmap", args.colmap)):
            target = output / f"{implementation}-poisson.ply"
            command = [executable, "poisson_mesher", "--input_path", str(fused),
                       "--output_path", str(target),
                       "--PoissonMeshing.depth", "8",
                       "--PoissonMeshing.point_weight", "1",
                       "--PoissonMeshing.color", "1",
                       "--PoissonMeshing.trim", "0"]
            report["runs"][f"{implementation}_poisson"] = run(command)
            if target.exists():
                report["artifacts"][f"{implementation}_poisson"] = ply_header(target)

    if "delaunay" in stages:
        colite_target = output / "colite-delaunay.ply"
        colmap_target = output / "colmap-delaunay.ply"
        report["runs"]["colite_delaunay"] = run([
            args.colite, "delaunay_mesher", "--input_path", str(workspace),
            "--output_path", str(colite_target),
            "--DelaunayMeshing.quality_regularization", "0",
            "--DelaunayMeshing.distance_sigma_factor", "2"])
        report["runs"]["colmap_delaunay"] = run([
            args.colmap, "delaunay_mesher", "--input_path", str(workspace),
            "--output_path", str(colmap_target), "--input_type", "dense",
            "--DelaunayMeshing.quality_regularization", "0",
            "--DelaunayMeshing.distance_sigma_factor", "2"])
        for name, target in (("colite_delaunay", colite_target),
                             ("colmap_delaunay", colmap_target)):
            if target.exists():
                report["artifacts"][name] = ply_header(target)

    common_mesh = pathlib.Path(args.mesh).resolve() if args.mesh else fused
    if "simplifier" in stages:
        for implementation, executable in (("colite", args.colite),
                                             ("colmap", args.colmap)):
            target = output / f"{implementation}-simplified.ply"
            ratio_name = ("--MeshSimplification.target_ratio" if implementation == "colite"
                          else "--MeshSimplification.target_face_ratio")
            report["runs"][f"{implementation}_simplifier"] = run([
                executable, "mesh_simplifier", "--input_path", str(common_mesh),
                "--output_path", str(target), ratio_name, str(args.target_ratio)])
            if target.exists():
                report["artifacts"][f"{implementation}_simplifier"] = ply_header(target)

    if "texturer" in stages:
        colite_dir = output / "colite-textured"
        colmap_dir = output / "colmap-textured"
        report["runs"]["colite_texturer"] = run([
            args.colite, "mesh_texturer", "--input_path", str(common_mesh),
            "--output_path", str(colite_dir), "--model_path", str(workspace / "sparse"),
            "--image_path", str(workspace / "images"),
            "--MeshTextureMapping.atlas_size", "512"])
        report["runs"]["colmap_texturer"] = run([
            args.colmap, "mesh_texturer", "--input_path", str(common_mesh),
            "--output_path", str(colmap_dir), "--workspace_path", str(workspace),
            "--output_type", "TXT"])
        for name, target in (("colite_texture", colite_dir / "texture.png"),
                             ("colmap_texture", colmap_dir / "texture.png")):
            if target.exists():
                report["artifacts"][name] = png_header(target)

    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    if any(item["returncode"] != 0 for item in report["runs"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
