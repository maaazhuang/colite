import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    source_dir = Path(os.environ.get("COLITE_SOURCE_DIR", ".")).resolve()
    script_path = source_dir / "scripts" / "mvs_sweep.py"
    spec = importlib.util.spec_from_file_location("mvs_sweep", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MvsSweepScriptTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_run_dir(self, name):
        run_dir = self.root / name
        (run_dir / "stereo").mkdir(parents=True)
        (run_dir / "dense").mkdir(parents=True)
        (run_dir / "summary.txt").write_text(
            "\n".join(
                [
                    "photometric_filtered_pixels 12",
                    "normal_refined_pixels 5",
                    "geometric_consistent_pixels 9",
                    "geometric_filtered_pixels 3",
                    "fused_points 7",
                    "fused_observations 9",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "dense" / "fusion_summary.txt").write_text(
            "\n".join(
                [
                    "num_tracks 7",
                    "num_single_observation_tracks 5",
                    "num_multi_observation_tracks 2",
                    "num_observations 9",
                    "mean_observations 1.2857142857",
                    "multi_observation_track_ratio 0.2857142857",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "dense" / "fusion_image_pairs.tsv").write_text(
            "\n".join(
                [
                    "reference_image_id\tsource_image_id\tnum_accepted\tnum_rejected_depth\tnum_rejected_mutual_consistency\tnum_rejected_invalid\tnum_rejected_normal\tnum_rejected_visited\tnum_rejected_duplicate",
                    "1\t2\t3\t4\t5\t6\t7\t8\t9",
                    "1\t3\t2\t1\t0\t1\t2\t3\t4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "stereo" / "patch_match.tsv").write_text(
            "\n".join(
                [
                    "image_id\trequested_backend\tactual_backend\tperturbation_geometric_candidate_rejections\tperturbation_geometric_candidate_acceptances\tperturbation_mean_abs_depth_update\tperturbation_max_abs_depth_update",
                    "1\tauto\tcpu\t10\t2\t0.5\t3.0",
                    "2\tauto\tcpu\t20\t4\t1.5\t5.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "dense" / "fusion_error_distribution.tsv").write_text(
            "\n".join(
                [
                    "metric\trecord\tlabel\tlower\tupper\tcount\tvalue",
                    "relative_depth_error\tpercentile\tp50\t\t\t4\t0.004",
                    "relative_depth_error\tpercentile\tp95\t\t\t4\t0.018",
                    "relative_depth_error\tpercentile\tp99\t\t\t4\t0.024",
                    "reprojection_error_pixels\tpercentile\tp50\t\t\t4\t0.3",
                    "reprojection_error_pixels\tpercentile\tp95\t\t\t4\t1.8",
                    "reprojection_error_pixels\tpercentile\tp99\t\t\t4\t2.4",
                    "normal_angle_degrees\tpercentile\tp50\t\t\t4\t2",
                    "normal_angle_degrees\tpercentile\tp95\t\t\t4\t9",
                    "normal_angle_degrees\tpercentile\tp99\t\t\t4\t12",
                    "view_angle_degrees\tpercentile\tp50\t\t\t4\t3",
                    "view_angle_degrees\tpercentile\tp95\t\t\t4\t15",
                    "view_angle_degrees\tpercentile\tp99\t\t\t4\t21",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "sweep_case.json").write_text(
            json.dumps(
                {
                    "patch_match_geometric_consistency_weight": 0.05,
                    "patch_match_geometric_max_relative_depth_error": 0.01,
                    "normal_refinement_max_relative_depth_difference": 0.005,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_summarize_run_directory_aggregates_key_metrics(self):
        run_dir = self.make_run_dir("case-a")

        row = self.module.summarize_run(run_dir)

        self.assertEqual(row["case"], "case-a")
        self.assertEqual(row["fused_observations"], "9")
        self.assertEqual(row["num_multi_observation_tracks"], "2")
        self.assertEqual(row["pair_num_accepted_sum"], "5")
        self.assertEqual(row["pair_num_rejected_depth_sum"], "5")
        self.assertEqual(row["perturbation_geometric_candidate_rejections_sum"], "30")
        self.assertEqual(row["perturbation_geometric_candidate_acceptances_sum"], "6")
        self.assertEqual(row["perturbation_mean_abs_depth_update_avg"], "1")
        self.assertEqual(row["perturbation_max_abs_depth_update_max"], "5")
        self.assertEqual(row["patch_match_geometric_consistency_weight"], "0.05")
        self.assertEqual(row["patch_match_requested_backend"], "auto")
        self.assertEqual(row["patch_match_actual_backend"], "cpu")
        self.assertEqual(row["fusion_relative_depth_error_p95"], "0.018")
        self.assertEqual(row["fusion_reprojection_error_pixels_p99"], "2.4")
        self.assertEqual(row["fusion_normal_angle_degrees_p50"], "2")
        self.assertEqual(row["fusion_view_angle_degrees_p95"], "15")

    def test_write_summary_tsv_for_multiple_runs(self):
        run_a = self.make_run_dir("case-a")
        run_b = self.make_run_dir("case-b")
        output_path = self.root / "summary.tsv"

        self.module.write_summary([run_a, run_b], output_path)

        with output_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual([row["case"] for row in rows], ["case-a", "case-b"])
        self.assertEqual(rows[0]["pair_num_rejected_normal_sum"], "9")

    def test_build_sweep_cases_creates_stable_case_names(self):
        cases = self.module.build_sweep_cases(
            weights=[0.05],
            relative_depth_errors=[0.01, 0.02],
            normal_depth_thresholds=[0.005],
            prefix="smoke",
        )

        self.assertEqual(
            [case.name for case in cases],
            [
                "smoke-w0p05-gd0p01-nd0p005",
                "smoke-w0p05-gd0p02-nd0p005",
            ],
        )
        self.assertEqual(cases[1].geometric_max_relative_depth_error, 0.02)


if __name__ == "__main__":
    unittest.main()
