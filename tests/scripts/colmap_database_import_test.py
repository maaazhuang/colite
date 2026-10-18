#!/usr/bin/env python3
"""Check typed COLMAP 4.0 imports, including the native Colite feature reader."""

import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    print("numpy is required for the COLMAP database import test")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import convert_colmap_database_to_colite_workspace as importer


class ColmapDatabaseImportTest(unittest.TestCase):
    def fixture(self, path, feature_type=0, malformed=False):
        source = np.array([[10 + i, 20 + i, 0, -4, 2, 0]
                           for i in range(8)], dtype="<f4")
        descriptors = np.eye(8, 128, dtype="<f4")
        if feature_type == 0:
            descriptors = (descriptors * 255).astype("u1")
        raw = descriptors.tobytes()
        with sqlite3.connect(path) as database:
            database.executescript("""
                CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
                CREATE TABLE images(image_id,name,camera_id);
                CREATE TABLE keypoints(image_id,rows,cols,data);
                CREATE TABLE descriptors(image_id,type,rows,cols,data);
                CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config,qvec,tvec);
            """)
            database.execute("INSERT INTO cameras VALUES(3,0,640,480,?,1)",
                             (struct.pack("<ddd", 500, 320, 240),))
            for image_id in (7, 23):
                database.execute("INSERT INTO images VALUES(?,?,3)",
                                 (image_id, f"subfolder/{image_id}.jpg"))
                database.execute("INSERT INTO keypoints VALUES(?,8,6,?)",
                                 (image_id, source.tobytes()))
                database.execute("INSERT INTO descriptors VALUES(?,?,8,?,?)",
                                 (image_id, feature_type, descriptors.shape[1] * descriptors.itemsize,
                                  raw[:-1] if malformed else raw))
        return source, descriptors

    def run_import(self, database, workspace, include=True):
        args = [sys.executable, str(ROOT / "scripts/convert_colmap_database_to_colite_workspace.py"),
                "--database", str(database), "--image_path", str(database.parent / "images"),
                "--workspace", str(workspace)]
        if include:
            args.append("--include_descriptors")
        return subprocess.run(args, capture_output=True, text=True)

    def test_typed_import_is_readable_by_native_matching(self):
        executable = os.environ.get("COLITE_EXE")
        self.assertTrue(executable and Path(executable).is_file(), "Set COLITE_EXE to the native executable")
        for feature_type in (0, 1, 2):
            with self.subTest(feature_type=feature_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, descriptors = self.fixture(root / "source.db", feature_type)
                workspace = root / "workspace"
                imported = self.run_import(root / "source.db", workspace)
                self.assertEqual(imported.returncode, 0, imported.stderr)
                payload = (workspace / "features/7.bin").read_bytes()
                header = struct.unpack_from("<IIIIIIQQQQQ", payload)
                self.assertEqual(header[:4], (0x434C4654, 2, feature_type + 1, 1))
                self.assertEqual(header[6:10], (8, 8, 128, 8 * 128 * 4))
                keypoints = np.frombuffer(payload, "<f4", 8 * 5, 64).reshape(8, 5)
                np.testing.assert_array_equal(keypoints[:, :2], source[:, :2])
                np.testing.assert_allclose(keypoints[:, 2], 3.0)
                np.testing.assert_allclose(keypoints[:, 3], 90.0)
                actual = np.frombuffer(payload, "<f4", offset=64 + 8 * 5 * 4).reshape(8, 128)
                np.testing.assert_array_equal(actual, descriptors / 512 if feature_type == 0 else descriptors)
                # Native reader checks the checksum and feature-family contract;
                # matching consumes the descriptors, not just our Python decoder.
                matcher = "SIFT_BRUTEFORCE" if feature_type == 0 else "ALIKED_BRUTEFORCE"
                matched = subprocess.run([executable, "exhaustive_matcher", "--Colite.workspace_path",
                                          str(workspace), "--FeatureMatching.type", matcher],
                                         capture_output=True, text=True)
                self.assertEqual(matched.returncode, 0, matched.stderr)
                self.assertTrue((workspace / "matches/raw_matches.tsv").is_file())

    def test_unsupported_type_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db", 99)
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported COLMAP descriptor type", result.stderr)
            self.assertFalse((root / "workspace").exists())

    def test_truncated_blob_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db", 1, malformed=True)
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid database blob length", result.stderr)
            self.assertFalse((root / "workspace").exists())

    def test_similarity_orientation_units(self):
        source = np.array([[12.5, 24.5, 2.0, -np.pi / 2]], dtype="<f4")
        actual = importer.convert_keypoints(source)
        np.testing.assert_array_equal(actual[0, :3], source[0, :3])
        self.assertAlmostEqual(float(actual[0, 3]), 270.0, places=4)

    def test_source_sift_l2_metadata_preserves_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature.bin"
            keypoints = np.zeros((1, 5), dtype="<f4")
            descriptors = np.full((1, 128), 127 / 512, dtype="<f4")
            importer.write_feature_v2(path, keypoints, descriptors, 0, "L2")
            payload = path.read_bytes()
            self.assertEqual(struct.unpack_from("<IIIIII", payload)[-1], 3)
            self.assertEqual(payload[64 + 20:], descriptors.tobytes())

    def test_existing_workspace_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            workspace = root / "workspace"
            workspace.mkdir()
            sentinel = workspace / "keep.bin"
            sentinel.write_bytes(b"existing workspace")
            result = self.run_import(root / "source.db", workspace)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b"existing workspace")


if __name__ == "__main__":
    unittest.main()
