#!/usr/bin/env python3
"""Compare COLMAP database raw or verified matches with a Colite TSV."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


MAX_IMAGE_ID = 2_147_483_647


def decode_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def read_colmap(database, verified):
    result = {}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    table = "two_view_geometries" if verified else "matches"
    for pair_id, rows, cols, blob in connection.execute(
        f"SELECT pair_id, rows, cols, data FROM {table} ORDER BY pair_id"
    ):
        pair = decode_pair_id(pair_id)
        if rows == 0 or blob is None:
            result[pair] = set()
            continue
        values = np.frombuffer(blob, dtype="<u4").reshape((rows, cols))
        result[pair] = {
            (int(values[index, 0]), int(values[index, 1]))
            for index in range(rows)
        }
    connection.close()
    return result


def read_colite(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            image1, image2 = map(int, fields[:2])
            idx1, idx2 = map(int, fields[-2:])
            result.setdefault((image1, image2), set()).add((idx1, idx2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--colite", type=Path, required=True)
    parser.add_argument("--verified", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    colmap = read_colmap(args.database, args.verified)
    colite = read_colite(args.colite)
    pairs = sorted(set(colmap) | set(colite))
    same = colmap_only = colite_only = 0
    pair_jaccards = []
    pair_exact = 0
    different_pairs = []
    for pair in pairs:
        reference = colmap.get(pair, set())
        candidate = colite.get(pair, set())
        intersection = reference & candidate
        union = reference | candidate
        same += len(intersection)
        colmap_only += len(reference - candidate)
        colite_only += len(candidate - reference)
        pair_jaccards.append(len(intersection) / len(union) if union else 1.0)
        pair_exact += reference == candidate
        if reference != candidate:
            different_pairs.append({
                "image_id1": pair[0],
                "image_id2": pair[1],
                "colmap": len(reference),
                "colite": len(candidate),
                "same": len(intersection),
                "colmap_only": len(reference - candidate),
                "colite_only": len(candidate - reference),
            })

    result = {
        "pairs": len(pairs),
        "exact_pairs": pair_exact,
        "colmap_edges": same + colmap_only,
        "colite_edges": same + colite_only,
        "same_edges": same,
        "colmap_only_edges": colmap_only,
        "colite_only_edges": colite_only,
        "edge_recall": same / (same + colmap_only),
        "edge_precision": same / (same + colite_only),
        "edge_jaccard": same / (same + colmap_only + colite_only),
        "pair_jaccard_median": float(np.median(pair_jaccards)),
        "pair_jaccard_p05": float(np.percentile(pair_jaccards, 5)),
        "different_pairs": different_pairs,
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
