#!/usr/bin/env python3
"""Audit how a COLMAP database's verified edges feed a final sparse model."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import read_model


MAX_IMAGE_ID = 2_147_483_647


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first, second):
        self.add(first)
        self.add(second)
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]


def decode_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def read_matches(blob, rows, cols):
    if blob is None or rows == 0:
        return np.empty((0, 2), dtype=np.uint32)
    matches = np.frombuffer(blob, dtype="<u4")
    return matches.reshape((rows, cols))[:, :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    _, _, points = read_model(args.model)
    owners = {}
    track_lengths = Counter()
    for point_id, (_, track) in points.items():
        track_lengths[len(track)] += 1
        for observation in track:
            owners[observation] = point_id

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    image_names = dict(connection.execute("SELECT image_id, name FROM images"))
    keypoint_rows = dict(connection.execute("SELECT image_id, rows FROM keypoints"))
    raw_pairs, raw_edges = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM matches"
    ).fetchone()

    union_find = UnionFind()
    endpoint_coverage = Counter()
    verified_pairs = 0
    verified_edges = 0
    geometry_configs = Counter()
    pair_counts = []
    degree = Counter()
    for pair_id, rows, cols, blob, config in connection.execute(
        "SELECT pair_id, rows, cols, data, config FROM two_view_geometries"
    ):
        if config == 0 or rows == 0 or blob is None:
            continue
        geometry_configs[int(config)] += int(rows)
        image_id1, image_id2 = decode_pair_id(pair_id)
        matches = read_matches(blob, rows, cols)
        verified_pairs += 1
        verified_edges += len(matches)
        pair_counts.append(len(matches))
        for idx1, idx2 in matches:
            first = (image_id1, int(idx1))
            second = (image_id2, int(idx2))
            union_find.union(first, second)
            degree[first] += 1
            degree[second] += 1
            first_owner = owners.get(first)
            second_owner = owners.get(second)
            if first_owner is None and second_owner is None:
                endpoint_coverage["neither"] += 1
            elif first_owner is None or second_owner is None:
                endpoint_coverage["one"] += 1
            elif first_owner == second_owner:
                endpoint_coverage["same_point"] += 1
            else:
                endpoint_coverage["different_points"] += 1
    connection.close()

    component_sizes = Counter()
    component_images = {}
    component_owners = {}
    for observation in union_find.parent:
        root = union_find.find(observation)
        component_sizes[root] += 1
        component_images.setdefault(root, Counter())[observation[0]] += 1
        owner = owners.get(observation)
        if owner is not None:
            component_owners.setdefault(root, set()).add(owner)
    conflicted_components = sum(
        any(count > 1 for count in image_counts.values())
        for image_counts in component_images.values()
    )
    multi_owner_components = sum(
        len(component_owners.get(root, ())) > 1
        for root in component_sizes
    )
    largest_root = max(component_sizes, key=component_sizes.get)

    num_points = len(points)
    num_observations = len(owners)
    result = {
        "database": str(args.database),
        "model": str(args.model),
        "images": len(image_names),
        "keypoints": sum(keypoint_rows.values()),
        "keypoints_per_image_min": min(keypoint_rows.values()),
        "keypoints_per_image_median": float(np.median(list(keypoint_rows.values()))),
        "keypoints_per_image_max": max(keypoint_rows.values()),
        "raw_pairs": raw_pairs,
        "raw_edges": raw_edges,
        "verified_pairs": verified_pairs,
        "verified_edges": verified_edges,
        "verified_edges_by_config": dict(sorted(geometry_configs.items())),
        "verified_pair_matches_median": float(np.median(pair_counts)),
        "verified_pair_matches_p95": float(np.percentile(pair_counts, 95)),
        "verified_nodes": len(union_find.parent),
        "verified_node_degree_median": float(np.median(list(degree.values()))),
        "verified_node_degree_p95": float(np.percentile(list(degree.values()), 95)),
        "components": len(component_sizes),
        "conflicted_components": conflicted_components,
        "multi_owner_components": multi_owner_components,
        "largest_component_nodes": component_sizes[largest_root],
        "largest_component_images": len(component_images[largest_root]),
        "largest_component_max_nodes_per_image": max(
            component_images[largest_root].values()
        ),
        "largest_component_model_points": len(
            component_owners.get(largest_root, ())
        ),
        "edge_endpoint_coverage": dict(endpoint_coverage),
        "model_points": num_points,
        "model_observations": num_observations,
        "mean_track": num_observations / num_points,
        "track_length_histogram": dict(sorted(track_lengths.items())),
        "model_observations_in_verified_graph": sum(
            observation in union_find.parent for observation in owners
        ),
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
