#!/usr/bin/env python3
"""Quantify final sparse-track fragmentation inside verified-match components."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_sfm_geometric_merges import read_model


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
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
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    _, _, points = read_model(args.model)
    owners = {}
    union_find = UnionFind()
    for point_id, (_, track) in points.items():
        for observation in track:
            owners[observation] = point_id
            union_find.add(observation)

    num_edges = 0
    with args.verified.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            first = (int(fields[0]), int(fields[11]))
            second = (int(fields[1]), int(fields[12]))
            union_find.union(first, second)
            num_edges += 1

    nodes_by_component = defaultdict(list)
    for node in union_find.parent:
        nodes_by_component[union_find.find(node)].append(node)

    multi_owner_components = 0
    conflict_free_multi_owner_components = 0
    multi_owner_points = set()
    conflict_free_multi_owner_points = set()
    unassigned_unique_image_capacity = 0
    fragmented_owner_excess = 0
    owner_count_histogram = Counter()
    image_multiplicity_histogram = Counter()
    largest = []
    for nodes in nodes_by_component.values():
        point_ids = {owners[node] for node in nodes if node in owners}
        owner_count_histogram[len(point_ids)] += 1
        image_counts = Counter(node[0] for node in nodes)
        max_image_multiplicity = max(image_counts.values())
        image_multiplicity_histogram[max_image_multiplicity] += 1
        assigned_images = {
            node[0] for node in nodes if node in owners
        }
        unassigned_unique_image_capacity += sum(
            1
            for image_id in image_counts
            if image_id not in assigned_images
        )
        if len(point_ids) > 1:
            multi_owner_components += 1
            fragmented_owner_excess += len(point_ids) - 1
            multi_owner_points.update(point_ids)
            if max_image_multiplicity == 1:
                conflict_free_multi_owner_components += 1
                conflict_free_multi_owner_points.update(point_ids)
        largest.append(
            {
                "nodes": len(nodes),
                "images": len(image_counts),
                "owners": len(point_ids),
                "max_image_multiplicity": max_image_multiplicity,
                "assigned_nodes": sum(node in owners for node in nodes),
            }
        )

    largest.sort(key=lambda item: (item["owners"], item["nodes"]), reverse=True)
    result = {
        "verified_edges": num_edges,
        "verified_nodes_including_model_singletons": len(union_find.parent),
        "components": len(nodes_by_component),
        "model_points": len(points),
        "model_observations": len(owners),
        "multi_owner_components": multi_owner_components,
        "points_in_multi_owner_components": len(multi_owner_points),
        "fragmented_owner_excess_upper_bound": fragmented_owner_excess,
        "conflict_free_multi_owner_components": (
            conflict_free_multi_owner_components
        ),
        "points_in_conflict_free_multi_owner_components": len(
            conflict_free_multi_owner_points
        ),
        "unassigned_unique_image_capacity_upper_bound": (
            unassigned_unique_image_capacity
        ),
        "owner_count_histogram": dict(sorted(owner_count_histogram.items())),
        "max_image_multiplicity_histogram": dict(
            sorted(image_multiplicity_histogram.items())
        ),
        "largest_fragmented_components": largest[:20],
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
