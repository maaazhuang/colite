#!/usr/bin/env python3
"""Measure triangle support and final-track ownership of verified edges."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import read_model


def read_edges(path):
    edges = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            edges.append((
                (int(fields[0]), int(fields[11])),
                (int(fields[1]), int(fields[12])),
            ))
    return edges


def ownership_class(owners, first, second):
    first_owner = owners.get(first)
    second_owner = owners.get(second)
    if first_owner is None and second_owner is None:
        return "neither"
    if first_owner is None or second_owner is None:
        return "one"
    return "same" if first_owner == second_owner else "different"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    _, _, points = read_model(args.model)
    owners = {
        observation: point_id
        for point_id, (_, track) in points.items()
        for observation in track
    }
    edges = read_edges(args.verified)
    adjacency = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    support_histogram = Counter()
    support_by_class = defaultdict(Counter)
    degree_by_class = defaultdict(list)
    for first, second in edges:
        support = len(adjacency[first].intersection(adjacency[second]))
        support_bucket = str(min(support, 3)) if support < 3 else "3+"
        edge_class = ownership_class(owners, first, second)
        support_histogram[support_bucket] += 1
        support_by_class[edge_class][support_bucket] += 1
        degree_by_class[edge_class].append(
            len(adjacency[first]) + len(adjacency[second])
        )

    result = {
        "verified_edges": len(edges),
        "verified_nodes": len(adjacency),
        "triangle_support_histogram": dict(support_histogram),
        "triangle_support_by_final_ownership": {
            edge_class: dict(counts)
            for edge_class, counts in sorted(support_by_class.items())
        },
        "endpoint_degree_sum_by_final_ownership": {
            edge_class: {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for edge_class, values in sorted(degree_by_class.items())
        },
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
