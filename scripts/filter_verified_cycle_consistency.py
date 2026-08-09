#!/usr/bin/env python3
"""Filter unsupported verified bridge edges that create image conflicts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


class ImageUnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.images = {}

    def add(self, observation):
        if observation not in self.parent:
            self.parent[observation] = observation
            self.rank[observation] = 0
            self.images[observation] = {observation[0]}

    def find(self, observation):
        root = observation
        while self.parent[root] != root:
            root = self.parent[root]
        while observation != root:
            parent = self.parent[observation]
            self.parent[observation] = root
            observation = parent
        return root

    def would_conflict(self, first, second):
        first = self.find(first)
        second = self.find(second)
        return first != second and not self.images[first].isdisjoint(
            self.images[second]
        )

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
        self.images[first].update(self.images[second])
        del self.images[second]
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    header = []
    entries = []
    adjacency = defaultdict(set)
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                header.append(line)
                continue
            fields = line.split()
            first = (int(fields[0]), int(fields[11]))
            second = (int(fields[1]), int(fields[12]))
            entries.append((line, first, second))
            adjacency[first].add(second)
            adjacency[second].add(first)

    ranked = []
    for index, (_, first, second) in enumerate(entries):
        support = len(adjacency[first].intersection(adjacency[second]))
        degree_sum = len(adjacency[first]) + len(adjacency[second])
        ranked.append((-support, -degree_sum, first, second, index))
    ranked.sort()

    union_find = ImageUnionFind()
    for _, first, second in entries:
        union_find.add(first)
        union_find.add(second)

    retained = [False] * len(entries)
    rejected_unsupported_conflicts = 0
    retained_supported = 0
    retained_unsupported = 0
    for negative_support, _, first, second, index in ranked:
        support = -negative_support
        if support == 0 and union_find.would_conflict(first, second):
            rejected_unsupported_conflicts += 1
            continue
        retained[index] = True
        union_find.union(first, second)
        if support > 0:
            retained_supported += 1
        else:
            retained_unsupported += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.writelines(header)
        for keep, (line, _, _) in zip(retained, entries):
            if keep:
                stream.write(line)

    result = {
        "input_edges": len(entries),
        "retained_edges": sum(retained),
        "retained_supported_edges": retained_supported,
        "retained_unsupported_edges": retained_unsupported,
        "rejected_unsupported_conflict_bridges": (
            rejected_unsupported_conflicts
        ),
        "output_components": len(union_find.images),
        "largest_output_component_images": max(
            len(images) for images in union_find.images.values()
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
