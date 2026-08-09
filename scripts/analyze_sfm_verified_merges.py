#!/usr/bin/env python3
"""Measure evidence-backed sparse-point merge opportunities in verified edges."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import project, read_model


def score_position(cameras, images, tracks, xyz):
    errors = []
    for track in tracks:
        for image_id, point2d_idx in track:
            image = images[image_id]
            predicted = project(cameras[image[2]], image, xyz)
            if predicted is None:
                return None
            errors.append(
                float(np.linalg.norm(predicted - image[3][point2d_idx]))
            )
    return max(errors), float(np.sqrt(np.mean(np.square(errors))))


def score_unique_image_position(cameras, images, tracks, xyz):
    """Score a merged track after retaining its best observation per image."""
    best_by_image = {}
    total_observations = 0
    for source_index, track in enumerate(tracks):
        for image_id, point2d_idx in track:
            total_observations += 1
            image = images[image_id]
            predicted = project(cameras[image[2]], image, xyz)
            if predicted is None:
                return None
            error = float(
                np.linalg.norm(predicted - image[3][point2d_idx])
            )
            previous = best_by_image.get(image_id)
            candidate = (error, source_index, point2d_idx)
            if previous is None or candidate < previous:
                best_by_image[image_id] = candidate
    errors = [item[0] for item in best_by_image.values()]
    if not errors:
        return None
    source_counts = Counter(item[1] for item in best_by_image.values())
    return (
        max(errors),
        float(np.sqrt(np.mean(np.square(errors)))),
        len(errors),
        total_observations - len(errors),
        source_counts[0],
        source_counts[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--max_error", type=float, default=4.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    cameras, images, points = read_model(args.model)
    owners = {}
    for point_id, (_, track) in points.items():
        for observation in track:
            owners[observation] = point_id

    support = Counter()
    completion_candidates = {}
    completion_same_image_conflicts = set()
    completion_conflict_candidates = {}
    with args.verified.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            first = owners.get((int(fields[0]), int(fields[11])))
            second = owners.get((int(fields[1]), int(fields[12])))
            if (first is None) != (second is None):
                owner = second if first is None else first
                target = (
                    (int(fields[0]), int(fields[11]))
                    if first is None
                    else (int(fields[1]), int(fields[12]))
                )
                owner_xyz, owner_track = points[owner]
                if target[0] in {image_id for image_id, _ in owner_track}:
                    completion_same_image_conflicts.add(target)
                    target_image = images[target[0]]
                    predicted = project(
                        cameras[target_image[2]], target_image, owner_xyz
                    )
                    if predicted is not None:
                        target_error = float(
                            np.linalg.norm(
                                predicted - target_image[3][target[1]]
                            )
                        )
                        existing_idx = next(
                            point2d_idx
                            for image_id, point2d_idx in owner_track
                            if image_id == target[0]
                        )
                        existing_error = float(
                            np.linalg.norm(
                                predicted
                                - target_image[3][existing_idx]
                            )
                        )
                        per_target = completion_conflict_candidates.setdefault(
                            target, {}
                        )
                        previous = per_target.get(owner)
                        candidate = (target_error, existing_error)
                        if previous is None or candidate < previous:
                            per_target[owner] = candidate
                else:
                    target_image = images[target[0]]
                    predicted = project(
                        cameras[target_image[2]], target_image, owner_xyz
                    )
                    if predicted is not None:
                        error = float(
                            np.linalg.norm(
                                predicted - target_image[3][target[1]]
                            )
                        )
                        if error <= args.max_error:
                            previous = completion_candidates.get(target)
                            if previous is None or error < previous[0]:
                                completion_candidates[target] = (error, owner)
            if first is None or second is None or first == second:
                continue
            support[tuple(sorted((first, second)))] += 1

    candidates = []
    overlap_pairs = 0
    for (first_id, second_id), edge_support in support.items():
        first_xyz, first_track = points[first_id]
        second_xyz, second_track = points[second_id]
        overlap = len(
            {image_id for image_id, _ in first_track}.intersection(
                image_id for image_id, _ in second_track
            )
        )
        if overlap:
            overlap_pairs += 1
        first_weight = len(first_track)
        second_weight = len(second_track)
        weighted = (
            first_weight * first_xyz + second_weight * second_xyz
        ) / (first_weight + second_weight)
        scorer = score_unique_image_position if overlap else score_position
        scores = [
            scorer(cameras, images, (first_track, second_track), xyz)
            for xyz in (weighted, first_xyz, second_xyz)
        ]
        scores = [score for score in scores if score is not None]
        if not scores:
            continue
        score = min(scores)
        max_error, rms_error = score[:2]
        if max_error <= args.max_error:
            if overlap:
                retained_observations = score[2]
                lost_observations = score[3]
                retained_first = score[4]
                retained_second = score[5]
            else:
                retained_observations = first_weight + second_weight
                lost_observations = 0
                retained_first = first_weight
                retained_second = second_weight
            candidates.append(
                (
                    lost_observations,
                    max_error,
                    rms_error,
                    -edge_support,
                    first_id,
                    second_id,
                    retained_observations,
                    retained_first,
                    retained_second,
                    overlap,
                )
            )

    candidates.sort()
    claimed = set()
    greedy = []
    for candidate in candidates:
        first_id, second_id = candidate[4], candidate[5]
        if first_id in claimed or second_id in claimed:
            continue
        claimed.update((first_id, second_id))
        greedy.append(candidate)

    num_points = len(points)
    num_observations = sum(len(track) for _, track in points.values())
    result = {
        "points": num_points,
        "observations": num_observations,
        "mean_track": num_observations / num_points,
        "different_point_pairs": len(support),
        "completion_candidates": len(completion_candidates),
        "completion_same_image_conflicts": len(
            completion_same_image_conflicts
        ),
        "completion_conflicts_with_geometry": len(
            completion_conflict_candidates
        ),
        "completion_conflicts_target_better": sum(
            1
            for candidates_by_owner in completion_conflict_candidates.values()
            if any(target_error < existing_error
                   and target_error <= args.max_error
                   for target_error, existing_error
                   in candidates_by_owner.values())
        ),
        "completion_conflicts_target_better_by_0_25px": sum(
            1
            for candidates_by_owner in completion_conflict_candidates.values()
            if any(target_error + 0.25 < existing_error
                   and target_error <= args.max_error
                   for target_error, existing_error
                   in candidates_by_owner.values())
        ),
        "completion_conflicts_link_multiple_points": sum(
            1 for candidates_by_owner in completion_conflict_candidates.values()
            if len(candidates_by_owner) >= 2
        ),
        "completion_estimated_observations": (
            num_observations + len(completion_candidates)
        ),
        "completion_estimated_mean_track": (
            (num_observations + len(completion_candidates)) / num_points
        ),
        "same_image_overlap_pairs": overlap_pairs,
        "disjoint_view_pairs": len(support) - overlap_pairs,
        "geometry_accepted_disjoint_pairs": sum(
            1 for item in candidates if item[9] == 0
        ),
        "geometry_accepted_overlap_pairs": sum(
            1 for item in candidates if item[9] > 0
        ),
        "greedy_non_overlapping_merges": len(greedy),
        "greedy_preserved_observations": sum(item[6] for item in greedy),
        "greedy_lost_observations": sum(item[0] for item in greedy),
        "greedy_estimated_observations": (
            num_observations - sum(item[0] for item in greedy)
        ),
        "greedy_estimated_points": num_points - len(greedy),
        "greedy_estimated_mean_track": (
            (num_observations - sum(item[0] for item in greedy))
            / (num_points - len(greedy))
            if len(greedy) < num_points
            else None
        ),
        "greedy_merges_losing_zero": sum(
            1 for item in greedy if item[0] == 0
        ),
        "greedy_merges_losing_one": sum(
            1 for item in greedy if item[0] == 1
        ),
        "greedy_merges_losing_two_or_more": sum(
            1 for item in greedy if item[0] >= 2
        ),
        "greedy_retains_both_source_tracks": sum(
            1 for item in greedy if item[7] > 0 and item[8] > 0
        ),
        "accepted_max_error_p50": (
            float(np.median([item[1] for item in candidates]))
            if candidates
            else None
        ),
        "accepted_direct_support_ge_2": sum(
            1 for item in candidates if -item[3] >= 2
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
