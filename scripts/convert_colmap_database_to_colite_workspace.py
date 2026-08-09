#!/usr/bin/env python3
"""Convert a COLMAP database into a mapper-only Colite workspace."""

import argparse
import sqlite3
import struct
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import read_model


MAX_IMAGE_ID = 2_147_483_647
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
    11: "RAD_TAN_THIN_PRISM_FISHEYE",
}


def decode_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def read_array(blob, dtype, rows, cols):
    if blob is None or rows == 0:
        return np.empty((0, cols), dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape((rows, cols))


def geometry_type(config):
    if config in (2, 3, 4, 5):
        return config
    if config == 6:
        return 4
    if config == 8:
        return 6
    return None


def rotation_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def relative_pose(images, image_id1, image_id2):
    rotation1, translation1, _, _ = images[image_id1]
    rotation2, translation2, _, _ = images[image_id2]
    rotation21 = rotation2 @ rotation1.T
    translation21 = translation2 - rotation21 @ translation1
    translation_norm = np.linalg.norm(translation21)
    if translation_norm > 0.0:
        translation21 /= translation_norm
    return rotation_to_quaternion(rotation21), translation21


def pair_median_angle(pose_images, pose_points, owners, image_id1,
                      image_id2, matches):
    rotation1, translation1, _, _ = pose_images[image_id1]
    rotation2, translation2, _, _ = pose_images[image_id2]
    center1 = -rotation1.T @ translation1
    center2 = -rotation2.T @ translation2
    angles = []
    for idx1, idx2 in matches:
        point_id1 = owners.get((image_id1, int(idx1)))
        point_id2 = owners.get((image_id2, int(idx2)))
        if point_id1 is None or point_id1 != point_id2:
            continue
        xyz = pose_points[point_id1][0]
        ray1 = xyz - center1
        ray2 = xyz - center2
        denominator = np.linalg.norm(ray1) * np.linalg.norm(ray2)
        if denominator <= 0.0:
            continue
        cosine = np.clip(float(ray1 @ ray2 / denominator), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return float(np.median(angles)) if angles else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--pose_model", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--include_descriptors",
        action="store_true",
        help="Convert COLMAP uint8 descriptors for matcher-level A/B tests.",
    )
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    features_path = args.workspace / "features"
    matches_path = args.workspace / "matches"
    features_path.mkdir(exist_ok=True)
    matches_path.mkdir(exist_ok=True)

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    pose_images = None
    pose_points = None
    pose_owners = None
    if args.pose_model:
        _, pose_images, pose_points = read_model(args.pose_model)
        pose_owners = {
            observation: point_id
            for point_id, (_, track) in pose_points.items()
            for observation in track
        }
    cameras = list(connection.execute(
        "SELECT camera_id, model, width, height, params, prior_focal_length "
        "FROM cameras ORDER BY camera_id"
    ))
    images = list(connection.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    ))

    with (args.workspace / "cameras.txt").open("w", encoding="utf-8") as stream:
        stream.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] PRIOR_FOCAL\n")
        for camera_id, model, width, height, params_blob, prior in cameras:
            if model not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model}")
            params = np.frombuffer(params_blob, dtype="<f8")
            values = " ".join(format(float(value), ".17g") for value in params)
            stream.write(
                f"{camera_id} {CAMERA_MODELS[model]} {width} {height} "
                f"{values} PRIOR_FOCAL={1 if prior else 0}\n"
            )

    with (args.workspace / "images.tsv").open("w", encoding="utf-8") as stream:
        stream.write("# image_id\tcamera_id\tname\tfile_path\n")
        for image_id, name, camera_id in images:
            stream.write(
                f"{image_id}\t{camera_id}\t{name}\t"
                f"{(args.image_path / name).as_posix()}\n"
            )

    keypoints_by_image = {
        image_id: (rows, cols, blob)
        for image_id, rows, cols, blob in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        )
    }
    descriptors_by_image = {}
    if args.include_descriptors:
        descriptors_by_image = {
            image_id: (rows, cols, blob)
            for image_id, rows, cols, blob in connection.execute(
                "SELECT image_id, rows, cols, data FROM descriptors"
            )
        }
    for image_id, _, _ in images:
        rows, cols, blob = keypoints_by_image[image_id]
        source = read_array(blob, "<f4", rows, cols)
        keypoints = np.zeros((rows, 5), dtype="<f4")
        keypoints[:, :2] = source[:, :2]
        if cols == 4:
            keypoints[:, 2:4] = source[:, 2:4]
        elif cols == 6:
            # Affine keypoints store x,y,a11,a12,a21,a22. Use the geometric
            # mean scale; mapper geometry only consumes x/y.
            determinants = source[:, 2] * source[:, 5] - source[:, 3] * source[:, 4]
            keypoints[:, 2] = np.sqrt(np.maximum(np.abs(determinants), 1e-12))
        else:
            keypoints[:, 2] = 1.0
        descriptors = np.empty((0, 0), dtype="<f4")
        if args.include_descriptors:
            descriptor_rows, descriptor_cols, descriptor_blob = (
                descriptors_by_image[image_id]
            )
            if descriptor_rows != rows:
                raise ValueError(
                    f"Keypoint/descriptor count mismatch for image {image_id}: "
                    f"{rows} != {descriptor_rows}"
                )
            descriptors = (
                read_array(
                    descriptor_blob, "u1", descriptor_rows, descriptor_cols
                ).astype("<f4")
                / 512.0
            )
        with (features_path / f"{image_id}.bin").open("wb") as stream:
            stream.write(
                struct.pack(
                    "<IIQQQ",
                    0x434C4654,
                    1,
                    rows,
                    descriptors.shape[0],
                    descriptors.shape[1],
                )
            )
            stream.write(keypoints.tobytes(order="C"))
            stream.write(descriptors.tobytes(order="C"))

    with (matches_path / "verified_matches.tsv").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write(
            "# image_id1\timage_id2\ttype\tmedian_angle\tqw\tqx\tqy\tqz"
            "\ttx\tty\ttz\tidx1\tidx2\n"
        )
        query = (
            "SELECT pair_id, rows, cols, data, config, qvec, tvec "
            "FROM two_view_geometries ORDER BY pair_id"
        )
        for pair_id, rows, cols, blob, config, qvec_blob, tvec_blob in connection.execute(query):
            mapped_type = geometry_type(config)
            if mapped_type is None or rows == 0 or blob is None:
                continue
            image_id1, image_id2 = decode_pair_id(pair_id)
            matches = read_array(blob, "<u4", rows, cols)[:, :2]
            qvec = (
                np.frombuffer(qvec_blob, dtype="<f8")
                if qvec_blob is not None else np.asarray([1.0, 0.0, 0.0, 0.0])
            )
            tvec = (
                np.frombuffer(tvec_blob, dtype="<f8")
                if tvec_blob is not None else np.asarray([0.0, 0.0, 0.0])
            )
            if pose_images is not None:
                qvec, tvec = relative_pose(
                    pose_images, image_id1, image_id2
                )
                median_angle = pair_median_angle(
                    pose_images, pose_points, pose_owners,
                    image_id1, image_id2, matches
                )
            else:
                median_angle = -1.0
            pose = "\t".join(
                format(float(value), ".17g") for value in np.r_[qvec, tvec]
            )
            for idx1, idx2 in matches:
                stream.write(
                    f"{image_id1}\t{image_id2}\t{mapped_type}\t"
                    f"{median_angle:.17g}\t{pose}\t"
                    f"{int(idx1)}\t{int(idx2)}\n"
                )
    connection.close()


if __name__ == "__main__":
    main()
