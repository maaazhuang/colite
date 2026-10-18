#!/usr/bin/env python3
"""Import COLMAP keypoints and optional typed descriptors into Colite.

This is a one-way benchmark bridge, not a lossless database converter: affine
shape, raw matches, F/E/H, rig/frame assignments and pose priors are not retained.
"""

import argparse
import sqlite3
import struct
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import quaternion_rotation, read_model


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
    if rows < 0 or cols < 0:
        raise ValueError("Negative database matrix dimensions")
    expected_bytes = rows * cols * np.dtype(dtype).itemsize
    if len(blob or b"") != expected_bytes:
        raise ValueError(f"Invalid database blob length: expected {expected_bytes}")
    if rows == 0:
        return np.empty((0, cols), dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape((rows, cols))


def convert_keypoints(source):
    """Keep COLMAP pixel coordinates and row order; convert radians to degrees."""
    rows, cols = source.shape
    if cols not in (2, 4, 6) or not np.isfinite(source).all():
        raise ValueError("Expected finite COLMAP keypoints with 2, 4 or 6 columns")
    keypoints = np.zeros((rows, 5), dtype="<f4")
    keypoints[:, :2] = source[:, :2]
    keypoints[:, 2] = 1.0
    if cols == 4:
        keypoints[:, 2] = source[:, 2]
        keypoints[:, 3] = np.degrees(source[:, 3]) % 360.0
    elif cols == 6:
        # Match COLMAP ComputeScale/ComputeOrientation. Full shear/anisotropy
        # cannot be expressed in the current Colite FeatureKeypoint.
        keypoints[:, 2] = (np.hypot(source[:, 2], source[:, 4]) +
                           np.hypot(source[:, 3], source[:, 5])) / 2.0
        keypoints[:, 3] = np.degrees(np.arctan2(source[:, 4], source[:, 2])) % 360.0
    return keypoints


def decode_descriptors(feature_type, rows, byte_cols, blob):
    """COLMAP 4.0 cols counts bytes, including for ALIKED float32 data."""
    if feature_type == 0:  # SIFT
        if byte_cols != 128:
            raise ValueError("SIFT descriptors must have 128 bytes per row")
        descriptors = read_array(blob, "u1", rows, byte_cols).astype("<f4") / 512.0
    elif feature_type in (1, 2):  # ALIKED_N16ROT / ALIKED_N32
        if byte_cols != 512:
            raise ValueError("ALIKED descriptors must have 512 bytes per row")
        descriptors = read_array(blob, "<f4", rows, byte_cols // 4)
    else:
        raise ValueError(f"Unsupported COLMAP descriptor type: {feature_type}")
    if not np.isfinite(descriptors).all():
        raise ValueError("Non-finite COLMAP descriptors")
    return descriptors


def write_feature_v2(path, keypoints, descriptors, feature_type,
                     sift_normalization="L1_ROOT"):
    chunks = [keypoints.tobytes(order="C"), descriptors.tobytes(order="C")]
    checksum = 14695981039346656037
    for chunk in chunks:
        for byte in chunk:
            checksum = ((checksum ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    # SQLite stores the extractor type, but not the SIFT normalization choice.
    # Default to COLMAP's RootSIFT; callers can declare L2 without re-normalizing.
    metric, normalization = ((1, 2 if sift_normalization == "L1_ROOT" else 3)
                             if feature_type == 0 else (2, 3))
    with path.open("wb") as stream:
        stream.write(struct.pack("<IIIIIIQQQQQ", 0x434C4654, 2,
                                 feature_type + 1, 1, metric, normalization,
                                 len(keypoints), *descriptors.shape,
                                 len(chunks[1]), checksum))
        for chunk in chunks:
            stream.write(chunk)


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


def camera_rays(model, params, xy):
    """Unproject the camera models used by the controlled SfM benchmark."""
    if model in (0, 2, 3):
        norm = (xy - params[1:3]) / params[0]
    elif model == 1:
        norm = (xy - params[2:4]) / params[:2]
    else:
        raise ValueError("--compute_pair_angles supports PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL")
    if model in (2, 3):
        distorted_radius = np.linalg.norm(norm, axis=1)
        radius = distorted_radius.copy()
        k1 = params[3]
        k2 = params[4] if model == 3 else 0.0
        for _ in range(30):
            r2 = radius * radius
            derivative = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
            radius -= (radius * (1 + k1 * r2 + k2 * r2 * r2) - distorted_radius) / derivative
        norm *= np.divide(radius, distorted_radius, out=np.ones_like(radius),
                          where=distorted_radius > 0)[:, None]
    rays = np.column_stack((norm, np.ones(len(norm))))
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def relative_pair_median_angle(cameras, image_cameras, keypoints,
                               image_id1, image_id2, matches, qvec, tvec):
    if np.linalg.norm(tvec) <= 1e-12:
        raise ValueError("Relative pose missing: run COLMAP matching with TwoViewGeometry.compute_relative_pose=1")
    rays = []
    for image_id, indices in [(image_id1, matches[:, 0]), (image_id2, matches[:, 1])]:
        model, params = cameras[image_cameras[image_id]]
        rows, cols, blob = keypoints[image_id]
        xy = read_array(blob, "<f4", rows, cols)[indices, :2].astype(np.float64)
        rays.append(camera_rays(model, params, xy))
    rotation = quaternion_rotation(qvec)
    cosines = np.sum(rays[0] * (rays[1] @ rotation), axis=1)
    return float(np.median(np.degrees(np.arccos(np.clip(cosines, -1, 1)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--pose_model", type=Path)
    parser.add_argument("--compute_pair_angles", action="store_true",
                        help="Derive pair parallax from database relative poses, without final SfM poses.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--SiftExtraction.normalization", dest="sift_normalization",
                        choices=("L1_ROOT", "L2"), default="L1_ROOT",
                        help="Source SIFT normalization, absent from SQLite; default is COLMAP's L1_ROOT.")
    parser.add_argument(
        "--include_descriptors",
        action="store_true",
        help="Import SIFT uint8 or ALIKED float32 descriptors with type metadata.",
    )
    args = parser.parse_args()

    if args.workspace.exists() and any(args.workspace.iterdir()):
        raise ValueError("Workspace must be empty; existing data is not overwritten")

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
    camera_values = {camera_id: (model, np.frombuffer(params_blob, dtype="<f8"))
                     for camera_id, model, _, _, params_blob, _ in cameras}
    image_cameras = {image_id: camera_id for image_id, _, camera_id in images}

    keypoints_by_image = {
        image_id: (rows, cols, blob)
        for image_id, rows, cols, blob in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints")
    }
    descriptor_columns = {row[1] for row in connection.execute("PRAGMA table_info(descriptors)")}
    type_column = "type" if "type" in descriptor_columns else "0"
    data_column = "data" if args.include_descriptors else "NULL"
    descriptors_by_image = {
        image_id: (feature_type, rows, cols, blob)
        for image_id, feature_type, rows, cols, blob in connection.execute(
            f"SELECT image_id, {type_column}, rows, cols, {data_column} FROM descriptors")
    }
    # Reject invalid types/lengths before creating output files.
    for image_id, _, _ in images:
        if image_id not in keypoints_by_image:
            raise ValueError(f"Missing keypoints for image {image_id}")
        rows, cols, blob = keypoints_by_image[image_id]
        convert_keypoints(read_array(blob, "<f4", rows, cols))
        if image_id not in descriptors_by_image:
            if args.include_descriptors:
                raise ValueError(f"Missing descriptors for image {image_id}")
            continue
        feature_type, descriptor_rows, byte_cols, descriptor_blob = descriptors_by_image[image_id]
        if feature_type not in (0, 1, 2):
            raise ValueError(f"Unsupported COLMAP descriptor type: {feature_type}")
        if args.include_descriptors:
            if descriptor_rows != rows:
                raise ValueError(f"Keypoint/descriptor count mismatch for image {image_id}")
            decode_descriptors(feature_type, descriptor_rows, byte_cols, descriptor_blob)

    args.workspace.mkdir(parents=True, exist_ok=True)
    features_path = args.workspace / "features"
    matches_path = args.workspace / "matches"
    features_path.mkdir(exist_ok=True)
    matches_path.mkdir(exist_ok=True)

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

    for image_id, _, _ in images:
        rows, cols, blob = keypoints_by_image[image_id]
        source = read_array(blob, "<f4", rows, cols)
        keypoints = convert_keypoints(source)
        feature_type = descriptors_by_image.get(image_id, (0,))[0]
        descriptors = np.empty((0, 0), dtype="<f4")
        if args.include_descriptors:
            feature_type, descriptor_rows, descriptor_cols, descriptor_blob = (
                descriptors_by_image[image_id]
            )
            descriptors = decode_descriptors(feature_type, descriptor_rows,
                                             descriptor_cols, descriptor_blob)
        write_feature_v2(features_path / f"{image_id}.bin", keypoints,
                         descriptors, feature_type, args.sift_normalization)

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
                if args.compute_pair_angles and mapped_type == 2:
                    median_angle = relative_pair_median_angle(
                        camera_values, image_cameras, keypoints_by_image,
                        image_id1, image_id2, matches, qvec, tvec)
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
