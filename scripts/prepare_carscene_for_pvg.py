#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


IMAGE_NAME_RE = re.compile(r"^(?P<frame>\d+)_(?P<camera>\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reshape a 3-camera CarTwin scene into the directory layout expected by PVG."
    )
    parser.add_argument("--src-scene", type=Path, required=True, help="Source CarTwin scene directory.")
    parser.add_argument(
        "--storm-annotation",
        type=Path,
        default=None,
        help="Optional STORM annotation JSON. If omitted, the script tries known default locations.",
    )
    parser.add_argument(
        "--dst-scene",
        type=Path,
        default=None,
        help="Destination PVG scene directory. Defaults to --src-scene (in-place).",
    )
    parser.add_argument("--num-cams", type=int, default=3, choices=[3], help="Number of cameras to expose to PVG.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files if they already exist.")
    return parser.parse_args()


def discover_annotation(scene_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Annotation JSON not found: {explicit_path}")
        return explicit_path

    scene_name = scene_root.name
    longtail_root = scene_root.parent.parent
    candidates = [
        longtail_root
        / "GSTORM"
        / f"STORM_custom_{scene_name}"
        / "annotations"
        / "waymo"
        / "training"
        / f"{scene_name}.json",
        longtail_root
        / "GaussianSTORM"
        / "data"
        / f"STORM_custom_{scene_name}"
        / "annotations"
        / "waymo"
        / "training"
        / f"{scene_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not auto-discover STORM annotation for {scene_name}. "
        "Pass --storm-annotation explicitly."
    )


def load_annotation(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_frames(scene_root: Path, num_cams: int) -> list[str]:
    image_dir = scene_root / "images"
    mask_dir = scene_root / "sky_masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing sky_masks directory: {mask_dir}")

    frames: dict[str, set[int]] = {}
    mask_frames: dict[str, set[int]] = {}

    for path in sorted(image_dir.glob("*.png")):
        match = IMAGE_NAME_RE.match(path.name)
        if match is None:
            continue
        frames.setdefault(match.group("frame"), set()).add(int(match.group("camera")))

    for path in sorted(mask_dir.glob("*.png")):
        match = IMAGE_NAME_RE.match(path.name)
        if match is None:
            continue
        mask_frames.setdefault(match.group("frame"), set()).add(int(match.group("camera")))

    ordered_frames = sorted(frames.keys(), key=lambda value: int(value))
    if not ordered_frames:
        raise RuntimeError(f"No frame images found under {image_dir}")

    expected_cams = set(range(num_cams))
    for frame in ordered_frames:
        if frames.get(frame) != expected_cams:
            raise ValueError(f"Frame {frame} is missing image cameras: expected {expected_cams}, got {frames.get(frame)}")
        if mask_frames.get(frame) != expected_cams:
            raise ValueError(
                f"Frame {frame} is missing sky-mask cameras: expected {expected_cams}, got {mask_frames.get(frame)}"
            )
    return ordered_frames


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def format_matrix_rows(matrix: list[list[float]], rows: int = 3) -> str:
    values = []
    for row_idx in range(rows):
        for col_idx in range(4):
            values.append(f"{float(matrix[row_idx][col_idx]):.6e}")
    return " ".join(values)


def build_projection_line(fx: float, fy: float, cx: float, cy: float) -> str:
    projection = [
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return format_matrix_rows(projection)


def invert_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation_t = [[float(matrix[col_idx][row_idx]) for col_idx in range(3)] for row_idx in range(3)]
    translation = [float(matrix[row_idx][3]) for row_idx in range(3)]
    inv_translation = [
        -sum(rotation_t[row_idx][col_idx] * translation[col_idx] for col_idx in range(3)) for row_idx in range(3)
    ]
    return [
        rotation_t[0] + [inv_translation[0]],
        rotation_t[1] + [inv_translation[1]],
        rotation_t[2] + [inv_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_pose(dst_scene: Path, frame_name: str, ego_to_world: list[list[float]], overwrite: bool) -> None:
    dst = dst_scene / "pose" / f"{frame_name}.txt"
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as handle:
        for row in ego_to_world:
            handle.write(" ".join(f"{float(value):.18e}" for value in row) + "\n")


def write_calib(
    dst_scene: Path,
    frame_name: str,
    annotation: dict,
    width: float,
    height: float,
    num_cams: int,
    overwrite: bool,
) -> None:
    dst = dst_scene / "calib" / f"{frame_name}.txt"
    if dst.exists() and not overwrite:
        return

    projection_lines = []
    lidar_to_cam_lines = []
    for camera_id in range(5):
        src_camera_id = min(camera_id, num_cams - 1)
        intr = annotation["normalized_intrinsics"][str(src_camera_id)]
        fx = float(intr[0]) * width
        fy = float(intr[1]) * height
        cx = float(intr[2]) * width
        cy = float(intr[3]) * height
        projection_lines.append(f"P{camera_id}: {build_projection_line(fx, fy, cx, cy)}")

        camera_to_ego = annotation["camera_to_ego"][str(src_camera_id)]
        lidar_to_cam = invert_rigid(camera_to_ego)
        lidar_to_cam_lines.append(f"Tr_velo_to_cam_{camera_id}: {format_matrix_rows(lidar_to_cam)}")

    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as handle:
        for line in projection_lines:
            handle.write(line + "\n")
        handle.write("R0_rect: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00\n")
        for line in lidar_to_cam_lines:
            handle.write(line + "\n")


def touch_empty_velodyne(dst_scene: Path, frame_name: str, overwrite: bool) -> None:
    dst = dst_scene / "velodyne" / f"{frame_name}.bin"
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    dst.write_bytes(b"")


def main() -> int:
    args = parse_args()
    src_scene = args.src_scene.resolve()
    dst_scene = (args.dst_scene or args.src_scene).resolve()
    annotation_path = discover_annotation(src_scene, args.storm_annotation)
    annotation = load_annotation(annotation_path)
    frame_names = discover_frames(src_scene, args.num_cams)

    expected_frames = int(annotation["num_timesteps"])
    if expected_frames != len(frame_names):
        raise ValueError(
            f"Frame count mismatch: annotation has {expected_frames}, scene images have {len(frame_names)}"
        )

    image_size = annotation.get("original_image_size", {}).get("0")
    if image_size is None or len(image_size) != 2:
        raise ValueError("Annotation is missing original_image_size['0']")
    height = float(image_size[0])
    width = float(image_size[1])

    for camera_id in range(args.num_cams):
        ensure_dir(dst_scene / f"image_{camera_id}")
        ensure_dir(dst_scene / f"sky_{camera_id}")
    ensure_dir(dst_scene / "pose")
    ensure_dir(dst_scene / "calib")
    ensure_dir(dst_scene / "velodyne")

    for frame_idx, frame_name in enumerate(frame_names):
        for camera_id in range(args.num_cams):
            copy_file(
                src_scene / "images" / f"{frame_name}_{camera_id}.png",
                dst_scene / f"image_{camera_id}" / f"{frame_name}.png",
                overwrite=args.overwrite,
            )
            copy_file(
                src_scene / "sky_masks" / f"{frame_name}_{camera_id}.png",
                dst_scene / f"sky_{camera_id}" / f"{frame_name}.png",
                overwrite=args.overwrite,
            )
        write_pose(dst_scene, frame_name, annotation["ego_to_world"][frame_idx], args.overwrite)
        write_calib(dst_scene, frame_name, annotation, width, height, args.num_cams, args.overwrite)
        touch_empty_velodyne(dst_scene, frame_name, args.overwrite)

    meta = {
        "source_scene": str(src_scene),
        "storm_annotation": str(annotation_path),
        "frames": len(frame_names),
        "num_cams": args.num_cams,
        "notes": [
            "Generated PVG directories image_{0..2}, sky_{0..2}, pose, calib, velodyne.",
            "Velodyne files are empty placeholders; PVG loader fallback is required for no-LiDAR scenes.",
            "Camera-to-world is reconstructed from STORM ego_to_world and camera_to_ego metadata.",
        ],
    }
    with open(dst_scene / "pvg_prepare_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Prepared PVG scene at {dst_scene}")
    print(f"Annotation: {annotation_path}")
    print(f"Frames: {len(frame_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
