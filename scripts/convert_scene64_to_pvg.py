#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


IMAGE_NAME_RE = re.compile(r"^(?P<frame>\d+)_(?P<camera>[0-2])\.png$", re.IGNORECASE)

CAMERA_ID_TO_NAME: Dict[int, str] = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
}
CAMERA_FRONT_NAME = CAMERA_ID_TO_NAME[0]
LIDAR_TOP_NAME = "lidar_top_360fov"

DEFAULT_SRC_ROOT = Path(
    "/ssd2/wenyan/CarTwin/longtail/data_64/scene64_complete/scene64_complete"
)
DEFAULT_DST_ROOT = Path(
    "/ssd2/wenyan/CarTwin/longtail/PVG/data/scene64_complete_pvg"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRACO_SOURCE_CANDIDATES = [
    REPO_ROOT / "third_party" / "draco",
    Path("/ssd2/jhp/ProbeSDF/libs/assimp/contrib/draco"),
]
DEFAULT_DRACO_SOURCE_GLOBS = [
    "/ssd2/*/ProbeSDF/libs/assimp/contrib/draco",
]
DEFAULT_CMAKE_CANDIDATES = [
    shutil.which("cmake"),
    "/ssd2/ziyi/miniconda3/envs/StreamVGGT/bin/cmake",
    "/ssd2/jhp/miniforge3/envs/eva/bin/cmake",
]


@dataclass(frozen=True)
class SceneSpec:
    category: str
    uuid: str
    src_scene: Path
    dst_scene: Path


@dataclass(frozen=True)
class FrameSelection:
    frame_indices: List[int]
    ego_to_world: List[List[List[float]]]
    trim_start_frames: int
    trim_end_frames: int
    source_frame_count: int
    output_frame_count: int
    left_max_delta_s: float
    right_max_delta_s: float
    ego_max_delta_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert rectified scene64_complete scenes into the PVG input layout."
    )
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional single-scene selector in the form <category>/<uuid>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--draco-decoder", type=Path, default=None)
    parser.add_argument("--draco-source", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/pvg_draco_build"))
    parser.add_argument("--max-lidar-delta", type=float, default=None)
    return parser.parse_args()


def discover_scenes(src_root: Path, dst_root: Path, selected_scene: str | None) -> List[SceneSpec]:
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    selected_category = None
    selected_uuid = None
    if selected_scene is not None:
        parts = selected_scene.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("--scene must have the form <category>/<uuid>")
        selected_category, selected_uuid = parts

    scenes: List[SceneSpec] = []
    for category_dir in sorted(src_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        if selected_category is not None and category_dir.name != selected_category:
            continue
        for scene_dir in sorted(category_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
                continue
            if selected_uuid is not None and scene_dir.name != selected_uuid:
                continue
            scenes.append(
                SceneSpec(
                    category=category_dir.name,
                    uuid=scene_dir.name,
                    src_scene=scene_dir,
                    dst_scene=dst_root / category_dir.name / scene_dir.name,
                )
            )

    if selected_scene is not None and not scenes:
        raise FileNotFoundError(f"Requested scene was not found: {selected_scene}")
    if not scenes:
        raise RuntimeError(f"No scenes found under {src_root}")
    return scenes


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(paths: Iterable[str | None]) -> str | None:
    for candidate in paths:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def iter_draco_source_candidates() -> Iterable[Path]:
    env_value = os.environ.get("DRACO_SOURCE")
    if env_value:
        yield Path(env_value).expanduser()

    yield from DEFAULT_DRACO_SOURCE_CANDIDATES

    for pattern in DEFAULT_DRACO_SOURCE_GLOBS:
        for candidate in sorted(Path("/").glob(pattern.lstrip("/"))):
            yield candidate


def resolve_draco_source(user_value: Path | None) -> Path:
    if user_value is not None:
        if not user_value.is_dir():
            raise FileNotFoundError(f"Draco source directory does not exist: {user_value}")
        return user_value

    checked: List[str] = []
    for candidate in iter_draco_source_candidates():
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find a Draco source tree. Pass --draco-source explicitly or set "
        f"DRACO_SOURCE. Checked: {checked or ['<none>']}"
    )


def build_draco_decoder(draco_source: Path, build_dir: Path) -> Path:
    cmake_bin = find_first_existing(DEFAULT_CMAKE_CANDIDATES)
    if cmake_bin is None:
        raise FileNotFoundError(
            "Could not find cmake. Install it or pass an existing --draco-decoder."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir = build_dir.parent / "pvg_draco_install"
    install_dir.mkdir(parents=True, exist_ok=True)

    decoder_path = build_dir / "draco_decoder"
    if decoder_path.is_file():
        return decoder_path

    configure_cmd = [
        cmake_bin,
        "-S",
        str(draco_source),
        "-B",
        str(build_dir),
        f"-DAssimp_BINARY_DIR={build_dir}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DDRACO_TESTS=OFF",
        "-DDRACO_JS_GLUE=OFF",
        "-DDRACO_WASM=OFF",
    ]
    build_cmd = [
        cmake_bin,
        "--build",
        str(build_dir),
        "--target",
        "draco_decoder",
        "-j",
        str(min(os.cpu_count() or 4, 8)),
    ]

    subprocess.run(configure_cmd, check=True)
    subprocess.run(build_cmd, check=True)

    if not decoder_path.is_file():
        raise FileNotFoundError(f"Build succeeded but decoder is missing: {decoder_path}")
    return decoder_path


def resolve_draco_decoder(args: argparse.Namespace) -> Path:
    if args.draco_decoder is not None:
        if not args.draco_decoder.is_file():
            raise FileNotFoundError(f"draco_decoder does not exist: {args.draco_decoder}")
        return args.draco_decoder

    repo_decoder = REPO_ROOT / "third_party" / "draco_build" / "draco_decoder"
    build_dir_decoder = args.build_dir / "draco_decoder"
    discovered = find_first_existing(
        [
            os.environ.get("DRACO_DECODER"),
            shutil.which("draco_decoder"),
            "/tmp/draco-build/draco_decoder",
            "/tmp/pvg_draco_build/draco_decoder",
            str(repo_decoder),
            str(build_dir_decoder),
        ]
    )
    if discovered is not None:
        return Path(discovered)

    draco_source = resolve_draco_source(args.draco_source)
    return build_draco_decoder(draco_source, args.build_dir)


def collect_frame_coverage(scene_root: Path) -> Tuple[List[str], Dict[str, set[int]], Dict[str, set[int]]]:
    image_dir = scene_root / "images"
    mask_dir = scene_root / "sky_masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing sky_masks directory: {mask_dir}")

    frames: Dict[str, set[int]] = {}
    mask_frames: Dict[str, set[int]] = {}

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

    expected_frame_names = [f"{frame_idx:04d}" for frame_idx in range(len(ordered_frames))]
    if ordered_frames != expected_frame_names:
        raise ValueError(
            f"Frame ids are not contiguous from 0000: first={ordered_frames[:3]}, last={ordered_frames[-3:]}"
        )
    return ordered_frames, frames, mask_frames


def validate_required_paths(scene_root: Path) -> None:
    required_paths = [
        scene_root / "images",
        scene_root / "sky_masks",
        scene_root / "intrinsics" / "0.txt",
        scene_root / "intrinsics" / "1.txt",
        scene_root / "intrinsics" / "2.txt",
        scene_root / "calibration" / "sensor_extrinsics.parquet",
        scene_root / "labels" / "egomotion" / f"{scene_root.name}.egomotion.parquet",
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[0]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[0]}.timestamps.parquet",
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[1]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[1]}.timestamps.parquet",
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[2]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[2]}.timestamps.parquet",
        scene_root / "lidar" / LIDAR_TOP_NAME / f"{scene_root.name}.{LIDAR_TOP_NAME}.parquet",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")


def copy_scene_tree(src_scene: Path, dst_scene: Path, overwrite: bool) -> None:
    if dst_scene.exists():
        if not overwrite:
            raise FileExistsError(f"Destination scene already exists: {dst_scene}")
        shutil.rmtree(dst_scene)
    ensure_dir(dst_scene.parent)
    shutil.copytree(src_scene, dst_scene)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def lower_column_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(column).lower(): str(column) for column in columns}


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    name_candidates: Sequence[str],
    clip_name_candidates: Sequence[str] = ("clip_id",),
) -> pd.DataFrame:
    if any(candidate in df.columns for candidate in name_candidates):
        return df.reset_index(drop=False) if df.index.names != [None] else df

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None or df.index.names != [None]:
        df = df.reset_index()
    elif not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    for candidate in name_candidates:
        if candidate in df.columns:
            break
        level_key = f"level_{1 if len(name_candidates) > 1 else 0}"
        if candidate not in df.columns and level_key in df.columns:
            df = df.rename(columns={level_key: candidate})
            break
        if candidate not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": candidate})
            break
        for column in list(df.columns):
            if str(column).lower() == candidate.lower():
                df = df.rename(columns={column: candidate})
                break

    for candidate in clip_name_candidates:
        if candidate in df.columns:
            continue
        for column in list(df.columns):
            if str(column).lower() == candidate.lower():
                df = df.rename(columns={column: candidate})
                break

    return df


def identity4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rigid_inverse4(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    rotation_t = [
        [float(matrix[col_idx][row_idx]) for col_idx in range(3)]
        for row_idx in range(3)
    ]
    translation = [float(matrix[row_idx][3]) for row_idx in range(3)]
    inv_translation = [
        -sum(rotation_t[row_idx][col_idx] * translation[col_idx] for col_idx in range(3))
        for row_idx in range(3)
    ]
    result = identity4()
    for row_idx in range(3):
        for col_idx in range(3):
            result[row_idx][col_idx] = rotation_t[row_idx][col_idx]
        result[row_idx][3] = inv_translation[row_idx]
    return result


def quaternion_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("Encountered zero-norm quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
    ]


def build_transform(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> List[List[float]]:
    matrix = identity4()
    for row_idx in range(3):
        for col_idx in range(3):
            matrix[row_idx][col_idx] = float(rotation[row_idx][col_idx])
        matrix[row_idx][3] = float(translation[row_idx])
    return matrix


def parse_transform_from_row(row: pd.Series) -> List[List[float]]:
    column_map = lower_column_map(row.index)

    transform_cols = [column_map.get(f"transform_{idx}") for idx in range(16)]
    if all(column is not None for column in transform_cols):
        values = [float(row[column]) for column in transform_cols]
        return [values[row_idx * 4 : (row_idx + 1) * 4] for row_idx in range(4)]

    matrix_cols = [column_map.get(f"m{i}{j}") for i in range(4) for j in range(4)]
    if all(column is not None for column in matrix_cols):
        values = [float(row[column]) for column in matrix_cols]
        return [values[row_idx * 4 : (row_idx + 1) * 4] for row_idx in range(4)]

    rotation_cols = [column_map.get(f"r{i}{j}") for i in range(3) for j in range(3)]
    translation_cols = [column_map.get(name) for name in ("tx", "ty", "tz")]
    if all(column is not None for column in rotation_cols + translation_cols):
        flat_rotation = [float(row[column]) for column in rotation_cols]
        rotation = [flat_rotation[row_idx * 3 : (row_idx + 1) * 3] for row_idx in range(3)]
        translation = [float(row[column]) for column in translation_cols]
        return build_transform(rotation, translation)

    quat_candidates = ("qx", "qy", "qz", "qw")
    xyz_candidates = ("x", "y", "z")
    if all(column_map.get(name) is not None for name in quat_candidates + xyz_candidates):
        rotation = quaternion_xyzw_to_matrix(
            float(row[column_map["qx"]]),
            float(row[column_map["qy"]]),
            float(row[column_map["qz"]]),
            float(row[column_map["qw"]]),
        )
        translation = [float(row[column_map[name]]) for name in xyz_candidates]
        return build_transform(rotation, translation)

    if all(column_map.get(name) is not None for name in quat_candidates + ("tx", "ty", "tz")):
        rotation = quaternion_xyzw_to_matrix(
            float(row[column_map["qx"]]),
            float(row[column_map["qy"]]),
            float(row[column_map["qz"]]),
            float(row[column_map["qw"]]),
        )
        translation = [float(row[column_map[name]]) for name in ("tx", "ty", "tz")]
        return build_transform(rotation, translation)

    raise ValueError(f"Unsupported transform schema with columns: {list(row.index)}")


def load_timestamp_series(path: Path) -> List[float]:
    df = normalize_dataframe(pd.read_parquet(path), name_candidates=("camera_name", "sensor_name"))
    column_map = lower_column_map(df.columns)
    timestamp_column = None
    for lowered, column in column_map.items():
        if "timestamp" in lowered or lowered == "time":
            timestamp_column = column
            break
    if timestamp_column is None:
        raise ValueError(f"Could not identify timestamp column in {path}")
    timestamps = [float(value) for value in df[timestamp_column].tolist()]
    if not timestamps:
        raise ValueError(f"No timestamps found in {path}")
    return timestamps


def infer_timestamp_scale_to_seconds(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 1.0
    sorted_times = sorted(float(value) for value in timestamps)
    positive = [
        sorted_times[idx + 1] - sorted_times[idx]
        for idx in range(len(sorted_times) - 1)
        if sorted_times[idx + 1] - sorted_times[idx] > 0
    ]
    if not positive:
        return 1.0
    median_delta = float(median(positive))
    if median_delta > 1e6:
        return 1e-9
    if median_delta > 1e3:
        return 1e-6
    if median_delta > 1:
        return 1e-3
    return 1.0


def nearest_indices_and_deltas(
    master_times_s: Sequence[float],
    candidate_times_s: Sequence[float],
    *,
    label: str,
) -> Tuple[List[int], List[float]]:
    if not candidate_times_s:
        raise ValueError(f"No candidate timestamps available for {label}")
    sorted_pairs = sorted(enumerate(float(value) for value in candidate_times_s), key=lambda item: item[1])
    sorted_times = [item[1] for item in sorted_pairs]
    sorted_indices = [item[0] for item in sorted_pairs]
    matched_indices: List[int] = [0] * len(master_times_s)
    matched_deltas: List[float] = [0.0] * len(master_times_s)

    for idx, master_time in enumerate(master_times_s):
        insertion = bisect_left(sorted_times, float(master_time))
        choices: List[int] = []
        if insertion < len(sorted_times):
            choices.append(insertion)
        if insertion > 0:
            choices.append(insertion - 1)
        best_choice = min(choices, key=lambda choice: abs(sorted_times[choice] - master_time))
        best_delta = abs(sorted_times[best_choice] - master_time)
        matched_indices[idx] = sorted_indices[best_choice]
        matched_deltas[idx] = float(best_delta)
    return matched_indices, matched_deltas


def find_largest_valid_window(valid_mask: Sequence[bool]) -> Tuple[int, int]:
    best_start = -1
    best_end = -1
    current_start = -1

    for idx, is_valid in enumerate(valid_mask):
        if is_valid:
            if current_start < 0:
                current_start = idx
            if best_start < 0 or (idx - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = idx
        else:
            current_start = -1

    if best_start < 0:
        raise ValueError("No contiguous frame window satisfies the multi-sensor alignment threshold.")
    return best_start, best_end


def build_complete_frame_mask(
    ordered_frames: Sequence[str],
    image_coverage: Dict[str, set[int]],
    mask_coverage: Dict[str, set[int]],
) -> List[bool]:
    expected_cams = {0, 1, 2}
    return [
        image_coverage.get(frame_name, set()) == expected_cams
        and mask_coverage.get(frame_name, set()) == expected_cams
        for frame_name in ordered_frames
    ]


def select_frames(scene_root: Path, threshold_ms: float = 50.0) -> FrameSelection:
    ordered_frames, image_coverage, mask_coverage = collect_frame_coverage(scene_root)
    frame_count = len(ordered_frames)
    complete_frame_mask = build_complete_frame_mask(ordered_frames, image_coverage, mask_coverage)

    front_timestamp_path = (
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[0]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[0]}.timestamps.parquet"
    )
    left_timestamp_path = (
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[1]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[1]}.timestamps.parquet"
    )
    right_timestamp_path = (
        scene_root
        / "camera"
        / CAMERA_ID_TO_NAME[2]
        / f"{scene_root.name}.{CAMERA_ID_TO_NAME[2]}.timestamps.parquet"
    )

    master_times = load_timestamp_series(front_timestamp_path)
    left_times = load_timestamp_series(left_timestamp_path)
    right_times = load_timestamp_series(right_timestamp_path)

    scale_to_seconds = infer_timestamp_scale_to_seconds(master_times)
    master_times_s = [value * scale_to_seconds for value in master_times]
    left_times_s = [value * scale_to_seconds for value in left_times]
    right_times_s = [value * scale_to_seconds for value in right_times]
    threshold_s = threshold_ms / 1000.0

    if not master_times_s:
        raise ValueError(f"No front-camera timestamps found in {front_timestamp_path}")

    _, left_deltas_master = nearest_indices_and_deltas(master_times_s, left_times_s, label="left camera")
    _, right_deltas_master = nearest_indices_and_deltas(master_times_s, right_times_s, label="right camera")

    ego_path = scene_root / "labels" / "egomotion" / f"{scene_root.name}.egomotion.parquet"
    ego_df = pd.read_parquet(ego_path)
    ego_df = normalize_dataframe(ego_df, name_candidates=("camera_name", "sensor_name"))
    ego_column_map = lower_column_map(ego_df.columns)

    timestamp_column = None
    for lowered, column in ego_column_map.items():
        if "timestamp" in lowered or lowered == "time":
            timestamp_column = column
            break
    if timestamp_column is None:
        raise ValueError(f"Could not identify timestamp column in {ego_path}")

    ego_times = [float(value) for value in ego_df[timestamp_column].tolist()]
    ego_scale_to_seconds = infer_timestamp_scale_to_seconds(ego_times)
    ego_times_s = [value * ego_scale_to_seconds for value in ego_times]
    ego_match, ego_deltas_master = nearest_indices_and_deltas(
        master_times_s, ego_times_s, label="egomotion"
    )

    world_to_ego = any("world_to_ego" in lowered for lowered in ego_column_map)
    ego_to_world_master: List[List[List[float]]] = []
    for ego_idx in ego_match:
        transform = parse_transform_from_row(ego_df.iloc[int(ego_idx)])
        if world_to_ego:
            transform = rigid_inverse4(transform)
        ego_to_world_master.append(transform)

    front_available_mask = [idx < len(master_times_s) for idx in range(frame_count)]
    left_available_mask = [idx < len(left_times_s) for idx in range(frame_count)]
    right_available_mask = [idx < len(right_times_s) for idx in range(frame_count)]

    left_deltas = [
        left_deltas_master[idx] if idx < len(left_deltas_master) else float("inf")
        for idx in range(frame_count)
    ]
    right_deltas = [
        right_deltas_master[idx] if idx < len(right_deltas_master) else float("inf")
        for idx in range(frame_count)
    ]
    ego_deltas = [
        ego_deltas_master[idx] if idx < len(ego_deltas_master) else float("inf")
        for idx in range(frame_count)
    ]
    ego_to_world_all = [
        ego_to_world_master[idx] if idx < len(ego_to_world_master) else identity4()
        for idx in range(frame_count)
    ]

    valid_mask = [
        complete_frame_mask[idx]
        and front_available_mask[idx]
        and left_available_mask[idx]
        and right_available_mask[idx]
        and left_deltas[idx] <= threshold_s
        and right_deltas[idx] <= threshold_s
        and ego_deltas[idx] <= threshold_s
        for idx in range(frame_count)
    ]
    start_idx, end_idx = find_largest_valid_window(valid_mask)
    frame_indices = list(range(start_idx, end_idx + 1))
    ego_to_world = [ego_to_world_all[idx] for idx in frame_indices]

    return FrameSelection(
        frame_indices=frame_indices,
        ego_to_world=ego_to_world,
        trim_start_frames=start_idx,
        trim_end_frames=frame_count - 1 - end_idx,
        source_frame_count=frame_count,
        output_frame_count=len(frame_indices),
        left_max_delta_s=max(left_deltas),
        right_max_delta_s=max(right_deltas),
        ego_max_delta_s=max(ego_deltas),
    )


def load_camera_to_ego(scene_root: Path) -> Dict[int, List[List[float]]]:
    path = scene_root / "calibration" / "sensor_extrinsics.parquet"
    df = normalize_dataframe(pd.read_parquet(path), name_candidates=("sensor_name", "camera_name"))
    if "sensor_name" not in df.columns:
        if "camera_name" in df.columns:
            df = df.rename(columns={"camera_name": "sensor_name"})
        else:
            raise ValueError(f"Missing sensor_name column in {path}")

    camera_to_ego: Dict[int, List[List[float]]] = {}
    for camera_id, sensor_name in CAMERA_ID_TO_NAME.items():
        sdf = df[df["sensor_name"].astype(str).str.strip() == sensor_name]
        if sdf.empty:
            raise ValueError(f"Missing sensor extrinsics for {sensor_name} in {path}")
        camera_to_ego[camera_id] = parse_transform_from_row(sdf.iloc[0])
    return camera_to_ego


def read_intrinsics(scene_root: Path) -> Dict[int, Tuple[float, float, float, float]]:
    intrinsics: Dict[int, Tuple[float, float, float, float]] = {}
    for camera_id in sorted(CAMERA_ID_TO_NAME):
        values = [
            float(line.strip())
            for line in (scene_root / "intrinsics" / f"{camera_id}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(values) < 4:
            raise ValueError(
                f"Intrinsics file {(scene_root / 'intrinsics' / f'{camera_id}.txt')} must contain at least 4 values"
            )
        intrinsics[camera_id] = tuple(values[:4])  # type: ignore[assignment]
    return intrinsics


def format_matrix_rows(matrix: Sequence[Sequence[float]], rows: int = 3) -> str:
    values: List[str] = []
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


def write_pose(dst_scene: Path, frame_name: str, ego_to_world: Sequence[Sequence[float]]) -> None:
    dst = dst_scene / "pose" / f"{frame_name}.txt"
    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as handle:
        for row in ego_to_world:
            handle.write(" ".join(f"{float(value):.18e}" for value in row) + "\n")


def write_calib(
    dst_scene: Path,
    frame_name: str,
    intrinsics: Dict[int, Tuple[float, float, float, float]],
    camera_to_ego: Dict[int, List[List[float]]],
) -> None:
    dst = dst_scene / "calib" / f"{frame_name}.txt"
    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as handle:
        for camera_id in range(5):
            src_camera_id = min(camera_id, 2)
            fx, fy, cx, cy = intrinsics[src_camera_id]
            handle.write(f"P{camera_id}: {build_projection_line(fx, fy, cx, cy)}\n")
        handle.write(
            "R0_rect: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 "
            "0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00\n"
        )
        for camera_id in range(5):
            src_camera_id = min(camera_id, 2)
            lidar_to_cam = rigid_inverse4(camera_to_ego[src_camera_id])
            handle.write(
                f"Tr_velo_to_cam_{camera_id}: {format_matrix_rows(lidar_to_cam)}\n"
            )


def create_pvg_views(
    dst_scene: Path,
    source_frame_names: Sequence[str],
    output_frame_names: Sequence[str],
) -> None:
    for camera_id in range(3):
        ensure_dir(dst_scene / f"image_{camera_id}")
        ensure_dir(dst_scene / f"sky_{camera_id}")
    ensure_dir(dst_scene / "pose")
    ensure_dir(dst_scene / "calib")
    ensure_dir(dst_scene / "velodyne")

    for source_frame_name, output_frame_name in zip(source_frame_names, output_frame_names):
        for camera_id in range(3):
            copy_file(
                dst_scene / "images" / f"{source_frame_name}_{camera_id}.png",
                dst_scene / f"image_{camera_id}" / f"{output_frame_name}.png",
            )
            copy_file(
                dst_scene / "sky_masks" / f"{source_frame_name}_{camera_id}.png",
                dst_scene / f"sky_{camera_id}" / f"{output_frame_name}.png",
            )


def ensure_scene_paths(scene_dir: Path) -> Tuple[Path, Path, Path]:
    lidar_parquet = scene_dir / "lidar" / LIDAR_TOP_NAME / f"{scene_dir.name}.{LIDAR_TOP_NAME}.parquet"
    camera_timestamps = (
        scene_dir
        / "camera"
        / CAMERA_FRONT_NAME
        / f"{scene_dir.name}.{CAMERA_FRONT_NAME}.timestamps.parquet"
    )
    velodyne_dir = scene_dir / "velodyne"

    if not lidar_parquet.is_file():
        raise FileNotFoundError(f"Missing lidar parquet: {lidar_parquet}")
    if not camera_timestamps.is_file():
        raise FileNotFoundError(f"Missing front-camera timestamps parquet: {camera_timestamps}")
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    return lidar_parquet, camera_timestamps, velodyne_dir


def nearest_lidar_indices(
    camera_timestamps: Sequence[float],
    lidar_timestamps: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    lidar_timestamps = np.asarray(lidar_timestamps, dtype=np.float64)
    camera_timestamps = np.asarray(camera_timestamps, dtype=np.float64)
    order = np.argsort(lidar_timestamps)
    sorted_lidar = lidar_timestamps[order]
    insertions = np.searchsorted(sorted_lidar, camera_timestamps, side="left")

    matched = np.zeros(len(camera_timestamps), dtype=np.int64)
    deltas = np.zeros(len(camera_timestamps), dtype=np.float64)

    for idx, timestamp in enumerate(camera_timestamps):
        choices: List[int] = []
        insertion = int(insertions[idx])
        if insertion < len(sorted_lidar):
            choices.append(insertion)
        if insertion > 0:
            choices.append(insertion - 1)
        if not choices:
            raise ValueError("No lidar timestamps available for matching.")
        best_local = min(choices, key=lambda choice: abs(sorted_lidar[choice] - timestamp))
        matched[idx] = int(order[best_local])
        deltas[idx] = float(abs(sorted_lidar[best_local] - timestamp))

    return matched, deltas


def decode_draco_blob_to_ply(
    decoder_path: Path,
    draco_blob: bytes,
    temp_dir: Path,
    stem: str,
) -> Path:
    drc_path = temp_dir / f"{stem}.drc"
    ply_path = temp_dir / f"{stem}.ply"
    drc_path.write_bytes(draco_blob)
    result = subprocess.run(
        [str(decoder_path), "-i", str(drc_path), "-o", str(ply_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        output = (result.stdout or "").strip()
        message = f"draco_decoder failed for {drc_path} with exit code {result.returncode}."
        if output:
            message = f"{message}\n{output}"
        raise RuntimeError(message)
    if not ply_path.is_file():
        raise FileNotFoundError(f"Decoded PLY is missing: {ply_path}")
    return ply_path


def ply_dtype_from_property(type_name: str) -> np.dtype:
    type_map = {
        "char": np.int8,
        "uchar": np.uint8,
        "short": np.int16,
        "ushort": np.uint16,
        "int": np.int32,
        "uint": np.uint32,
        "float": np.float32,
        "double": np.float64,
    }
    if type_name not in type_map:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    return np.dtype(type_map[type_name])


def parse_binary_ply_vertices(ply_path: Path) -> np.ndarray:
    with ply_path.open("rb") as handle:
        header_lines: List[str] = []
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {ply_path}")
            line = raw_line.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError(f"Invalid PLY file: {ply_path}")
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Only binary_little_endian PLY is supported: {ply_path}")

        vertex_count = None
        in_vertex_element = False
        properties: List[Tuple[str, str]] = []
        for line in header_lines[1:]:
            if line.startswith("element "):
                tokens = line.split()
                in_vertex_element = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(tokens[2])
                continue
            if in_vertex_element and line.startswith("property "):
                tokens = line.split()
                if len(tokens) != 3:
                    raise ValueError(f"Unsupported PLY property declaration: {line}")
                properties.append((tokens[2], tokens[1]))

        if vertex_count is None:
            raise ValueError(f"Missing vertex element in PLY: {ply_path}")
        if not properties:
            raise ValueError(f"PLY vertex element has no properties: {ply_path}")

        dtype = np.dtype([(name, ply_dtype_from_property(type_name)) for name, type_name in properties])
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)

    required = {"x", "y", "z"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"PLY is missing xyz properties: {ply_path}")
    return vertices


def vertices_to_velodyne(vertices: np.ndarray) -> np.ndarray:
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)

    if {"red", "green", "blue"}.issubset(vertices.dtype.names or ()):
        rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float32)
        intensity = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) / 255.0
    else:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)

    elongation = np.zeros((xyz.shape[0],), dtype=np.float32)
    point_time = np.zeros((xyz.shape[0],), dtype=np.float32)
    return np.column_stack([xyz, intensity, elongation, point_time]).astype(np.float32)


def decode_lidar_row_to_velodyne(
    decoder_path: Path,
    draco_blob: bytes,
    temp_dir: Path,
    stem: str,
) -> np.ndarray:
    ply_path = decode_draco_blob_to_ply(decoder_path, draco_blob, temp_dir, stem)
    vertices = parse_binary_ply_vertices(ply_path)
    return vertices_to_velodyne(vertices)


def decode_velodyne(
    scene_root: Path,
    decoder_path: Path,
    max_lidar_delta: float | None,
    source_frame_indices: Sequence[int],
) -> None:
    lidar_parquet, camera_timestamps_path, velodyne_dir = ensure_scene_paths(scene_root)

    lidar_df = pd.read_parquet(lidar_parquet)
    camera_df = pd.read_parquet(camera_timestamps_path)

    if "draco_encoded_pointcloud" not in lidar_df.columns:
        raise ValueError(f"Missing draco_encoded_pointcloud column in {lidar_parquet}")
    if "reference_timestamp" not in lidar_df.columns:
        raise ValueError(f"Missing reference_timestamp column in {lidar_parquet}")
    if "timestamp" not in camera_df.columns:
        raise ValueError(f"Missing timestamp column in {camera_timestamps_path}")

    all_camera_timestamps = camera_df["timestamp"].to_numpy(dtype=np.float64)
    camera_timestamps = all_camera_timestamps[np.asarray(source_frame_indices, dtype=np.int64)]
    lidar_timestamps = lidar_df["reference_timestamp"].to_numpy(dtype=np.float64)
    matched_lidar_indices, deltas = nearest_lidar_indices(camera_timestamps, lidar_timestamps)

    if max_lidar_delta is not None and float(deltas.max()) > float(max_lidar_delta):
        raise ValueError(
            f"Nearest lidar delta {float(deltas.max())} exceeds max allowed {float(max_lidar_delta)}"
        )

    frame_names = [f"{idx:04d}" for idx in range(len(source_frame_indices))]
    unique_lidar_indices = sorted(set(int(idx) for idx in matched_lidar_indices.tolist()))
    decoded_cache: Dict[int, np.ndarray] = {}

    with tempfile.TemporaryDirectory(prefix="pvg_draco_decode_") as temp_root:
        temp_dir = Path(temp_root)
        for lidar_idx in unique_lidar_indices:
            blob = lidar_df.iloc[lidar_idx]["draco_encoded_pointcloud"]
            if isinstance(blob, memoryview):
                blob = blob.tobytes()
            elif hasattr(blob, "as_py"):
                blob = blob.as_py()
            elif isinstance(blob, bytearray):
                blob = bytes(blob)
            elif not isinstance(blob, bytes):
                raise TypeError(
                    f"Unsupported draco blob type at lidar row {lidar_idx}: {type(blob).__name__}"
                )
            decoded_cache[lidar_idx] = decode_lidar_row_to_velodyne(
                decoder_path=decoder_path,
                draco_blob=blob,
                temp_dir=temp_dir,
                stem=f"lidar_{lidar_idx:04d}",
            )

    for output_frame_idx, frame_name in enumerate(frame_names):
        lidar_idx = int(matched_lidar_indices[output_frame_idx])
        decoded_cache[lidar_idx].tofile(velodyne_dir / f"{frame_name}.bin")

    frame_mapping_rows = []
    for output_frame_idx, frame_name in enumerate(frame_names):
        source_frame_idx = int(source_frame_indices[output_frame_idx])
        lidar_idx = int(matched_lidar_indices[output_frame_idx])
        frame_mapping_rows.append(
            {
                "output_frame_index": output_frame_idx,
                "source_frame_index": source_frame_idx,
                "frame_name": frame_name,
                "camera_timestamp": float(camera_timestamps[output_frame_idx]),
                "matched_lidar_index": lidar_idx,
                "matched_lidar_timestamp": float(lidar_timestamps[lidar_idx]),
                "nearest_delta": float(deltas[output_frame_idx]),
                "velodyne_path": str(velodyne_dir / f"{frame_name}.bin"),
            }
        )
    frame_mapping_df = pd.DataFrame(frame_mapping_rows)
    frame_mapping_df.to_csv(scene_root / "velodyne_frame_to_lidar.csv", index=False)

    unique_lidar_rows = []
    lidar_use_counts = np.bincount(matched_lidar_indices, minlength=len(lidar_df))
    for lidar_idx in unique_lidar_indices:
        unique_lidar_rows.append(
            {
                "lidar_index": lidar_idx,
                "lidar_timestamp": float(lidar_timestamps[lidar_idx]),
                "reuse_count": int(lidar_use_counts[lidar_idx]),
                "point_count": int(decoded_cache[lidar_idx].shape[0]),
                "first_output_frame_index": int(
                    frame_mapping_df.loc[
                        frame_mapping_df["matched_lidar_index"] == lidar_idx, "output_frame_index"
                    ].min()
                ),
                "last_output_frame_index": int(
                    frame_mapping_df.loc[
                        frame_mapping_df["matched_lidar_index"] == lidar_idx, "output_frame_index"
                    ].max()
                ),
            }
        )
    pd.DataFrame(unique_lidar_rows).to_csv(scene_root / "velodyne_unique_lidar.csv", index=False)

    meta = {
        "scene": scene_root.name,
        "draco_decoder": str(decoder_path),
        "lidar_parquet": str(lidar_parquet),
        "camera_timestamps": str(camera_timestamps_path),
        "source_camera_frame_count": int(len(camera_df)),
        "output_camera_frame_count": int(len(source_frame_indices)),
        "unique_lidar_rows_used": int(len(unique_lidar_indices)),
        "lidar_row_count": int(len(lidar_df)),
        "nearest_delta_min": float(deltas.min()),
        "nearest_delta_median": float(np.median(deltas)),
        "nearest_delta_max": float(deltas.max()),
        "matched_lidar_index_first_10": [int(idx) for idx in matched_lidar_indices[:10]],
        "matched_delta_first_10": [float(delta) for delta in deltas[:10]],
        "frame_mapping_csv": str(scene_root / "velodyne_frame_to_lidar.csv"),
        "unique_lidar_csv": str(scene_root / "velodyne_unique_lidar.csv"),
    }
    (scene_root / "velodyne_decode_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def write_prepare_meta(dst_scene: Path, src_scene: Path, selection: FrameSelection) -> None:
    meta = {
        "source_scene": str(src_scene),
        "frames": int(selection.output_frame_count),
        "num_cams": 3,
        "source_frame_count": int(selection.source_frame_count),
        "trim_start_frames": int(selection.trim_start_frames),
        "trim_end_frames": int(selection.trim_end_frames),
        "left_max_delta_s": float(selection.left_max_delta_s),
        "right_max_delta_s": float(selection.right_max_delta_s),
        "ego_max_delta_s": float(selection.ego_max_delta_s),
        "notes": [
            "Generated PVG directories image_{0..2}, sky_{0..2}, pose, calib, velodyne.",
            "Images are copied from already-rectified images/*.png in the source scene.",
            "Intrinsics are read from aligned rectified intrinsics/*.txt in the source scene.",
            "Pose is aligned from labels/egomotion using front-camera timestamps.",
            "Calib is derived from calibration/sensor_extrinsics.parquet.",
            "Velodyne is decoded from lidar/lidar_top_360fov parquet.",
            "Only the largest contiguous multi-sensor-aligned frame window is exported.",
        ],
    }
    (dst_scene / "pvg_prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def convert_one_scene(scene: SceneSpec, decoder_path: Path, max_lidar_delta: float | None) -> Tuple[str, int]:
    validate_required_paths(scene.src_scene)
    source_frame_names, _, _ = collect_frame_coverage(scene.src_scene)
    selection = select_frames(scene.src_scene)
    copy_scene_tree(scene.src_scene, scene.dst_scene, overwrite=False)

    selected_source_frame_names = [source_frame_names[idx] for idx in selection.frame_indices]
    output_frame_names = [f"{idx:04d}" for idx in range(selection.output_frame_count)]
    create_pvg_views(scene.dst_scene, selected_source_frame_names, output_frame_names)

    intrinsics = read_intrinsics(scene.dst_scene)
    camera_to_ego = load_camera_to_ego(scene.dst_scene)

    for frame_idx, frame_name in enumerate(output_frame_names):
        write_pose(scene.dst_scene, frame_name, selection.ego_to_world[frame_idx])
        write_calib(scene.dst_scene, frame_name, intrinsics, camera_to_ego)

    decode_velodyne(scene.dst_scene, decoder_path, max_lidar_delta, selection.frame_indices)
    write_prepare_meta(scene.dst_scene, scene.src_scene, selection)
    return f"{scene.category}/{scene.uuid}", selection.output_frame_count


def main() -> int:
    args = parse_args()
    scenes = discover_scenes(args.src_root.resolve(), args.dst_root.resolve(), args.scene)
    decoder_path = resolve_draco_decoder(args)

    converted: List[Tuple[str, int]] = []
    failures: List[Tuple[str, str]] = []

    for scene in scenes:
        try:
            if scene.dst_scene.exists():
                if not args.overwrite:
                    raise FileExistsError(f"Destination scene already exists: {scene.dst_scene}")
                shutil.rmtree(scene.dst_scene)
            converted.append(convert_one_scene(scene, decoder_path, args.max_lidar_delta))
            print(f"Converted {scene.category}/{scene.uuid}")
        except Exception as exc:
            failures.append((f"{scene.category}/{scene.uuid}", str(exc)))
            print(f"Failed {scene.category}/{scene.uuid}: {exc}")

    print(f"Converted scenes: {len(converted)}")
    for scene_name, frame_count in converted:
        print(f"  {scene_name}: {frame_count} frames")

    if failures:
        print(f"Failures: {len(failures)}")
        for scene_name, error in failures:
            print(f"  {scene_name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
