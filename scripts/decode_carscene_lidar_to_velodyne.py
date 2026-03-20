#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


CAMERA_FRONT_NAME = "camera_front_wide_120fov"
LIDAR_TOP_NAME = "lidar_top_360fov"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode CarTwin lidar Draco parquet into PVG velodyne/*.bin files."
    )
    parser.add_argument("--src-scene", type=Path, required=True, help="CarTwin scene directory.")
    parser.add_argument(
        "--draco-decoder",
        type=Path,
        default=None,
        help="Existing draco_decoder executable. If omitted, the script tries to find or build one.",
    )
    parser.add_argument(
        "--draco-source",
        type=Path,
        default=None,
        help="Draco source tree used to build draco_decoder when it is not already available.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/tmp/pvg_draco_build"),
        help="Temporary build directory used only when draco_decoder must be built.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing velodyne/*.bin files.",
    )
    parser.add_argument(
        "--max-lidar-delta",
        type=float,
        default=None,
        help="Optional maximum allowed nearest-neighbor timestamp delta in raw timestamp units.",
    )
    return parser.parse_args()


def ensure_scene_paths(scene_dir: Path) -> Tuple[Path, Path, Path]:
    lidar_parquet = (
        scene_dir
        / "lidar"
        / LIDAR_TOP_NAME
        / f"{scene_dir.name}.{LIDAR_TOP_NAME}.parquet"
    )
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
    velodyne = np.column_stack([xyz, intensity, elongation, point_time]).astype(np.float32)
    return velodyne


def decode_lidar_row_to_velodyne(
    decoder_path: Path,
    draco_blob: bytes,
    temp_dir: Path,
    stem: str,
) -> np.ndarray:
    ply_path = decode_draco_blob_to_ply(decoder_path, draco_blob, temp_dir, stem)
    vertices = parse_binary_ply_vertices(ply_path)
    return vertices_to_velodyne(vertices)


def main() -> None:
    args = parse_args()
    lidar_parquet, camera_timestamps_path, velodyne_dir = ensure_scene_paths(args.src_scene)
    decoder_path = resolve_draco_decoder(args)

    lidar_df = pd.read_parquet(lidar_parquet)
    camera_df = pd.read_parquet(camera_timestamps_path)

    if "draco_encoded_pointcloud" not in lidar_df.columns:
        raise ValueError(f"Missing draco_encoded_pointcloud column in {lidar_parquet}")
    if "reference_timestamp" not in lidar_df.columns:
        raise ValueError(f"Missing reference_timestamp column in {lidar_parquet}")
    if "timestamp" not in camera_df.columns:
        raise ValueError(f"Missing timestamp column in {camera_timestamps_path}")

    camera_timestamps = camera_df["timestamp"].to_numpy(dtype=np.float64)
    lidar_timestamps = lidar_df["reference_timestamp"].to_numpy(dtype=np.float64)
    matched_lidar_indices, deltas = nearest_lidar_indices(camera_timestamps, lidar_timestamps)

    if args.max_lidar_delta is not None and float(deltas.max()) > float(args.max_lidar_delta):
        raise ValueError(
            f"Nearest lidar delta {float(deltas.max())} exceeds max allowed {float(args.max_lidar_delta)}"
        )

    frame_names = [f"{idx:04d}" for idx in range(len(camera_df))]
    if not args.overwrite:
        existing = [name for name in frame_names if (velodyne_dir / f"{name}.bin").exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing velodyne bins such as {(velodyne_dir / f'{existing[0]}.bin')}. "
                "Pass --overwrite to replace them."
            )

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

    for frame_idx, frame_name in enumerate(frame_names):
        lidar_idx = int(matched_lidar_indices[frame_idx])
        output_path = velodyne_dir / f"{frame_name}.bin"
        decoded_cache[lidar_idx].tofile(output_path)

    frame_mapping_rows = []
    for frame_idx, frame_name in enumerate(frame_names):
        lidar_idx = int(matched_lidar_indices[frame_idx])
        frame_mapping_rows.append(
            {
                "frame_index": frame_idx,
                "frame_name": frame_name,
                "camera_timestamp": float(camera_timestamps[frame_idx]),
                "matched_lidar_index": lidar_idx,
                "matched_lidar_timestamp": float(lidar_timestamps[lidar_idx]),
                "nearest_delta": float(deltas[frame_idx]),
                "velodyne_path": str(velodyne_dir / f"{frame_name}.bin"),
            }
        )
    frame_mapping_df = pd.DataFrame(frame_mapping_rows)
    frame_mapping_df.to_csv(args.src_scene / "velodyne_frame_to_lidar.csv", index=False)

    unique_lidar_rows = []
    lidar_use_counts = np.bincount(matched_lidar_indices, minlength=len(lidar_df))
    for lidar_idx in unique_lidar_indices:
        unique_lidar_rows.append(
            {
                "lidar_index": lidar_idx,
                "lidar_timestamp": float(lidar_timestamps[lidar_idx]),
                "reuse_count": int(lidar_use_counts[lidar_idx]),
                "point_count": int(decoded_cache[lidar_idx].shape[0]),
                "first_frame_index": int(frame_mapping_df.loc[frame_mapping_df["matched_lidar_index"] == lidar_idx, "frame_index"].min()),
                "last_frame_index": int(frame_mapping_df.loc[frame_mapping_df["matched_lidar_index"] == lidar_idx, "frame_index"].max()),
            }
        )
    pd.DataFrame(unique_lidar_rows).to_csv(args.src_scene / "velodyne_unique_lidar.csv", index=False)

    meta = {
        "scene": args.src_scene.name,
        "draco_decoder": str(decoder_path),
        "lidar_parquet": str(lidar_parquet),
        "camera_timestamps": str(camera_timestamps_path),
        "camera_frame_count": int(len(camera_df)),
        "unique_lidar_rows_used": int(len(unique_lidar_indices)),
        "lidar_row_count": int(len(lidar_df)),
        "nearest_delta_min": float(deltas.min()),
        "nearest_delta_median": float(np.median(deltas)),
        "nearest_delta_max": float(deltas.max()),
        "matched_lidar_index_first_10": [int(idx) for idx in matched_lidar_indices[:10]],
        "matched_delta_first_10": [float(delta) for delta in deltas[:10]],
        "frame_mapping_csv": str(args.src_scene / "velodyne_frame_to_lidar.csv"),
        "unique_lidar_csv": str(args.src_scene / "velodyne_unique_lidar.csv"),
    }
    (args.src_scene / "velodyne_decode_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"Decoded {len(unique_lidar_indices)} lidar rows into {len(frame_names)} velodyne bins.")
    print(f"draco_decoder: {decoder_path}")
    print(f"velodyne dir: {velodyne_dir}")
    print(
        "nearest lidar delta (raw units): "
        f"min={float(deltas.min()):.1f}, median={float(np.median(deltas)):.1f}, max={float(deltas.max()):.1f}"
    )


if __name__ == "__main__":
    main()
