"""Exact streaming sufficient statistics for HCBH HSI inputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import EnviCubeRecord, open_envi_cube


@dataclass(frozen=True)
class CubeLayout:
    record_count: int
    common_lines: int
    samples: int
    bands: int
    pixel_count: int
    spectral_range: str | None
    acquisition_tag: str | None
    wavelengths_nm: np.ndarray | None
    dropped_lines_per_record: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "common_lines": self.common_lines,
            "samples": self.samples,
            "bands": self.bands,
            "pixel_count": self.pixel_count,
            "spectral_range": self.spectral_range,
            "acquisition_tag": self.acquisition_tag,
            "wavelengths_nm": None if self.wavelengths_nm is None else self.wavelengths_nm.tolist(),
            "dropped_lines_per_record": list(self.dropped_lines_per_record),
        }


@dataclass(frozen=True)
class StreamingStatistics:
    raw_gram: np.ndarray
    scaled_gram: np.ndarray
    global_minimum: float
    global_maximum: float
    raw_band_ranges: np.ndarray
    constant_bands: np.ndarray
    zero_vector_bands: np.ndarray
    preprojection_norms: np.ndarray
    projected_norms: np.ndarray
    ball_scale: float
    data_hash: str
    source_fingerprint: str
    layout: CubeLayout
    checkpoint_restored: bool


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _prepare_inputs(
    items: Sequence[Any],
) -> tuple[list[np.ndarray], tuple[EnviCubeRecord, ...], CubeLayout, str]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        raise ValueError("cubes_or_records must be a non-empty sequence")
    contains_records = [isinstance(item, EnviCubeRecord) for item in items]
    if any(contains_records) and not all(contains_records):
        raise ValueError("do not mix EnviCubeRecord and ndarray inputs")

    if all(contains_records):
        records = tuple(sorted(items, key=lambda record: record.group_id))
        cubes = [open_envi_cube(record) for record in records]
    else:
        records = ()
        cubes = [np.asarray(item) for item in items]

    for cube in cubes:
        if cube.ndim != 3:
            raise ValueError("each cube must have shape [H,W,B]")
        if cube.shape[0] < 1 or cube.shape[1] < 1 or cube.shape[2] < 2:
            raise ValueError("each cube must contain pixels and at least two bands")
        if not np.issubdtype(cube.dtype, np.number):
            raise ValueError("cube values must be numeric")
    if len({cube.shape[1] for cube in cubes}) != 1 or len({cube.shape[2] for cube in cubes}) != 1:
        raise ValueError("all cubes must share samples and band count")

    spectral_range = acquisition_tag = None
    wavelengths = None
    if records:
        if len({record.spectral_range for record in records}) != 1:
            raise ValueError("VNIR and SWIR records must be fitted separately")
        if len({record.acquisition_tag for record in records}) != 1:
            raise ValueError("acquisition tags must be fitted separately")
        if len({record.y_start for record in records}) != 1:
            raise ValueError("all records must share the same line start")
        reference = records[0]
        if any(not np.array_equal(record.wavelengths_nm, reference.wavelengths_nm) for record in records[1:]):
            raise ValueError("all records must share the same wavelength grid")
        spectral_range = reference.spectral_range
        acquisition_tag = reference.acquisition_tag
        wavelengths = reference.wavelengths_nm.copy()

    common_lines = min(cube.shape[0] for cube in cubes)
    samples, bands = cubes[0].shape[1:]
    layout = CubeLayout(
        record_count=len(cubes),
        common_lines=common_lines,
        samples=samples,
        bands=bands,
        pixel_count=len(cubes) * common_lines * samples,
        spectral_range=spectral_range,
        acquisition_tag=acquisition_tag,
        wavelengths_nm=wavelengths,
        dropped_lines_per_record=tuple(cube.shape[0] - common_lines for cube in cubes),
    )
    if records:
        source = [
            {
                "path": str(record.data_path),
                "size": record.data_path.stat().st_size,
                "mtime_ns": record.data_path.stat().st_mtime_ns,
                "shape": list(cube.shape),
                "dtype": cube.dtype.str,
                "group_id": record.group_id,
                "common_lines": common_lines,
            }
            for cube, record in zip(cubes, records, strict=True)
        ]
    else:
        source = [
            {
                "index": index,
                "shape": list(cube.shape),
                "dtype": cube.dtype.str,
                "sha256": hashlib.sha256(np.ascontiguousarray(cube[:common_lines]).tobytes()).hexdigest(),
            }
            for index, cube in enumerate(cubes)
        ]
    return cubes, records, layout, _json_hash(source)


def _scan_range_and_hash(
    cubes: Sequence[np.ndarray], layout: CubeLayout, chunk_lines: int
) -> tuple[float, float, np.ndarray, str]:
    minimum = float("inf")
    maximum = float("-inf")
    band_min = np.full(layout.bands, np.inf, dtype=np.float64)
    band_max = np.full(layout.bands, -np.inf, dtype=np.float64)
    record_hashes: list[str] = []
    for cube in cubes:
        digest = hashlib.sha256()
        for start in range(0, layout.common_lines, chunk_lines):
            stop = min(start + chunk_lines, layout.common_lines)
            block = np.asarray(cube[start:stop])
            if not np.isfinite(block).all():
                raise ValueError("cubes must contain only finite values")
            contiguous = np.ascontiguousarray(block)
            digest.update(contiguous.tobytes())
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
            band_min = np.minimum(band_min, block.min(axis=(0, 1)))
            band_max = np.maximum(band_max, block.max(axis=(0, 1)))
        record_hashes.append(digest.hexdigest())
    if not maximum > minimum:
        raise ValueError("global training value range must be non-zero")
    return minimum, maximum, band_max - band_min, _json_hash(record_hashes)


def _gram_for_cube(
    cube: np.ndarray,
    layout: CubeLayout,
    minimum: float,
    value_range: float,
    chunk_lines: int,
) -> np.ndarray:
    gram = np.zeros((layout.bands, layout.bands), dtype=np.float64)
    for start in range(0, layout.common_lines, chunk_lines):
        stop = min(start + chunk_lines, layout.common_lines)
        block = np.asarray(cube[start:stop], dtype=np.float64)
        matrix = ((block - minimum) / value_range).reshape(-1, layout.bands)
        gram += matrix.T @ matrix
    return gram


def collect_streaming_statistics(
    items: Sequence[Any],
    *,
    chunk_lines: int = 32,
    ball_margin: float = 1e-12,
    checkpoint_path: str | Path | None = None,
    config_hash: str,
) -> StreamingStatistics:
    if isinstance(chunk_lines, bool) or not isinstance(chunk_lines, int) or chunk_lines <= 0:
        raise ValueError("chunk_lines must be a positive integer")
    if not 0.0 < ball_margin < 1.0:
        raise ValueError("ball_margin must be in (0, 1)")
    cubes, _records, layout, fingerprint = _prepare_inputs(items)
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    manifest_path = None if checkpoint is None else checkpoint / "statistics.json"
    state_path = None if checkpoint is None else checkpoint / "statistics.npz"
    restored = False

    minimum, maximum, raw_ranges, data_hash = _scan_range_and_hash(cubes, layout, chunk_lines)
    value_range = maximum - minimum
    raw_gram = np.zeros((layout.bands, layout.bands), dtype=np.float64)
    completed = 0
    if manifest_path is not None and manifest_path.is_file() and state_path is not None and state_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "config_hash": config_hash,
            "source_fingerprint": fingerprint,
            "data_hash": data_hash,
            "bands": layout.bands,
            "global_minimum": minimum,
            "global_maximum": maximum,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint does not match the current data or configuration")
        saved = np.load(state_path)
        raw_gram = np.asarray(saved["raw_gram"], dtype=np.float64)
        completed = int(manifest["completed_records"])
        saved_completed = int(saved["completed_records"])
        if (
            raw_gram.shape != (layout.bands, layout.bands)
            or saved_completed != completed
            or not 0 <= completed <= len(cubes)
        ):
            raise ValueError("checkpoint state is malformed")
        restored = completed > 0

    for index in range(completed, len(cubes)):
        raw_gram += _gram_for_cube(cubes[index], layout, minimum, value_range, chunk_lines)
        completed = index + 1
        if manifest_path is not None and state_path is not None:
            atomic_write_npz(
                state_path,
                raw_gram=raw_gram,
                completed_records=np.asarray(completed, dtype=np.int64),
            )
            atomic_write_json(
                manifest_path,
                {
                    "phase": "gram_complete" if completed == len(cubes) else "gram_partial",
                    "config_hash": config_hash,
                    "source_fingerprint": fingerprint,
                    "data_hash": data_hash,
                    "bands": layout.bands,
                    "global_minimum": minimum,
                    "global_maximum": maximum,
                    "completed_records": completed,
                },
            )

    raw_gram = (raw_gram + raw_gram.T) * 0.5
    preprojection_norms = np.sqrt(np.maximum(np.diag(raw_gram), 0.0))
    maximum_norm = float(preprojection_norms.max())
    if not maximum_norm > 0.0:
        raise ValueError("all normalized band vectors are zero")
    ball_scale = (1.0 - ball_margin) / maximum_norm
    scaled_gram = raw_gram * ball_scale**2
    scaled_gram = (scaled_gram + scaled_gram.T) * 0.5
    projected_norms = np.sqrt(np.maximum(np.diag(scaled_gram), 0.0))
    if np.any(projected_norms >= 1.0):
        correction = (1.0 - ball_margin) / float(projected_norms.max())
        ball_scale *= correction
        scaled_gram *= correction**2
        projected_norms = np.sqrt(np.maximum(np.diag(scaled_gram), 0.0))
    return StreamingStatistics(
        raw_gram=raw_gram,
        scaled_gram=scaled_gram,
        global_minimum=minimum,
        global_maximum=maximum,
        raw_band_ranges=raw_ranges,
        constant_bands=np.flatnonzero(raw_ranges == 0.0),
        zero_vector_bands=np.flatnonzero(projected_norms == 0.0),
        preprojection_norms=preprojection_norms,
        projected_norms=projected_norms,
        ball_scale=float(ball_scale),
        data_hash=data_hash,
        source_fingerprint=fingerprint,
        layout=layout,
        checkpoint_restored=restored,
    )


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "dtype": "float64",
    }
