from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


_ENVI_DTYPES = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    12: "u2",
    13: "u4",
    14: "i8",
    15: "u8",
}
_NAME_PATTERN = re.compile(
    r"^(?P<temperature>[+-]?\d+(?:\.\d+)?)_black_(?P<tag>.+?)_FS[^()]+"
    r"\((?P<spectral_range>[^()]+)\)$",
    re.IGNORECASE,
)
_RANGE_ALIASES = {
    "vnir": "400-1000",
    "swir": "900-2500",
    "400-1000": "400-1000",
    "900-2500": "900-2500",
}


def _parse_envi_header(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    if not lines or lines[0].strip().upper() != "ENVI":
        raise ValueError(f"not an ENVI header: {path}")

    result: dict[str, str] = {}
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        while value.count("{") > value.count("}") and index < len(lines):
            value += " " + lines[index].strip()
            index += 1
        result[key.lower()] = value
    return result


def _integer(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing or invalid ENVI field {key!r}") from exc


def _float_list(value: str, key: str) -> np.ndarray:
    stripped = value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise ValueError(f"ENVI field {key!r} must use braces")
    try:
        values = [float(item.strip()) for item in stripped[1:-1].split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"ENVI field {key!r} contains a non-numeric value") from exc
    return np.asarray(values, dtype=np.float64)


def _optional_bool(metadata: dict[str, str], key: str) -> bool | None:
    if key not in metadata:
        return None
    value = metadata[key].strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid boolean ENVI field {key!r}")


def _normalize_range(value: str) -> str:
    normalized = value.strip().lower().replace("–", "-").replace("—", "-")
    return _RANGE_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class EnviCubeRecord:
    """Metadata for one blackbody ENVI ``.hdr``/``.spe`` acquisition."""

    header_path: Path
    data_path: Path
    temperature_c: float
    acquisition_tag: str
    spectral_range: str
    samples: int
    lines: int
    bands: int
    header_offset: int
    data_type: int
    interleave: str
    byte_order: int
    wavelengths_nm: np.ndarray
    x_start: int
    y_start: int
    exposure_time: float | None
    dark_current_corrected: bool | None

    @property
    def group_id(self) -> str:
        return self.header_path.stem

    @property
    def dtype(self) -> np.dtype[Any]:
        try:
            code = _ENVI_DTYPES[self.data_type]
        except KeyError as exc:
            raise ValueError(f"unsupported ENVI data type: {self.data_type}") from exc
        if code == "u1":
            return np.dtype(code)
        endian = "<" if self.byte_order == 0 else ">"
        return np.dtype(endian + code)

    @property
    def expected_file_size(self) -> int:
        values = self.lines * self.samples * self.bands
        return self.header_offset + values * self.dtype.itemsize

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "header_path": str(self.header_path),
            "data_path": str(self.data_path),
            "temperature_c": self.temperature_c,
            "acquisition_tag": self.acquisition_tag,
            "spectral_range": self.spectral_range,
            "samples": self.samples,
            "lines": self.lines,
            "bands": self.bands,
            "header_offset": self.header_offset,
            "data_type": self.data_type,
            "dtype": self.dtype.str,
            "interleave": self.interleave,
            "byte_order": self.byte_order,
            "wavelength_min_nm": float(self.wavelengths_nm[0]),
            "wavelength_max_nm": float(self.wavelengths_nm[-1]),
            "x_start": self.x_start,
            "y_start": self.y_start,
            "exposure_time": self.exposure_time,
            "dark_current_corrected": self.dark_current_corrected,
            "expected_file_size": self.expected_file_size,
            "actual_file_size": self.data_path.stat().st_size,
        }


def parse_blackbody_record(header_path: str | Path, *, strict_size: bool = True) -> EnviCubeRecord:
    header = Path(header_path)
    match = _NAME_PATTERN.match(header.stem)
    if match is None:
        raise ValueError(f"unsupported blackbody filename: {header.name}")
    metadata = _parse_envi_header(header)
    data_path = header.with_suffix(".spe")
    if not data_path.is_file():
        raise FileNotFoundError(f"missing paired SPE file: {data_path}")

    bands = _integer(metadata, "bands")
    wavelengths = _float_list(metadata.get("wavelength", ""), "wavelength")
    if wavelengths.size != bands:
        raise ValueError(
            f"wavelength count {wavelengths.size} does not match bands {bands}: {header}"
        )
    interleave = metadata.get("interleave", "").strip().lower()
    if interleave not in {"bil", "bip", "bsq"}:
        raise ValueError(f"unsupported ENVI interleave {interleave!r}: {header}")
    byte_order = _integer(metadata, "byte order")
    if byte_order not in {0, 1}:
        raise ValueError(f"byte order must be 0 or 1: {header}")

    exposure = metadata.get("exposure time")
    record = EnviCubeRecord(
        header_path=header.resolve(),
        data_path=data_path.resolve(),
        temperature_c=float(match.group("temperature")),
        acquisition_tag=match.group("tag"),
        spectral_range=_normalize_range(match.group("spectral_range")),
        samples=_integer(metadata, "samples"),
        lines=_integer(metadata, "lines"),
        bands=bands,
        header_offset=_integer(metadata, "header offset"),
        data_type=_integer(metadata, "data type"),
        interleave=interleave,
        byte_order=byte_order,
        wavelengths_nm=wavelengths,
        x_start=int(metadata.get("x start", "0")),
        y_start=int(metadata.get("y start", "0")),
        exposure_time=float(exposure) if exposure is not None else None,
        dark_current_corrected=_optional_bool(metadata, "deducting dark current"),
    )
    if record.samples <= 0 or record.lines <= 0 or record.bands <= 0:
        raise ValueError(f"ENVI dimensions must be positive: {header}")
    actual_size = data_path.stat().st_size
    if strict_size and actual_size != record.expected_file_size:
        raise ValueError(
            f"SPE size mismatch for {data_path.name}: expected "
            f"{record.expected_file_size}, got {actual_size}"
        )
    return record


def scan_blackbody_directory(
    root: str | Path,
    *,
    recursive: bool = False,
    strict_size: bool = True,
) -> list[EnviCubeRecord]:
    """Discover source acquisitions without reading pixel data.

    By default only the dataset root is scanned, so derived products under
    directories such as ``comparison_200C`` cannot enter an experiment.
    """

    directory = Path(root)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    headers = directory.rglob("*.hdr") if recursive else directory.glob("*.hdr")
    records: list[EnviCubeRecord] = []
    for header in sorted(headers, key=lambda item: item.name.lower()):
        if _NAME_PATTERN.match(header.stem) is None:
            continue
        records.append(parse_blackbody_record(header, strict_size=strict_size))
    if not records:
        raise ValueError(f"no supported blackbody ENVI acquisitions found in {directory}")
    return records


def open_envi_cube(record: EnviCubeRecord) -> np.ndarray:
    """Return a zero-copy logical ``[lines, samples, bands]`` memmap view."""

    raw = np.memmap(
        record.data_path,
        dtype=record.dtype,
        mode="r",
        offset=record.header_offset,
    )
    if record.interleave == "bil":
        return raw.reshape(record.lines, record.bands, record.samples).transpose(0, 2, 1)
    if record.interleave == "bip":
        return raw.reshape(record.lines, record.samples, record.bands)
    return raw.reshape(record.bands, record.lines, record.samples).transpose(1, 2, 0)


@dataclass(frozen=True)
class BlackbodyTable:
    X: np.ndarray
    y_temperature_c: np.ndarray
    groups: np.ndarray
    wavelengths_nm: np.ndarray
    line_indices: np.ndarray
    sample_indices: np.ndarray
    record_indices: np.ndarray
    records: tuple[EnviCubeRecord, ...]
    reduction: str
    alignment: str


def _select_records(
    records: Sequence[EnviCubeRecord],
    spectral_range: str,
    acquisition_tags: Iterable[str] | None,
) -> list[EnviCubeRecord]:
    wanted_range = _normalize_range(spectral_range)
    tags = None if acquisition_tags is None else set(acquisition_tags)
    selected = [
        record
        for record in records
        if record.spectral_range == wanted_range
        and (tags is None or record.acquisition_tag in tags)
    ]
    if not selected:
        raise ValueError(
            f"no records match spectral_range={wanted_range!r} and acquisition_tags={tags!r}"
        )
    reference = selected[0]
    for record in selected[1:]:
        if record.bands != reference.bands or not np.array_equal(
            record.wavelengths_nm, reference.wavelengths_nm
        ):
            raise ValueError("selected records do not share the same wavelength grid")
        if record.y_start != reference.y_start:
            raise ValueError("selected records do not share the same line start")
    return selected


def _line_limit(records: Sequence[EnviCubeRecord], alignment: str) -> int | None:
    if alignment == "common_prefix":
        return min(record.lines for record in records)
    if alignment == "native":
        return None
    raise ValueError("alignment must be 'common_prefix' or 'native'")


def _line_means(cube: np.ndarray, line_count: int, chunk_lines: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, line_count, chunk_lines):
        stop = min(start + chunk_lines, line_count)
        block = np.asarray(cube[start:stop], dtype=np.float64)
        chunks.append(block.mean(axis=1).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _cube_mean(cube: np.ndarray, line_count: int, chunk_lines: int) -> np.ndarray:
    total: np.ndarray | None = None
    count = 0
    for start in range(0, line_count, chunk_lines):
        stop = min(start + chunk_lines, line_count)
        block = np.asarray(cube[start:stop], dtype=np.float64)
        block_sum = block.sum(axis=(0, 1))
        total = block_sum if total is None else total + block_sum
        count += block.shape[0] * block.shape[1]
    assert total is not None
    return (total / count).astype(np.float32, copy=False)


def load_blackbody_table(
    root: str | Path,
    *,
    spectral_range: str,
    acquisition_tags: Iterable[str] | None = None,
    reduction: str = "line_mean",
    alignment: str = "common_prefix",
    max_samples_per_cube: int | None = None,
    random_state: int = 0,
    chunk_lines: int = 32,
) -> BlackbodyTable:
    """Convert ENVI cubes into a tabular dataset without training a model.

    ``line_mean`` is the conservative default for scans with a shared starting
    position.  ``common_prefix`` aligns every acquisition at line zero and
    truncates to the shortest cube, preventing longer recordings from receiving
    greater weight.  ``sampled_pixels`` requires an explicit per-cube cap.
    """

    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer")
    records = _select_records(
        scan_blackbody_directory(root), spectral_range, acquisition_tags
    )
    shared_limit = _line_limit(records, alignment)
    if reduction not in {"line_mean", "cube_mean", "sampled_pixels"}:
        raise ValueError("reduction must be 'line_mean', 'cube_mean', or 'sampled_pixels'")
    if reduction == "sampled_pixels" and (
        not isinstance(max_samples_per_cube, int)
        or isinstance(max_samples_per_cube, bool)
        or max_samples_per_cube <= 0
    ):
        raise ValueError("sampled_pixels requires a positive max_samples_per_cube")

    rows: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    line_indices: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    record_indices: list[np.ndarray] = []
    rng = np.random.default_rng(random_state)

    for record_index, record in enumerate(records):
        cube = open_envi_cube(record)
        line_count = shared_limit if shared_limit is not None else record.lines
        if reduction == "line_mean":
            x_part = _line_means(cube, line_count, chunk_lines)
            line_part = np.arange(line_count, dtype=np.int64) + record.y_start
            sample_part = np.full(line_count, -1, dtype=np.int64)
        elif reduction == "cube_mean":
            x_part = _cube_mean(cube, line_count, chunk_lines)[None, :]
            line_part = np.asarray([-1], dtype=np.int64)
            sample_part = np.asarray([-1], dtype=np.int64)
        else:
            assert max_samples_per_cube is not None
            population = line_count * record.samples
            count = min(max_samples_per_cube, population)
            flat = np.sort(rng.choice(population, size=count, replace=False))
            local_lines, local_samples = np.divmod(flat, record.samples)
            x_part = np.asarray(cube[local_lines, local_samples, :], dtype=np.float32)
            line_part = local_lines.astype(np.int64) + record.y_start
            sample_part = local_samples.astype(np.int64) + record.x_start

        n_rows = x_part.shape[0]
        rows.append(x_part)
        temperatures.append(np.full(n_rows, record.temperature_c, dtype=np.float64))
        groups.append(np.full(n_rows, record.group_id, dtype=object))
        line_indices.append(line_part)
        sample_indices.append(sample_part)
        record_indices.append(np.full(n_rows, record_index, dtype=np.int64))

    return BlackbodyTable(
        X=np.concatenate(rows, axis=0),
        y_temperature_c=np.concatenate(temperatures),
        groups=np.concatenate(groups),
        wavelengths_nm=records[0].wavelengths_nm.copy(),
        line_indices=np.concatenate(line_indices),
        sample_indices=np.concatenate(sample_indices),
        record_indices=np.concatenate(record_indices),
        records=tuple(records),
        reduction=reduction,
        alignment=alignment,
    )


def inspect_blackbody_directory(root: str | Path) -> dict[str, Any]:
    """Build a JSON-ready metadata report without reading cube samples."""

    records = scan_blackbody_directory(root)
    ranges: dict[str, Any] = {}
    for spectral_range in sorted({record.spectral_range for record in records}):
        subset = [record for record in records if record.spectral_range == spectral_range]
        ranges[spectral_range] = {
            "record_count": len(subset),
            "temperatures_c": sorted({record.temperature_c for record in subset}),
            "acquisition_tags": sorted({record.acquisition_tag for record in subset}),
            "samples": sorted({record.samples for record in subset}),
            "bands": sorted({record.bands for record in subset}),
            "line_start_values": sorted({record.y_start for record in subset}),
            "all_same_line_start": len({record.y_start for record in subset}) == 1,
            "line_count_min": min(record.lines for record in subset),
            "line_count_max": max(record.lines for record in subset),
            "common_prefix_lines": min(record.lines for record in subset),
            "wavelength_min_nm": float(subset[0].wavelengths_nm[0]),
            "wavelength_max_nm": float(subset[0].wavelengths_nm[-1]),
            "total_spe_bytes": sum(record.data_path.stat().st_size for record in subset),
        }
    return {
        "root": str(Path(root).resolve()),
        "format": "ENVI",
        "source_interleaves": sorted({record.interleave for record in records}),
        "logical_cube_shape": "[lines, samples, bands]",
        "record_count": len(records),
        "ranges": ranges,
        "records": [record.to_dict() for record in records],
        "training_started": False,
    }
