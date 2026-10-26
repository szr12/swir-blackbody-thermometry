from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from analyze_recapture_swir import scan_records


def build_manifest(input_npz: Path, dataset_root: Path, output_dir: Path) -> Path:
    """根据原始头文件时间为每个样本建立采集日期审计清单。"""
    with np.load(input_npz, allow_pickle=False) as archive:
        group_ids = np.asarray(archive["group_ids"]).astype(str)
        temperatures = np.asarray(archive["y_temperature_c"], dtype=np.float64)

    records = {
        record.group_id: record
        for record in scan_records(dataset_root)
        if "chongxin" not in record.group_id.lower()
    }
    if len(records) != group_ids.size:
        raise ValueError(
            f"原始扫描记录数与输入样本数不一致: records={len(records)}, samples={group_ids.size}"
        )

    rows: list[dict[str, object]] = []
    for index, (group_id, temperature) in enumerate(zip(group_ids, temperatures)):
        if group_id not in records:
            raise ValueError(f"输入样本找不到原始文件: {group_id}")
        header = records[group_id].header_path
        data_path = records[group_id].data_path
        header_mtime = datetime.fromtimestamp(header.stat().st_mtime).astimezone()
        data_mtime = datetime.fromtimestamp(data_path.stat().st_mtime).astimezone()
        rows.append({
            "sample_index": index,
            "group_id": group_id,
            "temperature_c": float(temperature),
            "temperature_parity": int(round(temperature)) % 2,
            "header_path": str(header),
            "data_path": str(data_path),
            "header_mtime": header_mtime.isoformat(),
            "header_date": header_mtime.date().isoformat(),
            "data_mtime": data_mtime.isoformat(),
            "data_date": data_mtime.date().isoformat(),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "swir_acquisition_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    date_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        date = str(row["header_date"])
        stats = date_counts.setdefault(date, {"samples": 0, "odd": 0, "even": 0})
        stats["samples"] += 1
        stats["odd" if row["temperature_parity"] else "even"] += 1
    (output_dir / "acquisition_manifest_summary.json").write_text(
        json.dumps({
            "status": "complete",
            "sample_count": len(rows),
            "date_counts": date_counts,
            "date_source": "原始HDR文件mtime",
            "temperature_used_only_for_audit": True,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成SWIR原始文件采集日期清单")
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_manifest(args.input_npz, args.dataset_root, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
