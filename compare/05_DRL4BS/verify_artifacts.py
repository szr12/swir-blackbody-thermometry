"""Verify final synthetic/inspection artifacts; never train or read raw spectra."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from drl4bs.selector import source_hashes


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def main():
    result = read_json(ROOT / "outputs/cuda_paper/result.json")
    require(result["profile"] == "paper", "expected formal-profile synthetic run")
    require(result["episodes_completed"] == 10000, "incomplete budget")
    require(result["gradient_updates"] == 9984, "wrong replay-update count")
    require(result["replay_size"] == 50000, "wrong final replay size")
    require(result["stop_reason"] == "budget_exhausted", "wrong stop reason")
    require(result["runtime"]["device"] == "cuda", "CUDA run required")
    require("NVIDIA" in result["runtime"]["gpu"], "actual NVIDIA GPU required")
    require(len(result["greedy_trajectory"]) == 100, "missing periodic greedy diagnostics")
    require([v["episode"] for v in result["greedy_trajectory"]] == list(range(100, 10001, 100)),
            "wrong trajectory sampling schedule")
    require(len(result["loss_history"]) == 9984, "missing losses")
    require(all(np.isfinite(v["loss"]) for v in result["loss_history"]), "nonfinite loss")
    require(len(result["epsilon_history"]) == 10000, "missing epsilon history")
    require(result["epsilon_history"][-1] == .01, "epsilon lower bound mismatch")
    require(result["source_sha256"] == source_hashes(), "demo source is not current source")
    order = result.get("selection_order_horizon", result.get("selection_order_6"))
    require(len(set(order)) == 6, "duplicate bands")
    for k in (4, 5, 6):
        require(result["selections"][str(k)] == sorted(order[:k]), "prefix mismatch")
    model_path = ROOT / "outputs/cuda_paper/model.pt"
    require(digest(model_path) == result["model_sha256"], "model hash mismatch")
    require(all(bool(torch.isfinite(v).all())
                for v in torch.load(model_path, weights_only=True).values()), "nonfinite saved weights")
    inspection = read_json(ROOT / "outputs/blackbody_format_inspection.json")
    require(inspection["training_started"] is False, "real data must be inspection-only")
    adapter_paths = {
        "01_XGBS": "xgbs", "02_IGAEBS": "igaebs", "07_SGAE": "sgae",
        "04_mRmR": "mrmr", "05_MI Top-K": "mifs", "02_CARS": "cars", "05_DRL4BS": "drl4bs",
    }
    adapter_hashes = {directory: digest(ROOT.parent / directory / package / "data.py")
                      for directory, package in adapter_paths.items()}
    require(set(adapter_hashes.values()) ==
            {"6466bcd39ce70c290c771833bf26719d42adea6c4ffa815fd494ea019caefcf9"},
            "shared adapter identity mismatch")
    paper_hash = digest(ROOT / "paper_DRL4BS_TGRS_2022.pdf")
    archive_hash = digest(ROOT / "official_historical_9ab5c9c.zip")
    require(paper_hash == "345f10b479a8844e01513b316b964b1a8f20cf0994986c6cc7575d5c490e4c1b",
            "original paper changed")
    require(archive_hash == "63c061b30bf3ccfc1c0d8ad03748a33b576204c763c3471f56189e103742710d",
            "official snapshot changed")
    suites = list(ET.parse(ROOT / "outputs/tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0,
            "tests did not all pass on server")
    acceptance = {
        "status": "passed", "tests": counts,
        "synthetic_only": True, "real_data_training_started": False,
        "actual_runtime": result["runtime"],
        "episodes": result["episodes_completed"], "updates": result["gradient_updates"],
        "stop_reason": result["stop_reason"], "selections": result["selections"],
        "training_seconds": result["training_seconds"],
        "final_inference_seconds": result["final_inference_seconds"],
        "blackbody_record_count": inspection["record_count"],
        "data_adapter_sha256": adapter_hashes, "source_sha256": result["source_sha256"],
        "paper_sha256": paper_hash, "official_archive_sha256": archive_hash,
        "model_sha256": result["model_sha256"],
    }
    (ROOT / "outputs/acceptance.json").write_text(
        json.dumps(acceptance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "SOURCE_SHA256SUMS.txt").write_text(
        "".join(f"{checksum}  {name}\n" for name, checksum in source_hashes().items()),
        encoding="utf-8")
    print(json.dumps(acceptance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
