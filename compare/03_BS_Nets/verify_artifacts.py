"""Verify final artifacts without training or reading raw spectral samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch

from bsnets.selector import source_hashes


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError(reason)


def main() -> None:
    result = read_json(ROOT / "outputs/cuda_paper/result.json")
    require(result["profile"] == "paper", "paper profile required")
    require(result["epochs_completed"] == 100, "wrong epoch budget")
    require(result["config"]["learning_rate"] == .002, "paper learning rate required")
    require(result["config"]["l1_lambda"] == .01, "paper lambda required")
    require(result["config"]["batch_size"] == 64, "official batch size required")
    expected_steps = result["epochs_completed"] * result["batches_per_epoch"]
    require(result["optimizer_steps"] == expected_steps, "wrong optimizer-step count")
    require(result["stop_reason"] == "maximum_epochs_reached", "wrong stop reason")
    require(result["runtime"]["device"] == "cuda", "actual CUDA run required")
    require("NVIDIA" in result["runtime"]["gpu"], "actual NVIDIA GPU required")
    require(len(result["loss_history"]) == 100, "incomplete history")
    for item in result["loss_history"]:
        require(all(np.isfinite(item[key]) for key in
                    ("loss", "reconstruction_loss", "l1_loss")), "nonfinite loss")
        require(np.isfinite(item["mean_attention"]).all(), "nonfinite attention")
    require(result["excluded_constant_bands"] == [0], "constant-band test failed")
    require(result["source_sha256"] == source_hashes(), "result predates current source")
    order = result["selection_order"]
    require(len(order) == len(set(order)) == len(result["retained_bands"]),
            "ranking must cover each retained band once")
    for k in (4, 5, 6):
        require(result["selections"][str(k)] == sorted(order[:k]), "nested prefix mismatch")
    model_path = ROOT / "outputs/cuda_paper/model.pt"
    require(digest(model_path) == result["model_sha256"], "model hash mismatch")
    require(all(bool(torch.isfinite(value).all())
                for value in torch.load(model_path, weights_only=True).values()),
            "nonfinite saved model")

    inspection = read_json(ROOT / "outputs/blackbody_format_inspection.json")
    require(inspection["training_started"] is False, "real data must remain inspection-only")
    require(inspection["record_count"] == 25, "unexpected provisional record count")
    adapters = {
        "01_XGBS": "xgbs", "02_IGAEBS": "igaebs", "07_SGAE": "sgae",
        "04_mRmR": "mrmr", "05_MI Top-K": "mifs", "02_CARS": "cars",
        "05_DRL4BS": "drl4bs", "03_BS_Nets": "bsnets",
    }
    adapter_hashes = {
        directory: digest(ROOT.parent / directory / package / "data.py")
        for directory, package in adapters.items()
    }
    require(len(set(adapter_hashes.values())) == 1, "shared data adapters differ")

    paper_hash = digest(ROOT / "paper_BS_Nets_TGRS_2020.pdf")
    require(paper_hash == "0370ba6b741819ddd41ade260f52820978021855a6d92715a8c056740498387d",
            "paper changed")
    official_fc_hash = digest(ROOT / "official_repository/BS_Net_FC.py")
    require(official_fc_hash ==
            "b96f130b80a57756fb83f4fbc18114cca6a2d88a682e85baee5705f9d103e305",
            "official FC source changed")
    commit = subprocess.run(
        ["git", "-C", str(ROOT / "official_repository"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(ROOT / "official_repository"), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout
    require(commit == "bbe8dc258b70fbf727bd3d577ea4d516d095bd35", "wrong official commit")
    require(status == "", "official worktree is not clean")

    suites = list(ET.parse(ROOT / "outputs/tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(item.attrib.get(key, 0)) for item in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] ==
            counts["skipped"] == 0, "server tests did not all pass")
    acceptance = {
        "status": "passed",
        "tests": counts,
        "synthetic_only": True,
        "real_data_training_started": False,
        "method": result["method"],
        "epochs": result["epochs_completed"],
        "optimizer_steps": result["optimizer_steps"],
        "stop_reason": result["stop_reason"],
        "degenerate_ranking": result["degenerate_ranking"],
        "selections": result["selections"],
        "training_seconds": result["training_seconds"],
        "final_inference_seconds": result["final_inference_seconds"],
        "runtime": result["runtime"],
        "blackbody_record_count": inspection["record_count"],
        "data_adapter_sha256": adapter_hashes,
        "source_sha256": result["source_sha256"],
        "paper_sha256": paper_hash,
        "official_commit": commit,
        "official_fc_sha256": official_fc_hash,
        "model_sha256": result["model_sha256"],
    }
    (ROOT / "outputs/acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    (ROOT / "SOURCE_SHA256SUMS.txt").write_text(
        "".join(f"{checksum}  {name}\n" for name, checksum in source_hashes().items()),
        encoding="utf-8")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
