"""Strict machine-readable acceptance report for the server project."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from hcbh.statistics import atomic_write_json


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paper = ROOT / "paper_HCBH_PatternRecognition_2022.pdf"
    demo = ROOT / "outputs" / "synthetic_demo.json"
    inspection = ROOT / "outputs" / "blackbody_format_inspection.json"
    required = [paper, demo, inspection, ROOT / "README.md", ROOT / "PROVENANCE.md"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")
    reader = PdfReader(paper)
    paper_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if "Novel hyperbolic clustering-based band hierarchy" not in paper_text:
        raise SystemExit("paper title verification failed")
    normalized_text = paper_text.replace(" ", "").replace("\n", "").lower()
    if "10.1016/j.patcog.2022.108788" not in normalized_text:
        raise SystemExit("paper DOI verification failed")
    demo_payload = json.loads(demo.read_text(encoding="utf-8"))
    inspection_payload = json.loads(inspection.read_text(encoding="utf-8"))
    if inspection_payload.get("selection_started") or inspection_payload.get("regression_started"):
        raise SystemExit("provisional-data policy was violated")
    for k in (4, 5, 6):
        selected = demo_payload["selections"][str(k)]["selected_bands"]
        if len(selected) != k or selected != sorted(set(selected)):
            raise SystemExit(f"invalid synthetic selection for K={k}")
    source_files = sorted((ROOT / "hcbh").glob("*.py"))
    source_manifest = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    (ROOT / "SOURCE_SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in source_manifest.items()),
        encoding="utf-8",
    )
    payload = {
        "status": "passed",
        "paper": {
            "path": paper.name,
            "pages": len(reader.pages),
            "sha256": sha256(paper),
            "title_verified": True,
            "doi_verified": True,
        },
        "synthetic_demo": {
            "device": demo_payload["environment"]["device"],
            "dtype": demo_payload["environment"]["dtype"],
            "selections": {k: demo_payload["selections"][k]["selected_bands"] for k in ("4", "5", "6")},
            "all_values_finite": bool(np.isfinite(np.asarray(demo_payload["origin_distances"])).all()),
        },
        "provisional_blackbody": {
            "record_count": inspection_payload["record_count"],
            "group_count": len(inspection_payload["hcbh_group_estimates"]),
            "selection_started": False,
            "regression_started": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": source_manifest,
    }
    atomic_write_json(ROOT / "outputs" / "acceptance.json", payload)
    artifact_files = [paper, demo, inspection, ROOT / "outputs" / "acceptance.json"]
    (ROOT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in artifact_files),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

