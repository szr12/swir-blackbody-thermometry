"""将四种已冻结的传统选带结果接入原生区域比色MLP。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_experiment/src"))

import numpy as np
import torch

from traditional_utils import load_data, regression_metrics, write_json
from run_component_ablations import fit_ensemble, audit_split

METHODS = ["01_SPA", "02_CARS", "04_STG", "08_Dual_sPLS"]
DATA_HASH = "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out, project = args.output_dir.resolve(), args.project_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    native = read(args.native_protocol)
    base = native["model_config"]
    source_data = project / base["data_path"]
    if sha(source_data) != DATA_HASH or native["data_sha256"] != DATA_HASH:
        raise ValueError("数据指纹与原生回归协议不符")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = load_data(source_data, base["feature_key"], base["uncertainty_key"])
    X, y, wave = data[base["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    original = np.asarray(data["original_band_indices"])
    split = read(args.split_manifest)
    train, val, test, fit = [np.asarray(split[k],dtype=int) for k in
                             ("train_indices","validation_indices","test_indices","fit_indices")]
    audit_split(y, train, val, test, False)
    np.testing.assert_array_equal(fit,np.sort(np.concatenate([train,val])))
    protocol = {"purpose":"same native expert-specific regional MLP for four previously selected external band subsets",
        "model_config":base,"data_sha256":DATA_HASH,"split_manifest_sha256":sha(args.split_manifest),
        "native_protocol_sha256":sha(args.native_protocol),"seeds":[4242,4243,4244],
        "method_source_sha256":{m:sha(args.comparison_root/m/"summary.json") for m in METHODS},
        "model_source_sha256":{p.name:sha(p) for p in [ROOT/"optimization_experiment/convergent_colorimetry.py",
            ROOT/"optimization_experiment/regional_colorimetry.py", ROOT/"optimization_experiment/run_component_ablations.py"]},
        "scope":"same previously viewed random holdout; selector outputs frozen; no test-based model choice",
        "selection_pool_caveat":"SPA, CARS and Dual-sPLS retain band 96, excluded by current Train-QC pool; selectors came from the original full-spectrum comparison",
        "failure_policy":"retain nonconvergence without reporting its Test metrics or replacing seeds"}
    if (out/"protocol.json").exists() and read(out/"protocol.json") != protocol:
        raise ValueError("协议变动，须使用新实验目录")
    write_json(out/"protocol.json",protocol)
    state = {"pid":os.getpid(),"status":"running","completed":[],"current":None,"test_evaluated":False}
    def update(value):
        state["current"],state["updated_epoch"]=value,time.time()
        write_json(out/"queue_state.json",state)
    update("start")
    try:
        for method in METHODS:
            source = args.comparison_root/method/"summary.json"
            old = read(source)
            selected_original = np.asarray(old["selector"]["selected_band_indices"],dtype=int)
            if len(set(selected_original)) != len(selected_original):
                raise ValueError(f"{method}波段重复")
            lookup = {int(number):position for position,number in enumerate(original)}
            selected = np.asarray([lookup[int(number)] for number in selected_original],dtype=int)
            np.testing.assert_allclose(wave[selected],old["selector"]["selected_wavelengths_nm"],rtol=0,atol=.005)
            if len(selected) != old["selector"]["natural_band_count"]:
                raise ValueError(f"{method}波段数不一致")
            write_json(out/method/"selection.json",{"selected_positions":selected,"selected_original_band_indices":selected_original,
                "wavelengths_nm":wave[selected],"source_sha256":sha(source),"n_bands":len(selected)})
            old_test=np.asarray(old["test_original_indices"],dtype=int)
            np.testing.assert_array_equal(test,old_test)
            np.testing.assert_array_equal(y[test],old["test_temperatures_c"])
            for seed in protocol["seeds"]:
                job=out/method/f"seed_{seed}"
                state["current"]=f"{method}/seed_{seed}"
                path=job/"summary.json"
                if path.exists():
                    row=read(path)
                else:
                    ensemble, diagnostics=fit_ensemble(X[fit][:,selected],y[fit],wave[selected],
                        base,seed+100000,job/"native_fit",update)
                    success=all(d["success"] for d in diagnostics)
                    row={"method":method,"seed":seed,"fit_seed":seed+100000,
                        "success":success,"n_fit":len(fit),"n_test":len(test),"band_count":len(selected),
                        "fit_diagnostics_success":[d["success"] for d in diagnostics]}
                    if success:
                        coarse, experts, weights=ensemble.predict_components(X[test][:,selected])
                        prediction=(experts*weights).sum(axis=1)
                        relative=100*np.mean(np.abs(prediction-y[test])/np.abs(y[test]))
                        row.update({"metrics":regression_metrics(y[test],prediction),"mard_percent":float(relative),
                            "test_evaluated":True})
                        job.mkdir(parents=True,exist_ok=True)
                        with (job/"predictions.csv").open("w",encoding="utf-8",newline="") as handle:
                            writer=csv.writer(handle);writer.writerow(["sample_index","temperature_c","prediction_c","global_c"])
                            writer.writerows(zip(test,y[test],prediction,coarse))
                    else:
                        row["test_evaluated"]=False
                    write_json(path,row)
                state["completed"].append(f"{method}/seed_{seed}")
                update("completed_cell")
                print("DONE",method,seed,row.get("metrics",{}).get("rmse"),row["success"],flush=True)
        state["status"]="complete"
        state["test_evaluated"]=any(read(p).get("test_evaluated",False) for p in out.glob("*/seed_*/summary.json"))
        update("complete")
    except BaseException:
        state["status"],state["error"]="failed",traceback.format_exc()
        update("failed")
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--native-protocol",type=Path,required=True)
    parser.add_argument("--split-manifest",type=Path,required=True)
    parser.add_argument("--comparison-root",type=Path,required=True)
    arguments=parser.parse_args()
    arguments.output_dir.mkdir(parents=True,exist_ok=True)
    with (arguments.output_dir/"queue.lock").open("a+b") as lock:
        if os.name=="posix":
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(arguments)
