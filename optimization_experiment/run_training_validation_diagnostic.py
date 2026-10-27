"""复用冻结训练函数，只观察Train/Validation曲线，不读取Test进行预测。"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse(target, prediction):
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def observed_class(module):
    """只在原函数记录训练误差后插入观察回调；删除该语句后AST须完全一致。"""
    original = ast.parse(textwrap.dedent(inspect.getsource(module.FixedBandColorimetricMLP.fit)))
    transformed = copy.deepcopy(original)
    inserted = 0

    class AddObserver(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal inserted
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "append" and isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "trace_"):
                inserted += 1
                extra = ast.parse("self._diagnostic_observer(self)").body[0]
                return [node, extra]
            return self.generic_visit(node)

    transformed = AddObserver().visit(transformed)
    assert inserted == 1

    class RemoveObserver(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "_diagnostic_observer"):
                return None
            return self.generic_visit(node)

    restored = RemoveObserver().visit(copy.deepcopy(transformed))
    assert ast.dump(restored) == ast.dump(original)
    ast.fix_missing_locations(transformed)
    namespace = {}
    exec(compile(transformed, "<diagnostic_observer_only>", "exec"), module.__dict__, namespace)

    class ObservedModel(module.FixedBandColorimetricMLP):
        pass

    ObservedModel.fit = namespace["fit"]
    return ObservedModel


def self_test(module, model_class, parameters):
    """改变观察集标签不得改变训练轨迹和权重；合成数据仅核验观察机制。"""
    rng = np.random.default_rng(7)
    values = rng.uniform(100, 2000, (36, 12))
    target = rng.uniform(50, 450, 36)
    wavelengths = np.linspace(900, 2450, 12)
    settings = dict(parameters, device="cpu", hidden_units=(8, 8), max_iter=40,
                    adaptive_weight_warmup_iterations=10)
    original = module.FixedBandColorimetricMLP(seed=19, **settings).fit(values, target, wavelengths)
    for shift in [0.0, 10000.0]:
        def observe(model):
            model.trace_[-1]["validation_rmse_c"] = rmse(target[:5] + shift, model.predict(values[:5]))
        monitored = model_class(seed=19, **settings)
        monitored._diagnostic_observer = observe
        monitored.fit(values, target, wavelengths)
        for name in ["model_", "attention_"]:
            for key, tensor in getattr(original, name).state_dict().items():
                assert torch.equal(tensor, getattr(monitored, name).state_dict()[key])
        for left, right in zip(original.trace_, monitored.trace_, strict=True):
            assert all(left[key] == right[key] for key in left)
        assert original.iterations_ == monitored.iterations_
    return {"training_weights_bitwise_identical": True, "training_trace_identical": True,
            "validation_label_perturbation_ignored_by_training": True, "synthetic_check_only": True}


def main(args):
    out, payload, project = args.output.resolve(), args.payload.resolve(), args.project.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(payload / "manifest.json")
    for name, expected in manifest["runtime_sha256"].items():
        assert sha(payload / "runtime" / name) == expected, name
    sys.path.insert(0, str(project / "main_experiment/src"))
    sys.path.insert(0, str(payload / "runtime"))
    import convergent_colorimetry as module
    from traditional_utils import load_data, model_parameters
    from regional_colorimetry import regional_training_masks, soft_region_weights

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read(payload / "config.json")
    parameters = model_parameters(config)
    parameters["device"] = args.device
    model_class = observed_class(module)
    guard = self_test(module, model_class, parameters)
    write(out / "observer_verification.json", guard)
    if args.self_test:
        print(json.dumps(guard), flush=True)
        return
    source = project / config["data_path"]
    assert sha(source) == manifest["data_sha256"]
    data = load_data(source, config["feature_key"], config["uncertainty_key"])
    train = np.asarray(manifest["train_indices"], dtype=int)
    validation = np.asarray(manifest["validation_indices"], dtype=int)
    test = set(manifest["test_indices"])
    assert len(train) == 280 and len(validation) == 60 and len(test) == 61
    assert not set(train) & set(validation) and not test & (set(train) | set(validation))
    # 以下只保留训练和验证切片；Test不进入拟合、观察或预测。
    X = data[config["feature_key"]]
    y = data["y_temperature_c"]
    train_x, train_y = X[train].copy(), y[train].copy()
    val_x, val_y = X[validation].copy(), y[validation].copy()
    wavelengths = data["wavelengths_nm"].copy()
    del X, y, data
    boundaries = np.asarray(config["temperature_boundaries_c"])
    overlap = config["search_grid"]["overlap_values_c"][0]
    routing = config["search_grid"]["routing_width_values_c"][0]
    train_masks = [np.ones(280, dtype=bool), *regional_training_masks(train_y, boundaries, overlap)]
    val_masks = [np.ones(60, dtype=bool), *regional_training_masks(val_y, boundaries, overlap)]
    protocol = {"purpose": "Train-only native MLP learning-curve diagnostic",
                "manifest_sha256": sha(payload / "manifest.json"), "test_evaluated": False,
                "training_validation_merged": False, "validation_used_for_stopping": False,
                "validation_used_for_checkpoint_selection": False, "selection_rerun": False,
                "expert_validation_scope": "same predeclared overlapping temperature interval as its training scope; diagnostic only",
                "source_ast_changes": "one observation callback after existing training-trace append",
                "seeds": list(range(4242, 4247)), "n_train": 280, "n_validation": 60,
                "torch": torch.__version__, "numpy": np.__version__, "device": args.device,
                "parameters": parameters, "runtime_sha256": manifest["runtime_sha256"]}
    existing = out / "protocol.json"
    if existing.exists():
        assert read(existing) == json.loads(json.dumps(protocol))
    write(existing, protocol)
    state = {"status": "running", "pid": os.getpid(), "completed_models": 0, "total_models": 25}
    write(out / "state.json", state)
    models_table, ensemble_table, traces = [], [], []
    try:
        for item in manifest["runs"]:
            seed, selected = item["seed"], np.asarray(item["selected_positions"], dtype=int)
            assert len(selected) == len(set(selected)) == 12
            models = []
            for role, (fit_mask, val_mask) in enumerate(zip(train_masks, val_masks, strict=True)):
                label = "global" if role == 0 else f"expert_{role - 1}"
                job = out / f"seed_{seed}" / label
                job.mkdir(parents=True, exist_ok=True)
                state.update(current=f"seed_{seed}/{label}")
                write(out / "state.json", state)
                xfit, yfit = train_x[fit_mask][:, selected], train_y[fit_mask]
                xval, yval = val_x[val_mask][:, selected], val_y[val_mask]
                assert len(yval) and len(yfit) >= 10
                started = time.monotonic()
                model = model_class(seed=seed + role, **parameters)

                def observe(current):
                    current.trace_[-1]["validation_rmse_c"] = rmse(yval, current.predict(xval))

                model._diagnostic_observer = observe
                model.fit(xfit, yfit, wavelengths[selected])
                current_trace = model.trace_
                write(job / "trace.json", {"trace": current_trace})
                endpoint_train, endpoint_val = model.predict(xfit), model.predict(xval)
                best_train_index = int(np.argmin([v["training_adaptive_rmse_c"] for v in current_trace if v["iteration"] > 100]))
                eligible = [v for v in current_trace if v["iteration"] > 100]
                restored_iteration = int(eligible[best_train_index]["iteration"])
                val_best = min(current_trace, key=lambda v: v["validation_rmse_c"])
                row = {"seed": seed, "role": label, "initialization_seed": seed + role,
                       "n_fit": int(fit_mask.sum()), "n_validation": int(val_mask.sum()),
                       "converged": bool(model.success_), "iterations": int(model.iterations_),
                       "restored_iteration": restored_iteration,
                       "training_rmse_c": rmse(yfit, endpoint_train), "validation_rmse_c": rmse(yval, endpoint_val),
                       "minimum_observed_validation_rmse_c": val_best["validation_rmse_c"],
                       "minimum_observed_validation_iteration": int(val_best["iteration"]),
                       "last_iteration_validation_rmse_c": current_trace[-1]["validation_rmse_c"],
                       "selected_positions": selected.tolist(), "elapsed_seconds": time.monotonic() - started,
                       "predictor_parameter_count": sum(v.numel() for v in model.model_.parameters()),
                       "test_evaluated": False}
                write(job / "summary.json", row)
                np.savez_compressed(job / "predictions.npz", train_indices=train[fit_mask], validation_indices=validation[val_mask],
                                    train_target_c=yfit, validation_target_c=yval,
                                    train_prediction_c=endpoint_train, validation_prediction_c=endpoint_val)
                for t in current_trace:
                    traces.append({"seed": seed, "role": label, **t})
                models_table.append(row)
                models.append(model)
                state["completed_models"] += 1
                write(out / "state.json", state)
                print("DONE", seed, label, row["iterations"], row["training_rmse_c"], row["validation_rmse_c"], flush=True)
            metrics = {}
            for subset, x, target in [("training", train_x[:, selected], train_y), ("validation", val_x[:, selected], val_y)]:
                coarse = models[0].predict(x)
                expert = np.column_stack([v.predict(x) for v in models[1:]])
                prediction = np.sum(expert * soft_region_weights(coarse, boundaries, routing), axis=1)
                metrics[subset + "_rmse_c"] = rmse(target, prediction)
                np.savez_compressed(out / f"seed_{seed}" / (subset + "_ensemble.npz"),
                                    indices=train if subset == "training" else validation, target_c=target, prediction_c=prediction)
            ensemble_table.append({"seed": seed, **metrics, "archived_validation_rmse_c": item["archived_validation_rmse_c"],
                                   "difference_from_archived_validation_c": metrics["validation_rmse_c"] - item["archived_validation_rmse_c"]})
        for name, rows in [("model_summary.csv", models_table), ("ensemble_summary.csv", ensemble_table), ("learning_curves.csv", traces)]:
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {"models": models_table, "ensembles": ensemble_table, "all_converged": all(v["converged"] for v in models_table),
                   "test_evaluated": False, "scientific_fits": 25, "capacity_search_performed": False,
                   "maximum_archived_validation_difference_c": max(abs(v["difference_from_archived_validation_c"]) for v in ensemble_table)}
        write(out / "summary.json", summary)
        state.update(status="completed", current=None)
        write(out / "state.json", state)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        write(out / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--self-test", action="store_true")
    main(parser.parse_args())
