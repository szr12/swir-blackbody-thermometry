from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_header(path):
    """直接解析ENVI头，波长与尺寸不从文件名猜测。"""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    fields = dict((match.group(1).strip().lower(), match.group(2).strip())
                  for match in re.finditer(r'(?m)^([^=\n]+)=([^\n]*)', text))
    wavelength = re.search(r'wavelength\s*=\s*\{([^}]+)\}', text, flags=re.I | re.S)
    if not wavelength:
        raise ValueError('ENVI头缺少波长')
    fields['wavelengths_nm'] = np.asarray([float(value) for value in wavelength.group(1).split(',')])
    for key in ('samples', 'lines', 'bands', 'data type', 'byte order', 'header offset'):
        fields[key] = int(fields[key])
    if fields['data type'] != 12 or fields['interleave'].lower() != 'bil':
        raise ValueError('本轮只接受已核实的uint16 BIL输入')
    return fields


def estimate_background(raw, border_fraction=0.12, keep_fraction=0.6):
    """用边缘低亮度像元估计同帧背景；不使用参考温度。"""
    height, width, _ = raw.shape
    border = np.zeros((height, width), dtype=bool)
    dy, dx = max(1, round(height * border_fraction)), max(1, round(width * border_fraction))
    border[:dy] = border[-dy:] = True
    border[:, :dx] = border[:, -dx:] = True
    samples = raw[border]
    brightness = np.median(samples, axis=1)
    selected = samples[brightness <= np.quantile(brightness, keep_fraction)]
    background = np.median(selected, axis=0)
    noise = np.maximum(1.4826 * np.median(np.abs(selected - background), axis=0), 1.0)
    return background, noise, len(selected)


def signal_mask(corrected, noise, minimum_pixels=16):
    """固定信噪比规则只标记有信号区域，绝不作为温度准确性认证。"""
    snr = corrected / noise
    candidate = (snr[..., -1] >= 6.0) | ((snr >= 6.0).sum(axis=-1) >= 2)
    labels, _ = ndimage.label(candidate)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels], int(candidate.sum())


def crop_bounds(mask, padding=24):
    if not mask.any():
        return [0, mask.shape[0], 0, mask.shape[1]]
    yy, xx = np.where(mask)
    return [max(0, int(yy.min()) - padding), min(mask.shape[0], int(yy.max()) + 1 + padding),
            max(0, int(xx.min()) - padding), min(mask.shape[1], int(xx.max()) + 1 + padding)]


def main():
    """加载既有模型逐像素推理，不重新选带、训练或根据实物温度调参。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空，避免覆盖既有实验')
    output.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / cfg['runtime_dir']
    for name, expected in cfg['runtime_sha256'].items():
        if sha256(runtime / name) != expected:
            raise ValueError(f'冻结运行代码不一致：{name}')
    sys.path.insert(0, str(ROOT / 'main_experiment/src'))
    sys.path.insert(0, str(runtime))
    import torch
    from regional_colorimetry import soft_region_weights
    from traditional_utils import load_data

    torch.set_num_threads(4)
    frozen_path = ROOT / cfg['frozen_parameters_path']
    if sha256(frozen_path) != cfg['frozen_parameters_sha256']:
        raise ValueError('等权择优冻结文件指纹不一致')
    frozen = read_json(frozen_path)
    if frozen['parameters'] != cfg['parameters'] or frozen['selection_metric'] != 'unweighted_rmse':
        raise ValueError('超参数或等权择优来源不一致')
    indices = np.asarray(frozen['selected_band_indices'], dtype=int)
    waves = np.asarray(frozen['selected_wavelengths_nm'])
    if len(indices) != 12 or len(set(indices)) != 12:
        raise ValueError('必须使用冻结的12个唯一波段')
    baseline = ROOT / cfg['model_source_dir']
    bands = read_json(baseline / 'selected_bands.json')
    if bands['original_band_indices'] != indices.tolist():
        raise ValueError('模型检查点波段与冻结波段不同')
    models, model_records = [], []
    for role in ('global', 'expert_0', 'expert_1', 'expert_2', 'expert_3'):
        path = baseline / 'train_plus_validation' / f'{role}.pt'
        actual_hash = sha256(path)
        if actual_hash != cfg['checkpoint_sha256'][role]:
            raise ValueError(f'已训练权重指纹不匹配：{role}')
        model = torch.load(path, map_location='cpu', weights_only=False)
        if not model.success_ or not np.allclose(model.wavelengths_nm_, waves, atol=1e-8):
            raise ValueError('检查点未收敛或波长不匹配')
        if model.regression_feature_mode != 'helmert' or model.model_.network[0].in_features != 11:
            raise ValueError('模型必须使用11维Helmert特征')
        model.device = cfg.get('device', 'cuda')
        model.model_.to(model.device).eval()
        model.attention_.to(model.device).eval()
        models.append(model)
        model_records.append({'role': role, 'sha256': actual_hash, 'seed': model.seed,
                              'training_rmse_c': model.training_rmse_c_})
    data_path = ROOT / cfg['blackbody_data_path']
    if sha256(data_path) != cfg['blackbody_data_sha256']:
        raise ValueError('黑体输入包络来源指纹不一致')
    data = load_data(data_path, 'X_raw_features', 'X_uncertainty')
    split = read_json(baseline.parent.parent / 'split_manifest.json')
    fit = np.sort(split['train_indices'] + split['validation_indices'])
    original_positions = {int(value): pos for pos, value in enumerate(data['original_band_indices'])}
    fit_positions = [original_positions[int(value)] for value in indices]
    fit_X = data['X_raw_features'][fit][:, fit_positions]
    envelope_min, envelope_max = fit_X.min(axis=0), fit_X.max(axis=0)
    write_json(output / 'protocol.json', {**cfg, 'model_records': model_records,
        'selected_band_indices': indices.tolist(), 'selected_wavelengths_nm': waves.tolist(),
        'model_fit_samples': len(fit), 'fit_input_min': envelope_min.tolist(), 'fit_input_max': envelope_max.tolist(),
        'reference_temperatures_used': False, 'new_training_runs': 0,
        'scope': '冻结黑体模型的实物逐像素迁移，未经点位及时间配准的实物温度验证',
        'mask_definition': {'0': '无显著信号', '1': '预测范围及训练输入包络内，未证明准确',
                            '2': '预测范围内但输入超出训练包络', '3': '预测小于50或大于450摄氏度',
                            '4': '数值无效或达到uint16存储上限'},
        'adc_saturation_threshold': '未知；仅检测uint16存储上限65535，不等同完整ADC饱和检测',
        'time_interpretation': '逐行扫描时各行可能对应不同时刻，本图未验证为瞬时温度场'})
    root = Path(cfg['dataset_root'])
    headers = sorted(root.glob('*_60_*900-2500*.hdr'))
    if len(headers) != cfg['expected_scene_count']:
        raise ValueError(f'60cm场景应有10个，实际{len(headers)}')
    started = time.monotonic()
    summaries = []
    for scene_id, header in enumerate(headers):
        name = header.name.split('_60_')[0]
        write_json(output / 'progress.json', {'status': 'predicting', 'completed': scene_id,
                                              'total': len(headers), 'current_scene': name})
        fields = parse_header(header)
        if fields['bands'] != 264 or fields['samples'] != 640 or not np.allclose(fields['wavelengths_nm'][indices], waves):
            raise ValueError('实物空间尺寸或波段不匹配')
        for key, expected in cfg['acquisition_header_values'].items():
            if fields.get(key) != expected:
                raise ValueError(f'{name}采集参数不匹配：{key}')
        spe = header.with_suffix('.spe')
        raw_bytes = spe.read_bytes()
        shape = (fields['lines'], fields['bands'], fields['samples'])
        if len(raw_bytes) - fields['header offset'] != np.prod(shape) * 2:
            raise ValueError('原始文件字节数不符合ENVI头')
        dtype = '<u2' if fields['byte order'] == 0 else '>u2'
        cube = np.frombuffer(raw_bytes, dtype=dtype, offset=fields['header offset']).reshape(shape)
        raw = cube[:, indices, :].transpose(0, 2, 1).astype(np.float64)
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        del cube, raw_bytes
        background, noise, background_n = estimate_background(raw)
        corrected = raw - background
        foreground, initial_n = signal_mask(corrected, noise)
        saturated = np.any(raw >= 65535, axis=-1)
        raw_oob = ((corrected < envelope_min) | (corrected > envelope_max)).sum(axis=-1).astype(np.uint8)
        floor_n = (corrected + models[0].offsets_ < models[0].floors_).sum(axis=-1).astype(np.uint8)
        values = corrected.reshape(-1, 12)
        prediction = np.empty(len(values), dtype=np.float64)
        coarse_all = np.empty_like(prediction)
        for start in range(0, len(values), cfg['batch_size']):
            part = values[start:start + cfg['batch_size']]
            coarse = models[0].predict(part)
            expert = np.column_stack([model.predict(part) for model in models[1:]])
            weights = soft_region_weights(coarse, np.asarray(cfg['temperature_boundaries_c']), cfg['parameters']['routing_width_c'])
            prediction[start:start + len(part)] = np.sum(expert * weights, axis=1)
            coarse_all[start:start + len(part)] = coarse
        prediction = prediction.reshape(raw.shape[:2])
        coarse_all = coarse_all.reshape(raw.shape[:2])
        finite = np.isfinite(prediction)
        in_range = (prediction >= 50) & (prediction <= 450)
        flags = np.zeros(raw.shape[:2], dtype=np.uint8)
        flags[foreground] = 1
        flags[foreground & (raw_oob > 0)] = 2
        flags[foreground & ~in_range] = 3
        flags[foreground & (saturated | ~finite)] = 4
        signal_values = prediction[foreground & finite & ~saturated]
        quantiles = np.quantile(signal_values, [0, .05, .25, .5, .75, .95, 1]).tolist() if signal_values.size else None
        case_dir = output / name
        case_dir.mkdir()
        np.savez_compressed(case_dir / 'predictions.npz', temperature_c=prediction,
                            coarse_temperature_c=coarse_all.astype(np.float32),
                            swir_intensity_dn=raw[..., -1].astype(np.float32),
                            foreground_mask=foreground, flags=flags, raw_oob_count=raw_oob,
                            log_floor_count=floor_n, selected_band_indices=indices,
                            background_dn=background, background_noise_mad_dn=noise)
        summary = {'scene': name, 'distance_cm': 60, 'source_header': header.name,
            'source_data': spe.name, 'source_header_sha256': sha256(header), 'source_data_sha256': source_hash,
            'shape': list(prediction.shape), 'background_method': '外侧12%边缘中较暗60%像元逐波段中位数',
            'background_sample_count': background_n, 'background_dn': background.tolist(), 'noise_dn': noise.tolist(),
            'foreground_rule': '2424.55nm信噪比>=6或至少两个选中波段信噪比>=6；去除小于16像元连通域',
            'foreground_before_component_filter': initial_n, 'foreground_pixels': int(foreground.sum()),
            'all_pixels': int(prediction.size), 'flag_counts': dict(Counter(int(value) for value in flags.ravel())),
            'range_50_450_foreground_fraction': float((foreground & in_range & finite & ~saturated).sum() / max(1, foreground.sum())),
            'training_envelope_foreground_fraction': float((foreground & (raw_oob == 0)).sum() / max(1, foreground.sum())),
            'storage_saturated_foreground_pixels': int((foreground & saturated).sum()),
            'prediction_quantiles_c': quantiles, 'quantile_probabilities': [0, .05, .25, .5, .75, .95, 1],
            'crop_y0_y1_x0_x1': crop_bounds(foreground), 'temperature_clipping_applied': False,
            'pixel_averaging_applied': False, 'reference_error_metrics_available': False}
        write_json(case_dir / 'summary.json', summary)
        summaries.append(summary)
        print(json.dumps({'completed': scene_id + 1, 'total': len(headers), 'scene': name,
                          'foreground': summary['foreground_pixels'], 'quantiles': quantiles,
                          'in_range_fraction': summary['range_50_450_foreground_fraction']}, ensure_ascii=False), flush=True)
    write_json(output / 'summary.json', {'status': 'complete', 'scene_count': len(summaries),
        'distance_cm': 60, 'new_training_runs': 0, 'reference_error_metrics_available': False,
        'elapsed_seconds': time.monotonic() - started, 'scenes': summaries})
    write_json(output / 'progress.json', {'status': 'complete', 'completed': len(summaries), 'total': len(summaries)})


if __name__ == '__main__':
    main()
