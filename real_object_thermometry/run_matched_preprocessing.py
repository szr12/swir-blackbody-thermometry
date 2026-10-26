from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage

from run_frozen_maps import ROOT, crop_bounds, parse_header, read_json, sha256, write_json


def normalized_sha256(path):
    """仅统一文本换行计算指纹，不修改被复用的原始代码。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_bundle(cfg):
    """校验并加载上一轮同一组模型，绝不重新拟合。"""
    runtime = ROOT / cfg['runtime_dir']
    for name, expected in cfg['runtime_sha256'].items():
        if sha256(runtime / name) != expected:
            raise ValueError(f'冻结运行模块指纹不一致：{name}')
    sys.path.insert(0, str(ROOT / 'main_experiment/src'))
    sys.path.insert(0, str(runtime))
    import torch
    from regional_colorimetry import soft_region_weights
    from traditional_utils import load_data

    torch.set_num_threads(4)
    frozen_path = ROOT / cfg['frozen_parameters_path']
    if sha256(frozen_path) != cfg['frozen_parameters_sha256']:
        raise ValueError('冻结选带配置指纹不同')
    frozen = read_json(frozen_path)
    if frozen['parameters'] != cfg['parameters'] or frozen['selection_metric'] != 'unweighted_rmse':
        raise ValueError('模型参数或等权择优来源不一致')
    indices = np.asarray(frozen['selected_band_indices'], dtype=int)
    waves = np.asarray(frozen['selected_wavelengths_nm'])
    baseline = ROOT / cfg['model_source_dir']
    if read_json(baseline / 'selected_bands.json')['original_band_indices'] != indices.tolist():
        raise ValueError('检查点波段与冻结配置不一致')
    models, records = [], []
    for role in ('global', 'expert_0', 'expert_1', 'expert_2', 'expert_3'):
        path = baseline / 'train_plus_validation' / f'{role}.pt'
        if sha256(path) != cfg['checkpoint_sha256'][role]:
            raise ValueError(f'检查点指纹不同：{role}')
        model = torch.load(path, map_location='cpu', weights_only=False)
        if not model.success_ or not np.allclose(model.wavelengths_nm_, waves, atol=1e-8):
            raise ValueError('检查点未收敛或波长不同')
        if model.regression_feature_mode != 'helmert' or model.model_.network[0].in_features != 11:
            raise ValueError('本实验必须复用12波段、11维Helmert模型')
        model.device = cfg['device']
        model.model_.to(model.device).eval()
        model.attention_.to(model.device).eval()
        models.append(model)
        records.append({'role': role, 'sha256': sha256(path), 'seed': model.seed})
    data_path = ROOT / cfg['blackbody_data_path']
    if sha256(data_path) != cfg['blackbody_data_sha256']:
        raise ValueError('黑体背景校正数据指纹不同')
    data = load_data(data_path, 'X_raw_features', 'X_uncertainty')
    split = read_json(baseline.parent.parent / 'split_manifest.json')
    fit = np.sort(split['train_indices'] + split['validation_indices'])
    positions = {int(value): pos for pos, value in enumerate(data['original_band_indices'])}
    values = data['X_raw_features'][fit][:, [positions[int(value)] for value in indices]]

    def predict(X):
        """使用同一全局模型和四专家；只处理有限光谱。"""
        X = np.asarray(X, dtype=np.float64)
        coarse = models[0].predict(X)
        expert = np.column_stack([model.predict(X) for model in models[1:]])
        weights = soft_region_weights(coarse, np.asarray(cfg['temperature_boundaries_c']), cfg['parameters']['routing_width_c'])
        return np.sum(expert * weights, axis=1), coarse

    return models, indices, waves, records, data, fit, values.min(axis=0), values.max(axis=0), predict


def nearest_grid_indices(coordinates, length):
    """显示取最近ROI网格，禁止把插值后的像素数量当作独立预测数量。"""
    midpoints = (coordinates[:-1] + coordinates[1:]) / 2.0
    return np.searchsorted(midpoints, np.arange(length), side='left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', help='仅用于独立试运行；正式运行不设置')
    args = parser.parse_args()
    cfg = read_json(args.config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空，拒绝覆盖')
    output.mkdir(parents=True, exist_ok=True)
    pp = cfg['matched_preprocessing']
    reference_root = ROOT / 'preprocessing_validation'
    for name, expected in pp['reference_normalized_sha256'].items():
        if normalized_sha256(reference_root / name) != expected:
            raise ValueError(f'原始预处理代码不同：{name}')
    sys.path.insert(0, str(reference_root))
    from analyze_recapture_swir import Geometry, open_cube, scan_records
    from spatial_uncertainty import process_cube, INNER_FRACTION, SIGMA_FRACTION, OUTLIER_SIGMA, ANNULUS_INNER, ANNULUS_OUTER

    constants = {'inner_fraction': INNER_FRACTION, 'sigma_fraction': SIGMA_FRACTION,
                 'outlier_sigma': OUTLIER_SIGMA, 'annulus_inner': ANNULUS_INNER, 'annulus_outer': ANNULUS_OUTER}
    if constants != pp['constants']:
        raise ValueError('原始预处理常数不一致')
    models, indices, waves, records, data, fit, envelope_min, envelope_max, predict = load_bundle(cfg)
    manifest_path = ROOT / pp['training_geometry_manifest']
    if sha256(manifest_path) != pp['training_geometry_manifest_sha256']:
        raise ValueError('训练ROI几何清单指纹不同')
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    radii = np.array([float(manifest[index]['radius_px']) for index in fit])
    radius = float(np.median(radii))
    if not np.allclose(radii, radius, atol=1e-8):
        raise ValueError('训练ROI半径不唯一，需明确空间尺度')

    # 用拟合样本的原始立方体重放完整264波段，验证最终float32相减也一致。
    record_map = {row.group_id: row for row in scan_records(Path(pp['blackbody_dataset_root']))}
    replay = []
    for index in fit[[0, len(fit) // 2, -1]]:
        row = manifest[int(index)]
        geometry = Geometry(float(row['center_x']), float(row['center_y']), float(row['radius_px']), 1.0, '冻结训练几何')
        cube = open_cube(record_map[str(data['group_ids'][index])])
        extracted = process_cube(cube, geometry, include_alternative_backgrounds=False)
        corrected = extracted['roi_mean'] - extracted['background_median']
        expected = data['X_raw_features'][index]
        error = float(np.max(np.abs(corrected[data['original_band_indices']] - expected)))
        if error > 1e-6:
            raise ValueError(f'原训练输入无法精确重放：sample={index},差值={error}')
        replay.append({'fit_sample_index': int(index), 'group_id': row['group_id'], 'max_absolute_dn_difference': error})
        del cube
    write_json(output / 'training_preprocessing_replay.json', {'status': 'PASS', 'samples': replay, 'test_samples_used': False})
    protocol = {**cfg, 'model_records': records, 'selected_band_indices': indices.tolist(),
        'selected_wavelengths_nm': waves.tolist(), 'roi_radius_px': radius, 'roi_inner_radius_px': radius * INNER_FRACTION,
        'original_function': 'preprocessing_validation.spatial_uncertainty.process_cube', 'new_training_runs': 0,
        'temperature_smoothing': False, 'prediction_clipping': False, 'reference_temperatures_used': False,
        'spatial_grid_step_px': pp['grid_step_px'], 'map_display': '最近网格ROI预测，非每个原始像元独立测温',
        'foreground_definition': '固定使用上一版信号掩膜作同区域对照，不参与光谱预处理或回归输入',
        'annulus_foreground_fraction': '原环形背景内落入上一版信号掩膜的比例，仅诊断，不删环形像元',
        'control': '空间ROI处理相同、使用上一版边缘背景的单因素对照；不参与参数选择'}
    write_json(output / 'protocol.json', protocol)
    headers = sorted(Path(cfg['dataset_root']).glob('*_60_*900-2500*.hdr'))
    if len(headers) != cfg['expected_scene_count']:
        raise ValueError('60cm扫描数量不同')
    if args.scene:
        headers = [p for p in headers if p.name.split('_60_')[0] == args.scene]
        if len(headers) != 1:
            raise ValueError('试运行场景不存在或不唯一')
    summaries = []
    started = time.monotonic()
    for scene_number, header in enumerate(headers):
        name = header.name.split('_60_')[0]
        fields = parse_header(header)
        if fields['bands'] != 264 or fields['samples'] != 640 or not np.allclose(fields['wavelengths_nm'][indices], waves):
            raise ValueError('实物采集波段或尺寸不同')
        for key, expected in cfg['acquisition_header_values'].items():
            if fields.get(key) != expected:
                raise ValueError(f'采集字段不同：{name}/{key}')
        spe = header.with_suffix('.spe')
        source_hash = sha256(spe)
        old_root = ROOT / pp['previous_predictions'] / name
        old_summary = read_json(old_root / 'summary.json')
        if source_hash != old_summary['source_data_sha256']:
            raise ValueError('前后对照原始数据不同')
        with np.load(old_root / 'predictions.npz') as old:
            foreground = old['foreground_mask'].copy()
            old_background = old['background_dn'].copy()
            old_temperature = old['temperature_c'].copy()
        dtype = '<u2' if fields['byte order'] == 0 else '>u2'
        cube = np.memmap(spe, dtype=dtype, mode='r', offset=fields['header offset'], shape=(fields['lines'], fields['bands'], fields['samples']))
        raw = np.asarray(cube[:, indices, :].transpose(0, 2, 1)).copy()
        shape = raw.shape[:2]
        step = pp['grid_step_px']
        gy = np.unique(np.r_[np.arange(0, shape[0], step), shape[0] - 1])
        gx = np.unique(np.r_[np.arange(0, shape[1], step), shape[1] - 1])
        support = ndimage.maximum_filter(foreground, size=2 * int(np.ceil(radius * INNER_FRACTION)) + 1)
        centers = [(iy, ix, int(y), int(x)) for iy, y in enumerate(gy) for ix, x in enumerate(gx) if support[y, x]]

        def extract(center):
            iy, ix, y, x = center
            geometry = Geometry(float(x), float(y), radius, 1.0, '训练半径的滑动圆形ROI')
            result = process_cube(raw, geometry, include_alternative_backgrounds=False)
            corrected = result['roi_mean'] - result['background_median']
            outer = ANNULUS_OUTER * radius
            y0, y1 = max(0, int(np.floor(y - outer))), min(shape[0], int(np.ceil(y + outer)) + 1)
            x0, x1 = max(0, int(np.floor(x - outer))), min(shape[1], int(np.ceil(x + outer)) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            distance = np.hypot((xx - x) / radius, (yy - y) / radius)
            annulus = (distance >= ANNULUS_INNER) & (distance <= ANNULUS_OUTER)
            occupancy = float(foreground[y0:y1, x0:x1][annulus].mean())
            return corrected, result['roi_mean'], result['background_median'], result['roi_standard_error'], result['uncertainty_annulus'], result['outlier_fraction_by_band'], occupancy

        extracted = []
        write_json(output / 'progress.json', {'status': 'extracting', 'scene': name, 'completed_scenes': scene_number, 'total_scenes': len(headers), 'roi_count': len(centers)})
        case_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=pp['workers']) as pool:
            for number, result in enumerate(pool.map(extract, centers)):
                extracted.append(result)
                if (number + 1) % 128 == 0:
                    print(json.dumps({'scene': name, 'roi_completed': number + 1, 'roi_total': len(centers), 'seconds': round(time.monotonic() - case_started, 1)}), flush=True)
        features, roi_mean, backgrounds, roi_se, difference_se, outliers, occupancy = map(np.asarray, zip(*extracted))
        temperature, coarse = predict(features)
        spatial_only, _ = predict(roi_mean - old_background.astype(np.float32))
        grid_shape = (len(gy), len(gx))
        grid_t = np.full(grid_shape, np.nan)
        grid_coarse = np.full(grid_shape, np.nan)
        grid_control = np.full(grid_shape, np.nan)
        grid_oob = np.zeros(grid_shape, dtype=np.uint8)
        grid_floor = np.zeros(grid_shape, dtype=np.uint8)
        grid_occupancy = np.full(grid_shape, np.nan, dtype=np.float32)
        iys, ixs = np.asarray(centers)[:, :2].T
        grid_t[iys, ixs], grid_coarse[iys, ixs], grid_control[iys, ixs] = temperature, coarse, spatial_only
        grid_oob[iys, ixs] = ((features < envelope_min) | (features > envelope_max)).sum(axis=1)
        grid_floor[iys, ixs] = (features + models[0].offsets_ < models[0].floors_).sum(axis=1)
        grid_occupancy[iys, ixs] = occupancy
        yi, xi = nearest_grid_indices(gy, shape[0]), nearest_grid_indices(gx, shape[1])
        expand = lambda values: values[np.ix_(yi, xi)]
        full_t, full_coarse, control = expand(grid_t), expand(grid_coarse), expand(grid_control)
        oob, floor_n = expand(grid_oob), expand(grid_floor)
        finite = np.isfinite(full_t)
        flags = np.zeros(shape, dtype=np.uint8)
        flags[foreground] = 1
        flags[foreground & (oob > 0)] = 2
        flags[foreground & ((full_t < 50) | (full_t > 450))] = 3
        flags[foreground & ~finite] = 4
        valid = foreground & finite
        quantiles = np.quantile(full_t[valid], [0, .05, .25, .5, .75, .95, 1]).tolist()
        case = output / name
        case.mkdir()
        np.savez_compressed(case / 'predictions.npz', temperature_c=full_t, coarse_temperature_c=full_coarse,
            spatial_only_temperature_c=control, swir_intensity_dn=raw[..., -1].astype(np.float32),
            foreground_mask=foreground, flags=flags, raw_oob_count=oob, log_floor_count=floor_n,
            grid_temperature_c=grid_t, grid_spatial_only_temperature_c=grid_control, grid_y=gy, grid_x=gx,
            roi_centers_iy_ix_y_x=np.asarray(centers), roi_corrected_dn=features, roi_mean_dn=roi_mean,
            roi_background_dn=backgrounds, roi_standard_error_dn=roi_se, difference_standard_error_dn=difference_se,
            outlier_fraction_by_roi_band=outliers, annulus_foreground_fraction=occupancy,
            selected_band_indices=indices, background_dn=np.median(backgrounds, axis=0))
        summary = {'scene': name, 'distance_cm': 60, 'shape': list(shape), 'source_data_sha256': source_hash,
            'source_header_sha256': sha256(header), 'foreground_pixels': int(foreground.sum()),
            'valid_foreground_pixels': int(valid.sum()), 'evaluated_roi_count': len(centers), 'grid_shape': list(grid_shape),
            'overlapping_rois_are_independent_replicates': False, 'spatial_grid_step_px': step,
            'prediction_quantiles_c': quantiles, 'quantile_probabilities': [0, .05, .25, .5, .75, .95, 1],
            'negative_fraction': float(np.mean(full_t[valid] < 0)), 'above_1000_fraction': float(np.mean(full_t[valid] > 1000)),
            'range_50_450_foreground_fraction': float(np.mean((full_t[valid] >= 50) & (full_t[valid] <= 450))),
            'training_envelope_foreground_fraction': float(np.mean(oob[valid] == 0)),
            'annulus_foreground_fraction_quantiles': np.quantile(occupancy, [0, .5, .95, 1]).tolist(),
            'outlier_fraction_mean': float(outliers.mean()), 'crop_y0_y1_x0_x1': crop_bounds(foreground),
            'previous_prediction_quantiles_c': np.quantile(old_temperature[valid], [0, .05, .25, .5, .75, .95, 1]).tolist(),
            'spatial_only_prediction_quantiles_c': np.quantile(control[valid], [0, .05, .25, .5, .75, .95, 1]).tolist(),
            'temperature_clipping_applied': False, 'reference_error_metrics_available': False,
            'pixel_averaging_applied': True, 'elapsed_seconds': time.monotonic() - case_started}
        write_json(case / 'summary.json', summary)
        summaries.append(summary)
        write_json(output / 'summary.json', {'status': 'running', 'scenes': summaries})
        print(json.dumps({'scene': name, 'complete': True, 'roi_count': len(centers), 'quantiles_c': quantiles,
                          'elapsed_seconds': round(summary['elapsed_seconds'], 1)}, ensure_ascii=False), flush=True)
        del cube, raw
    write_json(output / 'summary.json', {'status': 'complete', 'scene_count': len(summaries), 'distance_cm': 60,
        'new_training_runs': 0, 'reference_error_metrics_available': False, 'scenes': summaries,
        'elapsed_seconds': time.monotonic() - started})
    write_json(output / 'progress.json', {'status': 'complete', 'completed': len(summaries), 'total': len(summaries)})


if __name__ == '__main__':
    main()
