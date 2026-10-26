from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = {'A4': 'A4纸', 'baozhuangxiang': '包装箱', 'dingwan': '丁烷',
          'jiujingdeng': '酒精灯', 'jiujingkuai': '酒精块', 'lazhu': '蜡烛',
          'mianhua': '棉花', 'mutan': '木炭', 'yinhuokuai': '引火块', 'yinhuomu': '引火木'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """计算图中整个信号区域的非负预测均温，不重训、不推理、不改动原数据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.predictions / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'complete' or summary['scene_count'] != 10:
        raise ValueError('必须使用完整10场景的已完成预测')
    rows, evidence = [], []
    total_sum, total_count = 0.0, 0
    for scene in summary['scenes']:
        name = scene['scene']
        path = args.predictions / name / 'predictions.npz'
        before = sha256(path)
        with np.load(path, allow_pickle=False) as data:
            temperature = data['temperature_c']
            foreground = data['foreground_mask']
            finite_valid = foreground & np.isfinite(temperature) & (data['flags'] != 4)
            negative = finite_valid & (temperature < 0)
            selected = finite_valid & (temperature >= 0)
            values = temperature[selected]
            if len(values) == 0:
                raise ValueError(f'场景无非负可用预测：{name}')
            # 用分块累加独立复核平均值，避免错误地平均网格或按色标截断。
            block_sums = [np.sum(part, dtype=np.float64) for part in np.array_split(values, 7)]
            mean = float(np.mean(values, dtype=np.float64))
            check_mean = float(sum(block_sums) / len(values))
            if not np.isclose(mean, check_mean, atol=1e-10, rtol=0):
                raise ValueError('均值复核失败')
            count = int(selected.sum())
            negative_count = int(negative.sum())
            invalid_count = int((foreground & ~finite_valid).sum())
            if count + negative_count + invalid_count != int(foreground.sum()):
                raise ValueError('像元计数不守恒')
            rows.append({'scene': name, 'label': LABELS[name], 'distance_cm': 60,
                'foreground_pixels': int(foreground.sum()), 'retained_nonnegative_pixels': count,
                'excluded_negative_pixels': negative_count, 'excluded_invalid_pixels': invalid_count,
                'mean_temperature_c': mean, 'median_temperature_c': float(np.median(values)),
                'spatial_std_temperature_c_ddof0': float(np.std(values, ddof=0)),
                'p95_temperature_c': float(np.quantile(values, .95)), 'max_temperature_c': float(values.max()),
                'above_1000_pixels_retained': int((values > 1000).sum()),
                'negative_fraction_of_finite_foreground': float(negative_count / finite_valid.sum()),
                'sum_temperature_c': float(np.sum(values, dtype=np.float64))})
            total_sum += rows[-1]['sum_temperature_c']
            total_count += count
        if before != sha256(path):
            raise ValueError('统计期间原始预测文件改变')
        evidence.append({'scene': name, 'prediction_npz_sha256': before,
                         'prediction_file_unchanged': True, 'block_sum_mean_check_c': check_mean})
    with (output / 'whole_scene_nonnegative_means.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {'scope': '与所示原图/热图总览一致的整个有信号区域，逐场景排除负数及无效预测',
        'selection': 'foreground_mask & isfinite(temperature_c) & (flags != 4) & (temperature_c >= 0)',
        'zero_temperature_included': True, 'above_1000_predictions_retained': True,
        'temperature_clipping': False, 'new_training_runs': 0, 'new_inference_runs': 0,
        'rows': rows, 'total_retained_pixels': total_count,
        'pooled_pixel_weighted_mean_c': float(total_sum / total_count),
        'equal_scene_mean_of_means_c': float(np.mean([r['mean_temperature_c'] for r in rows])),
        'pooling_interpretation': '跨不同对象和扫描的描述性汇总，像元加权及场景等权是不同口径',
        'source_evidence': evidence}
    (output / 'whole_scene_nonnegative_means.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'means': [{'scene': r['scene'], 'mean_c': r['mean_temperature_c'],
                               'excluded_negative': r['excluded_negative_pixels'], 'n': r['retained_nonnegative_pixels']} for r in rows],
                      'pooled_mean_c': result['pooled_pixel_weighted_mean_c'],
                      'equal_scene_mean_c': result['equal_scene_mean_of_means_c']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
