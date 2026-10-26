from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
from scipy import ndimage


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def describe(values):
    """描述整个区域的预测分布，标准差不是均温测量不确定度。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('统计区域为空或包含未处理的非有限值')
    return {'n_pixels': int(values.size), 'mean_c': float(values.mean()),
        'median_c': float(np.median(values)), 'spatial_std_c_ddof0': float(values.std(ddof=0)),
        'min_c': float(values.min()), 'p05_c': float(np.quantile(values, .05)),
        'p95_c': float(np.quantile(values, .95)), 'max_c': float(values.max()),
        'negative_count': int((values < 0).sum()), 'above_1000_count': int((values > 1000).sum()),
        'negative_fraction': float(np.mean(values < 0)), 'above_1000_fraction': float(np.mean(values > 1000))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    current_path, previous_path = Path(cfg['current_predictions']), Path(cfg['previous_predictions'])
    hashes = {'current': sha256(current_path), 'previous': sha256(previous_path)}
    with zipfile.ZipFile(cfg['preview_archive']) as archive:
        original_bytes = archive.read(cfg['preview_member'])
        original = np.asarray(Image.open(io.BytesIO(original_bytes)).convert('RGB'))

    # 选区仅依赖原始预览和事先记录的坐标、形态参数。
    bright = original.mean(axis=2) >= cfg['selection']['rgb_mean_threshold']
    labels, _ = ndimage.label(bright, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    core = labels == np.argmax(sizes)
    core = ndimage.binary_fill_holes(ndimage.binary_closing(core, iterations=cfg['selection']['closing_iterations']))
    candidate = core & (ndimage.distance_transform_edt(core) <= cfg['selection']['inner_band_width_px'])
    x0, x1, y0, y1 = cfg['selection']['display_bounds_x0_x1_y0_y1']
    yy, xx = np.indices(candidate.shape)
    candidate &= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
        matched = current['temperature_c'].T.copy()
        spatial = current['spatial_only_temperature_c'].T.copy()
        old = previous['temperature_c'].T.copy()
        foreground = current['foreground_mask'].T.copy()
        flags = current['flags'].T.copy()
        gy, gx = current['grid_y'], current['grid_x']
        grid = current['grid_temperature_c']
        np.testing.assert_array_equal(current['foreground_mask'], previous['foreground_mask'])
    if candidate.shape != matched.shape:
        raise ValueError('原图与预测矩阵的显示方向不一致')
    shared = candidate & np.isfinite(matched) & np.isfinite(spatial) & np.isfinite(old)
    signal = shared & foreground & (flags != 4)
    methods = {'original_pixel': old, 'roi_with_previous_background': spatial, 'matched_original_preprocessing': matched}
    rows = []
    for scope, mask in [('candidate_available_predictions', shared), ('candidate_signal_subset', signal)]:
        for method, temperature in methods.items():
            rows.append({'scope': scope, 'method': method, **describe(temperature[mask])})
    main_result = next(row for row in rows if row['scope'] == 'candidate_available_predictions' and row['method'] == 'matched_original_preprocessing')
    signal_result = next(row for row in rows if row['scope'] == 'candidate_signal_subset' and row['method'] == 'matched_original_preprocessing')

    # 显示像元来自最近ROI网格；按网格覆盖次数复核算术均值，避免伪造独立样本量。
    y, x = np.where(shared)
    iy = np.searchsorted((gy[:-1] + gy[1:]) / 2.0, x, side='left')
    ix = np.searchsorted((gx[:-1] + gx[1:]) / 2.0, y, side='left')
    ids, counts = np.unique(iy * len(gx) + ix, return_counts=True)
    grid_values = grid.ravel()[ids]
    weighted_check = float(np.average(grid_values, weights=counts))
    if not np.isclose(weighted_check, main_result['mean_c'], rtol=0, atol=1e-9):
        raise ValueError('像元均值与网格覆盖加权均值不同')
    write_json(output / 'region_definition.json', {**cfg, 'preview_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'prediction_npz_sha256': hashes, 'temperature_used_for_region_selection': False,
        'candidate_pixels': int(candidate.sum()), 'available_prediction_pixels': int(shared.sum()),
        'unavailable_prediction_pixels': int((candidate & ~shared).sum()), 'signal_subset_pixels': int(signal.sum()),
        'available_coverage_fraction': float(shared.sum() / candidate.sum()),
        'signal_coverage_fraction': float(signal.sum() / candidate.sum()),
        'distinct_evaluated_grid_cells': int(len(ids)), 'overlapping_rois_are_independent_replicates': False,
        'primary_mean_definition': '候选带内已有有限预测的像元等权算术均值；缺失不补零、不插值补齐',
        'quality': '候选外缘来自预览高亮包络，不能确认其为真实外焰；弱信号结果另列口径',
        'negative_or_above_1000_predictions_removed': False, 'temperature_color_limits_c': [0, 1000]})
    write_json(output / 'temperature_statistics.json', {'scene': cfg['scene'], 'rows': rows,
        'reference_temperature_used': False, 'new_training_runs': 0, 'new_inference_runs': 0})
    with (output / 'temperature_statistics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'selected_pixel_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['display_x', 'display_y', 'matched_temperature_c', 'original_pixel_temperature_c',
                         'spatial_only_temperature_c', 'passes_signal_mask', 'native_grid_row', 'native_grid_col'])
        for number, (pixel_y, pixel_x) in enumerate(zip(y, x)):
            writer.writerow([pixel_x, pixel_y, matched[pixel_y, pixel_x], old[pixel_y, pixel_x],
                             spatial[pixel_y, pixel_x], int(signal[pixel_y, pixel_x]), iy[number], ix[number]])
    np.savez_compressed(output / 'region_masks.npz', candidate_mask_native=candidate.T,
                        statistics_mask_native=shared.T, signal_subset_mask_native=signal.T)
    Image.fromarray((candidate * 255).astype(np.uint8)).save(output / 'candidate_mask.png')
    Image.fromarray((signal * 255).astype(np.uint8)).save(output / 'signal_subset_mask.png')

    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    # 橙色标记完整候选带；青色标记原有信号掩膜内参与统计的子集。
    display_x0, display_x1, display_y0, display_y1 = cfg['display_crop_x0_x1_y0_y1']
    crop = np.s_[display_y0:display_y1, display_x0:display_x1]
    original_crop = original[crop].copy()
    candidate_crop, shared_crop, signal_crop = candidate[crop], shared[crop], signal[crop]
    original_overlay = original_crop.copy()
    original_overlay[candidate_crop] = (original_crop[candidate_crop] * .55 + np.array([255, 110, 0]) * .45).astype(np.uint8)
    original_overlay[signal_crop] = (original_crop[signal_crop] * .45 + np.array([0, 185, 220]) * .55).astype(np.uint8)
    fig = plt.figure(figsize=(12, 7.4))
    layout = fig.add_gridspec(1, 2, left=.055, right=.945, bottom=.32, top=.84, wspace=.12)
    original_ax, temperature_ax = fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])
    original_ax.imshow(original_overlay, interpolation='nearest', aspect='equal')
    cmap = mpl.colormaps['inferno'].copy()
    cmap.set_bad('#f3f4f6')
    cmap.set_under('#aeb6bf')
    cmap.set_over(cmap(1.0))
    display_mask = (foreground[crop] | shared_crop) & np.isfinite(matched[crop])
    temperature = np.ma.array(matched[crop], mask=~display_mask)
    image = temperature_ax.imshow(temperature, cmap=cmap, norm=colors.Normalize(0, 1000, clip=False),
                                  interpolation='nearest', aspect='equal')
    temperature_ax.contour(candidate_crop, levels=[.5], colors=['#ff8c00'], linewidths=1.0)
    temperature_ax.contour(signal_crop, levels=[.5], colors=['#00b9dc'], linewidths=.8)
    original_ax.set_title('a  采集原图与候选选区', loc='left', pad=9)
    temperature_ax.set_title('b  原ROI预处理预测与同一选区', loc='left', pad=9)
    for ax in (original_ax, temperature_ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.34, .22, .32, .018])
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend='both', ticks=[0, 200, 400, 600, 800, 1000])
    cb.set_label('模型预测温度（℃）', labelpad=5)
    fig.suptitle('纸板（包装箱）｜上部候选外缘区域统计', fontsize=16, y=.965)
    fig.text(.5, .905, '60 cm · 橙色：原图候选外缘带；青色：有信号子集', ha='center', fontsize=10, color='#56606a')
    fig.text(.5, .107, f'候选带已有预测均值：{main_result["mean_c"]:.1f} ℃  ｜  中位数：{main_result["median_c"]:.1f} ℃  ｜  空间标准差：{main_result["spatial_std_c_ddof0"]:.1f} ℃', ha='center', fontsize=11)
    fig.text(.5, .065, f'有预测：{shared.sum():,}/{candidate.sum():,}像元；有信号子集：{signal.sum():,}像元，均值{signal_result["mean_c"]:.1f} ℃', ha='center', fontsize=10)
    fig.text(.5, .025, '统计保留全部有限预测值；缺失未补齐。候选外缘未经真实内外焰分区确认。', ha='center', fontsize=9, color='#57606a')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[original_ax, temperature_ax], panel_ids=['a', 'b'],
        json_out=figures / 'cardboard_region.alignment.json', overlay_svg=figures / 'cardboard_region.alignment.svg')
    fig.savefig(figures / 'cardboard_region.png', dpi=300, facecolor='white')
    fig.savefig(figures / 'cardboard_region.pdf', dpi=300, facecolor='white')
    fig.savefig(figures / 'cardboard_region.svg', dpi=300, facecolor='white')
    plt.close(fig)
    if hashes != {'current': sha256(current_path), 'previous': sha256(previous_path)}:
        raise ValueError('统计期间预测文件发生变化')
    write_json(output / 'verification.json', {'status': 'PASS', 'prediction_files_unchanged': True,
        'display_mask_transposition_explicit': True, 'grid_coverage_weighted_mean_check_c': weighted_check,
        'direct_mean_c': main_result['mean_c'], 'missing_predictions_not_imputed': True,
        'numeric_temperature_clipping': False, 'new_training_runs': 0, 'new_inference_runs': 0})
    print(json.dumps({'main': main_result, 'signal_subset': signal_result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
