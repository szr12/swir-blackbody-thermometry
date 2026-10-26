from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


LABELS = {'A4': 'A4纸', 'baozhuangxiang': '包装箱', 'dingwan': '丁烷',
          'jiujingdeng': '酒精灯', 'jiujingkuai': '酒精块', 'lazhu': '蜡烛',
          'mianhua': '棉花', 'mutan': '木炭', 'yinhuokuai': '引火块', 'yinhuomu': '引火木'}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    """仅用保存的预测画图，不修改模型、温度矩阵或空间预测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    root = args.results.resolve()
    prediction_root = root / 'predictions'
    summary = json.loads((prediction_root / 'summary.json').read_text(encoding='utf-8'))
    output = root / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    scenes = []
    maximum = 0.0
    for row in summary['scenes']:
        name = row['scene']
        with np.load(prediction_root / name / 'predictions.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ('temperature_c', 'swir_intensity_dn', 'foreground_mask', 'flags')}
        y0, y1, x0, x1 = row['crop_y0_y1_x0_x1']
        # 采集软件预览以samples为纵轴、扫描lines为横轴；只转置显示，不改预测矩阵。
        cropped = {key: value[y0:y1, x0:x1].T for key, value in arrays.items()}
        mask = cropped['foreground_mask'] & (cropped['flags'] != 4) & np.isfinite(cropped['temperature_c'])
        masked = np.ma.array(cropped['temperature_c'], mask=~mask)
        if mask.any():
            maximum = max(maximum, float(masked.max()))
        scenes.append((row, cropped, masked))
    # 用户指定统一显示范围；大于1000摄氏度仍保留原始预测，只使用最高颜色。
    color_max = 1000.0
    norm = colors.Normalize(vmin=0, vmax=color_max, clip=False)
    cmap = mpl.colormaps['inferno'].copy()
    cmap.set_bad('#f3f4f6')
    cmap.set_under('#aeb6bf')
    cmap.set_over(cmap(1.0))
    max_height = max(item[2].shape[0] for item in scenes)
    max_width = max(item[2].shape[1] for item in scenes)
    image_integrity = []
    contract = {
        'core_question': '冻结黑体模型对60cm燃烧实物产生怎样的空间预测响应？',
        'claim': '本图显示模型逐像元预测分布；不存在同步同位置参考，未建立实物测温准确性。',
        'archetype': 'image plate + quant', 'backend': 'python/matplotlib',
        'scene_count': len(scenes), 'independent_acquisitions_per_scene': 1,
        'color_limits_c': [0, color_max], 'colormap': 'inferno',
        'negative_prediction_display': '灰色，原始负数保留在温度矩阵中',
        'range_boundary_annotations': False, 'predictions_clipped': False,
        'above_1000_display': '使用最高颜色及色标上限延伸，原始温度数据保持不变',
        'temperature_smoothing': False, 'interpolation': 'nearest',
        'spatial_scale': '显示按采集软件方向转置；未校准为毫米，逐扫描行时刻未验证',
        'exports': ['PNG 300dpi', 'PDF with editable text', 'SVG with editable text'],
    }
    write_json(output / 'figure_contract.json', contract)

    def export(fig, stem, axes, ids, row_groups=None):
        """先按最终画布测量面板，再导出全部格式。"""
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=ids, row_groups=row_groups,
            json_out=output / f'{stem}.alignment.json', overlay_svg=output / f'{stem}.alignment.svg')
        fig.savefig(output / f'{stem}.png', dpi=300, facecolor='white')
        fig.savefig(output / f'{stem}.pdf', dpi=300, facecolor='white')
        fig.savefig(output / f'{stem}.svg', dpi=300, facecolor='white')
        plt.close(fig)

    def style_image_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 总览统一像素尺寸补边，避免把不同场景的形状拉伸成同一长宽比。
    fig = plt.figure(figsize=(12.0, 6.2))
    grid = fig.add_gridspec(2, 5, left=.035, right=.965, bottom=.22, top=.82, wspace=.14, hspace=.37)
    axes = []
    for i, (row, cropped, masked) in enumerate(scenes):
        ax = fig.add_subplot(grid[i // 5, i % 5])
        canvas = np.ma.masked_all((max_height, max_width), dtype=np.float64)
        dy, dx = (max_height - masked.shape[0]) // 2, (max_width - masked.shape[1]) // 2
        canvas[dy:dy + masked.shape[0], dx:dx + masked.shape[1]] = masked
        image = ax.imshow(canvas, cmap=cmap, norm=norm, origin='upper', interpolation='nearest', aspect='equal')
        ax.set_title(f'{chr(97 + i)}  {LABELS.get(row["scene"], row["scene"])}', loc='left', pad=8)
        style_image_axis(ax)
        axes.append(ax)
    cax = fig.add_axes([.28, .12, .44, .018])
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend='both', ticks=[0, 200, 400, 600, 800, 1000])
    cb.set_label('模型预测温度（℃）', labelpad=5)
    fig.suptitle('60 cm实物燃烧场景：二维温度预测', fontsize=16, y=.965)
    fig.text(.5, .895, '冻结12波段模型 · 原始逐像元输出 · 统一连续色标', ha='center', fontsize=10, color='#4c5560')
    fig.text(.5, .018, '灰色：负预测值；浅灰色：无显著信号或无效像元。', ha='center', fontsize=9, color='#57606a')
    export(fig, 'overview_60cm', axes, list('abcdefghij'))

    for row, cropped, masked in scenes:
        name = row['scene']
        fig_width = 7.2 if masked.shape[0] > masked.shape[1] * 1.5 else 10.2
        fig = plt.figure(figsize=(fig_width, 6.2))
        grid = fig.add_gridspec(1, 2, left=.065, right=.935, bottom=.22, top=.81, wspace=.22)
        observed, predicted = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        intensity = cropped['swir_intensity_dn']
        vmin, vmax = np.quantile(intensity, [.01, .995])
        vmax = max(float(vmax), float(vmin) + 1)
        observed.imshow(intensity, cmap='gray', norm=colors.PowerNorm(.5, vmin=vmin, vmax=vmax),
                        origin='upper', interpolation='nearest', aspect='equal')
        image = predicted.imshow(masked, cmap=cmap, norm=norm, origin='upper', interpolation='nearest', aspect='equal')
        observed.set_title('a  SWIR亮度（2424.55 nm）', loc='left', pad=10)
        predicted.set_title('b  模型预测温度', loc='left', pad=10)
        for ax in (observed, predicted):
            style_image_axis(ax)
        cax = fig.add_axes([.62, .13, .25, .022])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend='both', ticks=[0, 200, 400, 600, 800, 1000])
        cb.set_label('温度（℃）', labelpad=5)
        fig.suptitle(f'{LABELS.get(name, name)}｜60 cm', fontsize=16, y=.955)
        fig.text(.5, .875, '同一空间裁剪 · 冻结模型逐像元推理', ha='center', fontsize=10, color='#56606a')
        fig.text(.5, .025, '未进行温度平滑或截断；灰色为负预测值，浅灰色为无显著信号或无效像元。',
                 ha='center', fontsize=9, color='#57606a')
        export(fig, f'{name}_comparison', [observed, predicted], ['a', 'b'])
        image_integrity.append({'scene': name, 'source_data_sha256': row['source_data_sha256'],
            'crop_y0_y1_x0_x1': row['crop_y0_y1_x0_x1'], 'source_shape': row['shape'],
            'intensity_display_percentiles': [.01, .995], 'intensity_display_bounds_dn': [float(vmin), float(vmax)],
            'intensity_display_gamma': .5, 'temperature_display_bounds_c': [0, color_max],
            'temperature_data_changed': False, 'spatial_interpolation': 'nearest',
            'display_orientation': 'transpose to match acquisition software preview; no flip',
            'reference_error_metrics_available': False})
    write_json(output / 'image_integrity.json', {'scenes': image_integrity})
    with (root / 'scene_statistics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scene', 'distance_cm', 'foreground_pixels', 'prediction_median_c',
                         'prediction_p95_c', 'prediction_max_c', 'within_50_450_fraction', 'input_envelope_fraction'])
        for row, _, _ in scenes:
            q = row['prediction_quantiles_c']
            writer.writerow([row['scene'], 60, row['foreground_pixels'], q[3], q[5], q[6],
                             row['range_50_450_foreground_fraction'], row['training_envelope_foreground_fraction']])
    print(json.dumps({'figures': 11, 'color_max_c': color_max, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
