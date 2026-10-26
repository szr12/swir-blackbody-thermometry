from __future__ import annotations

import argparse
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

from plot_matched_maps import LABELS, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """将真实采集BMP与已保存的温度图配对，不重新推理或改变预测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--previous-results', type=Path, required=True)
    parser.add_argument('--source-previews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    summary = json.loads((args.results / 'predictions/summary.json').read_text(encoding='utf-8'))
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    cmap = mpl.colormaps['inferno'].copy()
    cmap.set_bad('#f3f4f6')
    cmap.set_under('#aeb6bf')
    cmap.set_over(cmap(1.0))
    norm = colors.Normalize(0, 1000, clip=False)
    scenes, integrity = [], []
    with zipfile.ZipFile(args.source_previews) as archive:
        for row in summary['scenes']:
            name = row['scene']
            old_dir = args.previous_results / 'predictions' / name
            previous_summary = json.loads((old_dir / 'summary.json').read_text(encoding='utf-8'))
            member = str(Path(previous_summary['source_data']).with_suffix('.bmp'))
            original_bytes = archive.read(member)
            with Image.open(io.BytesIO(original_bytes)) as source:
                original = np.asarray(source.convert('RGB')).copy()
            # BMP已经采用采集软件的显示方向，不能再转置；裁剪坐标同步交换。
            if original.shape[:2] != (row['shape'][1], row['shape'][0]):
                raise ValueError(f'原图尺寸与高光谱扫描不同：{name}')
            y0, y1, x0, x1 = row['crop_y0_y1_x0_x1']
            original_crop = original[x0:x1, y0:y1]
            current_path = args.results / 'predictions' / name / 'predictions.npz'
            previous_path = old_dir / 'predictions.npz'
            before = {'current': sha256(current_path), 'previous': sha256(previous_path)}
            with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
                mask = current['foreground_mask'][y0:y1, x0:x1].T
                mask = mask & (current['flags'][y0:y1, x0:x1].T != 4)
                temperature = current['temperature_c'][y0:y1, x0:x1].T
                old_temperature = previous['temperature_c'][y0:y1, x0:x1].T
                mask = mask & np.isfinite(temperature)
                masked = np.ma.array(temperature, mask=~mask)
                old_masked = np.ma.array(old_temperature, mask=~mask)
                np.testing.assert_array_equal(current['foreground_mask'], previous['foreground_mask'])
            if original_crop.shape[:2] != masked.shape:
                raise ValueError('原图裁剪和热图无法逐坐标对应')
            after = {'current': sha256(current_path), 'previous': sha256(previous_path)}
            if before != after:
                raise ValueError('预测矩阵在绘图时发生变化')
            scenes.append((name, original_crop, old_masked, masked))
            integrity.append({'scene': name, 'source_preview_member': member,
                'source_preview_sha256': hashlib.sha256(original_bytes).hexdigest(),
                'source_preview_shape': list(original.shape), 'original_crop_shape': list(original_crop.shape),
                'source_native_crop_y0_y1_x0_x1': row['crop_y0_y1_x0_x1'],
                'bmp_crop_rows_cols': [x0, x1, y0, y1], 'source_preview_type': '采集软件SWIR预览，不是可见光照片',
                'original_contrast_gamma_changed': False, 'original_image_resampled': False,
                'prediction_npz_sha256': before, 'prediction_files_unchanged': before == after,
                'temperature_color_limits_c': [0, 1000], 'temperature_smoothing': False,
                'temperature_grid_step_px': row['spatial_grid_step_px'], 'temperature_clipping': False})
    if len(scenes) != 10:
        raise ValueError('必须完整生成10个场景')
    contract = {'question': '采集原始SWIR预览与同位置温度预测如何对应？', 'archetype': 'image plate + quant',
        'backend': 'python/matplotlib', 'scene_count': 10, 'new_training_runs': 0, 'new_inference_runs': 0,
        'color_limits_c': [0, 1000], 'preview_source_archive_sha256': sha256(args.source_previews),
        'preview_contrast_changed': False, 'predictions_changed': False, 'same_crop_and_orientation': True,
        'range_boundary_annotations': False, 'temperature_accuracy_claimed': False,
        'exports': ['PNG 300dpi', 'PDF editable text', 'SVG editable text']}
    write_json(figures / 'figure_contract.json', contract)
    write_json(figures / 'image_integrity.json', {'scenes': integrity})

    def style(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def export(fig, name, axes, ids, rows=None):
        """先检查最终面板，再保存科研图片的三种格式。"""
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=ids, row_groups=rows,
            json_out=figures / f'{name}.alignment.json', overlay_svg=figures / f'{name}.alignment.svg')
        fig.savefig(figures / f'{name}.png', dpi=300, facecolor='white')
        fig.savefig(figures / f'{name}.pdf', dpi=300, facecolor='white')
        fig.savefig(figures / f'{name}.svg', dpi=300, facecolor='white')
        plt.close(fig)

    max_height = max(value[3].shape[0] for value in scenes)
    max_width = max(value[3].shape[1] for value in scenes)
    fig = plt.figure(figsize=(14, 11.4))
    grid = fig.add_gridspec(5, 4, left=.035, right=.965, bottom=.14, top=.89, wspace=.12, hspace=.40)
    axes, ids = [], []
    for i, (name, original, _, temperature) in enumerate(scenes):
        dy, dx = (max_height - temperature.shape[0]) // 2, (max_width - temperature.shape[1]) // 2
        original_canvas = np.full((max_height, max_width, 3), [243, 244, 246], dtype=np.uint8)
        original_canvas[dy:dy + original.shape[0], dx:dx + original.shape[1]] = original
        temperature_canvas = np.ma.masked_all((max_height, max_width), dtype=np.float64)
        temperature_canvas[dy:dy + temperature.shape[0], dx:dx + temperature.shape[1]] = temperature
        row, start = i // 2, (i % 2) * 2
        original_ax, temperature_ax = fig.add_subplot(grid[row, start]), fig.add_subplot(grid[row, start + 1])
        original_ax.imshow(original_canvas, interpolation='nearest', aspect='equal')
        image = temperature_ax.imshow(temperature_canvas, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        original_ax.set_title(f'{chr(97 + 2*i)}  {LABELS[name]}｜原图', loc='left', pad=7, fontsize=10)
        temperature_ax.set_title(f'{chr(98 + 2*i)}  {LABELS[name]}｜热图', loc='left', pad=7, fontsize=10)
        for ax in (original_ax, temperature_ax):
            style(ax)
            axes.append(ax)
        ids.extend([chr(97 + 2*i), chr(98 + 2*i)])
    cax = fig.add_axes([.29, .070, .42, .014])
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend='both', ticks=[0, 200, 400, 600, 800, 1000])
    cb.set_label('模型预测温度（℃）', labelpad=5)
    fig.suptitle('60 cm实物：采集原图与温度预测', fontsize=17, y=.97)
    fig.text(.5, .925, '同一位置、同一裁剪 · 原ROI预处理 · 统一温度色标', ha='center', fontsize=11, color='#56606a')
    fig.text(.5, .018, '原图为采集软件SWIR预览；灰色：负预测值；浅灰色：无显著信号或无效像元。', ha='center', fontsize=9, color='#57606a')
    export(fig, 'overview_original_comparison', axes, ids)

    for name, original, previous, current in scenes:
        width = 10.8 if current.shape[0] > current.shape[1] * 1.5 else 13.2
        fig = plt.figure(figsize=(width, 6.2))
        grid = fig.add_gridspec(1, 3, left=.055, right=.945, bottom=.22, top=.81, wspace=.18)
        original_ax, previous_ax, current_ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
        original_ax.imshow(original, interpolation='nearest', aspect='equal')
        previous_ax.imshow(previous, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        image = current_ax.imshow(current, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        for ax, title in zip([original_ax, previous_ax, current_ax], ['a  采集原图（SWIR预览）', 'b  原单像元方案', 'c  原ROI预处理']):
            style(ax)
            ax.set_title(title, loc='left', pad=10)
        cax = fig.add_axes([.40, .13, .42, .022])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend='both', ticks=[0, 200, 400, 600, 800, 1000])
        cb.set_label('温度（℃）', labelpad=5)
        fig.suptitle(f'{LABELS[name]}｜60 cm', fontsize=16, y=.955)
        fig.text(.5, .875, '原始BMP与同位置预测对照 · 同一空间裁剪', ha='center', fontsize=10, color='#56606a')
        fig.text(.5, .025, '原图未调整对比度；预测数值保持不变。灰色：负值；浅灰色：无信号或无效。', ha='center', fontsize=9, color='#57606a')
        export(fig, f'{name}_original_comparison', [original_ax, previous_ax, current_ax], ['a', 'b', 'c'])
    write_json(output / 'verification.json', {'status': 'PASS', 'scene_count': len(scenes),
        'prediction_files_unchanged': True, 'source_preview_crop_shapes_match': True,
        'source_preview_contrast_changed': False, 'inference_runs': 0, 'training_runs': 0})
    print(json.dumps({'figure_count': 11, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
