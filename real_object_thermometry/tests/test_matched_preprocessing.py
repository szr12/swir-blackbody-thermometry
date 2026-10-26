from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'preprocessing_validation'))
sys.path.insert(0, str(ROOT / 'real_object_thermometry'))

from analyze_recapture_swir import Geometry
from spatial_uncertainty import process_cube
from run_matched_preprocessing import nearest_grid_indices


class MatchedPreprocessingTest(unittest.TestCase):
    def test_band_subset_preserves_original_full_cube_features(self):
        """先取冻结波段与先处理完整立方体必须产生完全相同的输入。"""
        rng = np.random.default_rng(42)
        cube = rng.integers(600, 620, size=(43, 47, 264), dtype=np.uint16)
        yy, xx = np.mgrid[:43, :47]
        cube[np.hypot(xx - 23, yy - 21) <= 4] += 800
        cube[21, 23, :] = 65000
        indices = np.array([0, 6, 20, 30, 33, 34, 38, 55, 56, 57, 59, 251])
        geometry = Geometry(23, 21, 10, 1.0, '测试')
        full = process_cube(cube, geometry, include_alternative_backgrounds=False)
        selected = process_cube(cube[..., indices], geometry, include_alternative_backgrounds=False)
        for key in ['roi_mean', 'background_median', 'roi_standard_error', 'uncertainty_annulus', 'outlier_fraction_by_band']:
            np.testing.assert_array_equal(selected[key], full[key][indices])
        actual = selected['roi_mean'] - selected['background_median']
        expected = (full['roi_mean'] - full['background_median'])[indices]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(float(selected['outlier_fraction']), 0)

    def test_constant_offset_is_removed_by_original_annulus(self):
        """同一场景整体DN偏移不应改变原始同帧背景差分结果。"""
        cube = np.full((51, 51, 12), 1000, dtype=np.uint16)
        yy, xx = np.mgrid[:51, :51]
        cube[np.hypot(xx - 25, yy - 25) <= 4] += 200
        geometry = Geometry(25, 25, 10, 1.0, '测试')
        results = [process_cube(a, geometry, include_alternative_backgrounds=False) for a in [cube, cube + 500]]
        np.testing.assert_array_equal(results[0]['roi_mean'] - results[0]['background_median'],
                                      results[1]['roi_mean'] - results[1]['background_median'])

    def test_display_grid_uses_nearest_original_coordinates(self):
        """非等宽末尾网格也按真实坐标就近显示，边界不发生拉伸错位。"""
        coordinates = np.array([0, 16, 32, 34])
        expected = np.argmin(np.abs(np.arange(35)[:, None] - coordinates), axis=1)
        np.testing.assert_array_equal(nearest_grid_indices(coordinates, 35), expected)


if __name__ == '__main__':
    unittest.main()
