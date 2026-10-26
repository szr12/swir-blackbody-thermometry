from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_frozen_maps import crop_bounds, estimate_background, parse_header, signal_mask


class PreprocessingTests(unittest.TestCase):
    def test_background_rejects_bright_border_contamination(self):
        raw = np.full((100, 80, 12), 20.0)
        raw[:4, :, :] = 900
        raw[40:60, 30:50] = 500
        background, noise, count = estimate_background(raw)
        np.testing.assert_equal(background, np.full(12, 20.0))
        self.assertGreater(count, 0)
        self.assertTrue((noise >= 1).all())

    def test_signal_filter_keeps_target_and_removes_isolated_spike(self):
        corrected = np.zeros((40, 50, 12))
        corrected[10:20, 15:30, -1] = 40
        corrected[0, 0, -1] = 100
        mask, before = signal_mask(corrected, np.ones(12))
        self.assertEqual(before, 151)
        self.assertEqual(int(mask.sum()), 150)
        self.assertFalse(mask[0, 0])

    def test_crop_retains_complete_signal(self):
        mask = np.zeros((80, 100), dtype=bool)
        mask[1:4, 92:99] = True
        self.assertEqual(crop_bounds(mask, padding=10), [0, 14, 82, 100])

    def test_header_uses_explicit_dimensions(self):
        text = ('ENVI\nsamples = 2\nlines = 3\nbands = 2\ndata type = 12\n'
                'byte order = 0\nheader offset = 0\ninterleave = bil\nwavelength = {900, 2400}\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.hdr'
            path.write_text(text, encoding='utf-8')
            fields = parse_header(path)
            self.assertEqual((fields['lines'], fields['samples'], fields['bands']), (3, 2, 2))
            np.testing.assert_equal(fields['wavelengths_nm'], [900, 2400])


if __name__ == '__main__':
    unittest.main()
