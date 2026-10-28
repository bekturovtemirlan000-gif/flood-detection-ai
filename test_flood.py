"""Behavior tests for raster outputs, invalid data and change detection."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import rasterio
from rasterio.transform import from_origin
from flood import features, predict, probability
from train import scores


class SimpleModel:
    def predict_proba(self, x):
        p = (x[:, 0] < -15).astype(float)
        return np.stack([1 - p, p], axis=1)


class FloodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.joblib'
        joblib.dump({'model': SimpleModel(), 'threshold': 0.5}, self.model)

    def tearDown(self):
        self.temp.cleanup()

    def raster(self, name, a, transform=None):
        path = self.root / name
        with rasterio.open(path, 'w', driver='GTiff', width=a.shape[2], height=a.shape[1],
                           count=a.shape[0], dtype='float32', crs='EPSG:4326', nodata=-9999,
                           transform=transform or from_origin(70, 40, 0.0001, 0.0001)) as dst:
            dst.write(a.astype('float32'))
        return path

    def test_mask_nodata_and_georeferencing(self):
        a = np.full((2, 8, 8), -10.)
        a[:, :4] = -20
        a[:, 0, 0] = -9999
        image = self.raster('after.tif', a)
        out = self.root / 'nested' / 'mask.tif'
        mask = predict(self.model, image, out)
        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(int((mask == 1).sum()), 31)
        with rasterio.open(out) as src, rasterio.open(image) as original:
            self.assertEqual(src.transform, original.transform)
            self.assertEqual(src.crs, original.crs)
            self.assertEqual(src.nodata, 255)

    def test_before_removes_permanent_water(self):
        a = np.full((2, 8, 8), -20.)
        after = self.raster('after.tif', a)
        before = self.raster('before.tif', a)
        mask = predict(self.model, after, self.root / 'mask.tif', before)
        self.assertFalse(mask.any())

    def test_misaligned_before_rejected(self):
        a = np.full((2, 8, 8), -20.)
        after = self.raster('after.tif', a)
        before = self.raster('before.tif', a, from_origin(71, 40, .0001, .0001))
        with self.assertRaisesRegex(ValueError, 'identical'):
            predict(self.model, after, self.root / 'mask.tif', before)

    def test_invalid_values_produce_finite_features(self):
        a = np.full((2, 8, 8), -20.)
        a[:, 0, 0] = np.nan
        x, valid, _ = features(self.raster('nan.tif', a))
        self.assertFalse(valid[0, 0])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape, (64, 11))

    def test_wrong_band_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'two'):
            features(self.raster('rgb.tif', np.ones((3, 8, 8))))

    def test_metrics(self):
        result = scores(np.array([1, 1, 0, 0]), np.array([True, False, True, False]))
        self.assertAlmostEqual(result['water_iou'], 1 / 3)
        self.assertEqual(result['water_f1'], .5)

    def test_empty_valid_pixel_set(self):
        self.assertEqual(probability(SimpleModel(), np.empty((0, 11))).size, 0)

    def test_wider_features_preserve_baseline_columns(self):
        rng = np.random.default_rng(42)
        a = rng.normal(-15, 4, (2, 40, 40))
        a[:, 0, 0] = np.nan
        path = self.raster('context.tif', a)
        original, valid, _ = features(path, 'v1')
        wider, valid2, _ = features(path, 'v2')
        self.assertEqual(wider.shape, (1600, 19))
        np.testing.assert_array_equal(valid, valid2)
        np.testing.assert_array_equal(original, wider[:, :11])
        self.assertTrue(np.isfinite(wider).all())

    def test_unknown_feature_version_rejected(self):
        with self.assertRaisesRegex(ValueError, 'feature version'):
            features('unused.tif', 'bad')


if __name__ == '__main__':
    unittest.main()
