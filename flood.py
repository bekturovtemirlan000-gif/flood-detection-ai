"""CPU satellite water segmentation with optional pre/post flood differencing."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
import joblib
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def features(path, version='v1'):
    if version not in ('v1', 'v2'):
        raise ValueError('Unknown feature version: ' + version)
    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError('Expected exactly two Sentinel-1 bands: VV then VH, in dB.')
        a = src.read().astype('float32')
        valid = (src.read_masks() > 0).all(axis=0) & np.isfinite(a).all(axis=0)
        profile = src.profile.copy()
    a = np.where(valid[None], np.clip(a, -50, 10), 0)
    bands = [a[0], a[1], a[0] - a[1]]
    for size in ((3, 7) if version == 'v1' else (3, 7, 15, 31)):
        weight = uniform_filter(valid.astype('float32'), size=size)
        for band in a:
            mean = uniform_filter(band, size=size) / np.maximum(weight, 1e-6)
            second = uniform_filter(band * band, size=size) / np.maximum(weight, 1e-6)
            bands.extend([mean, np.sqrt(np.maximum(second - mean * mean, 0))])
    return np.stack(bands, axis=-1).reshape(-1, len(bands)), valid, profile


def probability(model, x):
    if len(x) == 0:
        return np.empty(0, dtype='float32')
    return np.concatenate([model.predict_proba(x[i:i + 65536])[:, 1]
                           for i in range(0, len(x), 65536)])


def infer(bundle, path):
    x, valid, profile = features(path, bundle.get('feature_version', 'v1'))
    p = probability(bundle['model'], x).reshape(valid.shape)
    return p, valid, profile


def predict(model_path, image, output, before=None):
    # Load only trusted model files: joblib uses Python pickle.
    bundle = joblib.load(model_path)
    p, valid, profile = infer(bundle, image)
    water = p >= bundle['threshold']
    if before:
        prior, prior_valid, prior_profile = infer(bundle, before)
        if (any(profile[k] != prior_profile[k] for k in ('crs', 'width', 'height'))
                or not profile['transform'].almost_equals(prior_profile['transform'], precision=1e-10)):
            raise ValueError('Before and after rasters must have identical CRS, grid and extent.')
        valid &= prior_valid
        water &= prior < bundle['threshold']
    mask = np.where(valid, water, 255).astype('uint8')
    profile.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(mask, 1)
        dst.update_tags(classes='0=other,1=candidate_flood' if before else '0=land,1=water',
                        threshold=str(bundle['threshold']))
    print(json.dumps({'output': str(output), 'positive_pixels': int((mask == 1).sum()),
                      'valid_pixels': int(valid.sum()), 'mode': 'candidate_flood' if before else 'surface_water'}))
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='Two-band Sentinel-1 VV/VH dB GeoTIFF')
    parser.add_argument('--model', default='models/water_model.joblib')
    parser.add_argument('--output', default='outputs/water_mask.tif')
    parser.add_argument('--before', help='Optional aligned pre-flood Sentinel-1 GeoTIFF')
    args = parser.parse_args()
    predict(args.model, args.image, args.output, args.before)
