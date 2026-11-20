"""Detecta y alinea todas las caras de LFW (13.233 imagenes) y las cachea en un .npz.

Uso:
    python scripts/prepare_lfw.py --lfw ~/.cache/bigbrother/lfw_home/lfw --out ~/.cache/bigbrother/lfw_crops.npz

LFW se descarga con:  python -c "from sklearn.datasets import fetch_lfw_people as f; f(data_home='~/.cache/bigbrother', funneled=False)"
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_det = None


def _init():
    global _det
    import cv2
    cv2.setNumThreads(1)
    from bigbrother.detector import FaceDetector
    _det = FaceDetector(score_min=0.5)


def _process(path):
    import cv2
    from bigbrother.detector import check_quality
    from bigbrother.errors import LowQualityFace
    from bigbrother.preprocess import align
    img = cv2.imread(path)
    faces = _det.detect_all(img)
    if not faces:
        return None, False
    h, w = img.shape[:2]
    # LFW: la cara de interes es la centrada
    c = np.array([w / 2, h / 2])
    f = min(faces, key=lambda f: np.linalg.norm(np.array(f.bbox[:2]) + np.array(f.bbox[2:]) / 2 - c))
    try:
        check_quality(f)
        ok = True
    except LowQualityFace:
        ok = False
    crop = cv2.cvtColor(align(img, f.landmarks), cv2.COLOR_BGR2GRAY)
    return crop, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default=os.path.expanduser("~/.cache/bigbrother/lfw_home/lfw"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    paths = sorted(str(p) for p in Path(a.lfw).glob("*/*.jpg"))
    names = [Path(p).parent.name for p in paths]
    with Pool(a.workers, initializer=_init) as pool:
        res = pool.map(_process, paths, chunksize=64)
    keep = [i for i, (c, _) in enumerate(res) if c is not None]
    crops = np.stack([res[i][0] for i in keep])
    print(f"{len(keep)}/{len(paths)} caras detectadas; "
          f"{sum(res[i][1] for i in keep)} pasan el control de calidad")
    np.savez_compressed(a.out, crops=crops, names=np.array([names[i] for i in keep]),
                        paths=np.array([os.path.relpath(paths[i], a.lfw) for i in keep]),
                        quality_ok=np.array([res[i][1] for i in keep]),
                        n_total=len(paths))
    print("guardado", a.out)


if __name__ == "__main__":
    main()
