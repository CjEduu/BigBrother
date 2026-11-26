"""Umbral de IDENTIFICACION (open-set). Con ~50 identidades, usar el umbral de verificacion
(FAR 0.1% por comparacion) da ~7% de falsas identificaciones de desconocidos. Se fija el
umbral para un FPIR del 1% con las identidades de ENTRENAMIENTO de LFW contra la galeria.

    python scripts/calibrate_identification.py   # actualiza models/calibration.json
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_projection as tp  # noqa: E402
from build_db import build  # noqa: E402

from bigbrother import config  # noqa: E402

FPIR = 0.01


def main():
    d = Path(tempfile.mkdtemp())
    bb = build(d / "db.csv", d / "keys", verbose=False)
    cache = Path(os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz"))
    z = np.load(cache)
    X = tp.features(cache, z["crops"], config.LBP_RADII, config.GRID, config.LBP_METHOD,
                    config.PHOTOMETRIC, False)
    split = json.loads((config.MODELS_DIR / "lfw_split.json").read_text(encoding="utf-8"))
    tr = np.isin(z["names"], split["train_ids"])
    te = ~tr & ~np.isin(z["names"], split["reserved"])
    M = np.stack([r for _, r in bb.db.all()])
    best_tr = (bb.projection(np.asarray(X[tr])) @ M.T).max(1)
    best_te = (bb.projection(np.asarray(X[te])) @ M.T).max(1)
    thr = float(np.quantile(best_tr, 1 - FPIR))
    cal = json.loads(config.CALIBRATION_PATH.read_text(encoding="utf-8"))
    cal.update(ident_threshold=thr, ident_target_fpir=FPIR,
               ident_fpir_train=float((best_tr >= thr).mean()),
               ident_fpir_test=float((best_te >= thr).mean()))
    config.CALIBRATION_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print({k: cal[k] for k in ("ident_threshold", "ident_fpir_train", "ident_fpir_test")})


if __name__ == "__main__":
    main()
