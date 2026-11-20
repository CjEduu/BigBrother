"""Entrena el filtro "parece una cara" sobre el template MLBP (regresion logistica sobre
sqrt(histogramas)). Protege los metodos *_sd y *_t, que se saltan el detector: sin este
filtro, una imagen de ruido aleatorio llegaba a identificarse como una persona inscrita.

    python scripts/train_faceness.py      # -> models/faceness.npz
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_projection import extract_all  # noqa: E402

from bigbrother import config  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz"))
LFW_DIR = CACHE.parent / "lfw_home" / "lfw"
S = config.FACE_SIZE


def jitter(crop, rng):
    """Pequenos desalineados: el recorte de otro grupo no sera identico al nuestro."""
    a = rng.uniform(-10, 10)
    sc = rng.uniform(0.9, 1.12)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), a, sc)
    M[:, 2] += rng.uniform(-7, 7, 2)
    out = cv2.warpAffine(crop, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(.5, 2))
    return out


def photometric_aug(crop, rng):
    """Caras oscuras / poco contraste / ruido: LFW tiene pocas personas de piel oscura y el
    primer filtro rechazaba fotos reales del dataset (Forest Whitaker, Samuel L. Jackson...)."""
    x = crop.astype(np.float32) / 255
    x = x ** rng.uniform(1.0, 2.5) * rng.uniform(0.25, 0.8)
    x = x * 255 + rng.normal(0, rng.uniform(0, 4), x.shape)
    if rng.random() < .5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 1.2))
    return np.clip(x, 0, 255).astype(np.uint8)


def negatives(paths, rng, n):
    out = []
    yy, xx = np.mgrid[:S, :S]
    while len(out) < n:
        k = rng.integers(12)
        if k < 6:  # recorte de fondo de una foto LFW (fuera de la cara central)
            img = cv2.imread(str(paths[rng.integers(len(paths))]), cv2.IMREAD_GRAYSCALE)
            w = int(rng.integers(50, 110))
            corners = [(0, 0), (250 - w, 0), (0, 250 - w), (250 - w, 250 - w)]
            x, y = corners[rng.integers(4)]
            x = int(np.clip(x + rng.integers(-15, 16), 0, 250 - w))
            y = int(np.clip(y + rng.integers(-15, 16), 0, 250 - w))
            out.append(cv2.resize(img[y:y + w, x:x + w], (S, S)))
        elif k == 6:
            out.append(rng.integers(0, 256, (S, S), dtype=np.uint8))
        elif k == 7:
            out.append(cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8), (0, 0), rng.uniform(1, 6)))
        elif k == 8:
            v = rng.normal(rng.uniform(40, 200), rng.uniform(1, 30), (S, S))
            out.append(np.clip(v, 0, 255).astype(np.uint8))
        elif k == 9:
            f = rng.uniform(2, 40)
            th = rng.uniform(0, np.pi)
            v = 127 + 120 * np.sin((xx * np.cos(th) + yy * np.sin(th)) / f)
            out.append(v.astype(np.uint8))
        elif k == 10:
            b = int(rng.integers(2, 32))
            out.append((((xx // b + yy // b) % 2) * 255).astype(np.uint8))
        else:  # cara completamente desordenada (bloques permutados): textura de piel, no cara
            z = np.load(CACHE)["crops"] if not hasattr(negatives, "_c") else negatives._c
            negatives._c = z
            c = z[rng.integers(len(z))]
            b = 32
            tiles = [c[i:i + b, j:j + b] for i in range(0, S, b) for j in range(0, S, b)]
            rng.shuffle(tiles)
            out.append(np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)]))
    return np.stack(out)


def main():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    z = np.load(CACHE)
    crops, names = z["crops"], z["names"]
    split = json.loads((config.MODELS_DIR / "lfw_split.json").read_text())
    train = np.array([n in set(split["train_ids"]) for n in names])
    test = ~train & ~np.isin(names, split["reserved"])
    paths = sorted(LFW_DIR.glob("*/*.jpg"))
    rp = rng.permutation(len(paths))
    p_tr, p_te = [paths[i] for i in rp[: len(paths) // 2]], [paths[i] for i in rp[len(paths) // 2:]]

    tr_idx = np.where(train)[0]
    pos_tr = np.concatenate([crops[train], np.stack([jitter(c, rng) for c in crops[train][:5000]]),
                             np.stack([photometric_aug(jitter(crops[i], rng), rng)
                                       for i in rng.choice(tr_idx, 6000, replace=False)])])
    neg_tr = negatives(p_tr, rng, 14000)
    pos_te = np.concatenate([crops[test], np.stack([jitter(c, rng) for c in crops[test][:2000]]),
                             np.stack([photometric_aug(c, rng) for c in crops[test][:2000]])])
    neg_te = negatives(p_te, np.random.default_rng(1), 6000)
    args = (config.LBP_RADII, config.GRID, config.LBP_METHOD, config.PHOTOMETRIC)
    F = {k: np.sqrt(extract_all(v, *args)) for k, v in
         {"pos_tr": pos_tr, "neg_tr": neg_tr, "pos_te": pos_te, "neg_te": neg_te}.items()}
    X = np.concatenate([F["pos_tr"], F["neg_tr"]])
    y = np.r_[np.ones(len(F["pos_tr"])), np.zeros(len(F["neg_tr"]))]
    mu, sd = X.mean(0), X.std(0) + 1e-3
    clf = LogisticRegression(C=0.05, max_iter=3000)
    clf.fit((X - mu) / sd, y)
    w = (clf.coef_[0] / sd).astype(np.float32)
    b = float(clf.intercept_[0] - (mu / sd) @ clf.coef_[0])

    def score(A):
        return A @ w + b
    sp, sn = score(F["pos_te"]), score(F["neg_te"])
    sp_aligned = score(F["pos_te"][: test.sum()])
    # umbral conservador: lo justo para que la basura no pase; preferimos dejar pasar algo de
    # basura (que luego aun tiene que superar el umbral de similitud) a rechazar caras reales
    thr = float(np.quantile(sn, 0.995))
    res = {"threshold": thr,
           "frr_caras_alineadas": float((sp_aligned < thr).mean()),
           "frr_caras_desalineadas": float((score(F["pos_te"][test.sum():test.sum() + 2000]) < thr).mean()),
           "frr_caras_oscuras_bajo_contraste": float((score(F["pos_te"][test.sum() + 2000:]) < thr).mean()),
           "far_no_caras": float((sn >= thr).mean()),
           "n_pos_test": int(len(sp)), "n_neg_test": int(len(sn))}
    np.savez_compressed(config.MODELS_DIR / "faceness.npz", w=w, b=np.float32(b), threshold=np.float32(thr))
    (config.MODELS_DIR / "faceness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
