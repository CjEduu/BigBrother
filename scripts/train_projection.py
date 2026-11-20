"""Entrena la proyeccion discriminante sobre MLBP (sqrt -> PCA blanqueado -> LDA) con LFW y
calibra los umbrales.

- Identidades de LFW que estan en nuestro dataset (y sus gemelos: Ashley Olsen, Oliver
  Phelps) se EXCLUYEN del entrenamiento y se reservan para la evaluacion de ataques.
- El resto se divide por identidad: 70% entrenamiento / 30% test (calibracion de umbrales).

Uso:
    python scripts/prepare_lfw.py              # una vez
    python scripts/train_projection.py         # genera models/mlbp_projection.npz y calibration.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bigbrother import config  # noqa: E402
from bigbrother.dataset import dataset_ids  # noqa: E402
from bigbrother.mlbp import MLBPExtractor  # noqa: E402
from bigbrother.preprocess import photometric  # noqa: E402

TWINS = {"Ashley_Olsen", "Oliver_Phelps"}  # gemelos de identidades del dataset


def reserved_lfw_names(names):
    ids = set(dataset_ids())
    return {n for n in set(names) if n.lower() in ids} | (TWINS & set(names))


def _feat_chunk(crops, radii, grid, method, phot, flip):
    ex = MLBPExtractor(radii=radii, grid=grid, method=method)
    out = np.empty((len(crops), ex.dim), np.float32)
    for i, c in enumerate(crops):
        g = c[:, ::-1] if flip else c
        out[i] = ex.extract(photometric(np.ascontiguousarray(g), phot))
    return out


def extract_all(crops, radii, grid, method, phot, flip=False, n_jobs=-1):
    chunks = np.array_split(np.arange(len(crops)), max(1, len(crops) // 256))
    parts = Parallel(n_jobs=n_jobs)(delayed(_feat_chunk)(crops[c], radii, grid, method, phot, flip)
                                    for c in chunks)
    return np.concatenate(parts)


def features(cache, crops, radii, grid, method, phot, flip):
    key = f"r{''.join(map(str, radii))}_g{grid[0]}x{grid[1]}_{method}_{phot}{'_flip' if flip else ''}"
    f = Path(cache).parent / f"lfw_feats_{key}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    t = time.time()
    X = extract_all(crops, radii, grid, method, phot, flip)
    np.save(f, X)
    print(f"  features {key}: {X.shape} en {time.time() - t:.0f}s")
    return np.load(f, mmap_mode="r")


def train_projection(X, y, n_pca, n_lda, reg=1e-3):
    """X: templates (n, D). Devuelve mean (D,), W (D, n_lda)."""
    from sklearn.decomposition import PCA
    S = np.sqrt(X)
    mean = S.mean(0)
    pca = PCA(n_components=n_pca, whiten=True, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(S - mean)
    P = (pca.components_ / np.sqrt(pca.explained_variance_)[:, None]).T  # D x n_pca
    # LDA en el espacio blanqueado
    classes, yi = np.unique(y, return_inverse=True)
    counts = np.bincount(yi)
    keep = counts[yi] >= 2
    Z, yi = Z[keep], yi[keep]
    mu = Z.mean(0)
    Sw = np.zeros((n_pca, n_pca))
    Sb = np.zeros((n_pca, n_pca))
    for c in np.unique(yi):
        Zc = Z[yi == c]
        mc = Zc.mean(0)
        Sw += (Zc - mc).T @ (Zc - mc)
        Sb += len(Zc) * np.outer(mc - mu, mc - mu)
    Sw /= len(Z)
    Sb /= len(Z)
    from scipy.linalg import eigh
    vals, vecs = eigh(Sb, Sw + reg * np.trace(Sw) / n_pca * np.eye(n_pca))
    V = vecs[:, np.argsort(vals)[::-1][:n_lda]]
    return mean.astype(np.float32), (P @ V).astype(np.float32)


def project(X, mean, W):
    Y = (np.sqrt(X) - mean) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)


def pair_scores(Y, y, rng, n_imp=300_000, max_per_id=10):
    """Similitudes de los pares genuinos (max_per_id imagenes por identidad, para que G.W.
    Bush con 530 fotos no domine) y de n_imp pares impostores aleatorios."""
    yi = np.unique(y, return_inverse=True)[1]
    gen = []
    for c in np.unique(yi):
        idx = np.where(yi == c)[0]
        if len(idx) > max_per_id:
            idx = np.random.default_rng(int(c)).choice(idx, max_per_id, replace=False)
        if len(idx) > 1:
            S = Y[idx] @ Y[idx].T
            gen.append(S[np.triu_indices(len(idx), 1)])
    gen = np.concatenate(gen)
    a, b = rng.integers(0, len(Y), (2, n_imp))
    m = yi[a] != yi[b]
    imp = np.einsum("ij,ij->i", Y[a[m]], Y[b[m]])
    return gen, imp


def metrics(gen, imp, higher_better=True):
    if not higher_better:
        gen, imp = -gen, -imp
    ths = np.unique(np.quantile(imp, np.linspace(0, 1, 4001)))
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    i = np.argmin(np.abs(far - frr))
    out = {"eer": float((far[i] + frr[i]) / 2)}
    for target in (1e-2, 1e-3):
        t = np.quantile(imp, 1 - target)
        out[f"tar@far{target:g}"] = float((gen >= t).mean())
        out[f"thr@far{target:g}"] = float(t if higher_better else -t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz"))
    ap.add_argument("--radii", default=",".join(map(str, config.LBP_RADII)))
    ap.add_argument("--grid", type=int, default=config.GRID[0])
    ap.add_argument("--method", default=config.LBP_METHOD)
    ap.add_argument("--phot", default=config.PHOTOMETRIC)
    ap.add_argument("--pca", type=int, default=1000)
    ap.add_argument("--lda", type=int, default=300)
    ap.add_argument("--no-flip", action="store_true", help="no anadir espejos al entrenamiento")
    ap.add_argument("--baseline", action="store_true", help="evaluar tambien chi2 sin entrenar")
    ap.add_argument("--save", action="store_true", help="guardar modelo y calibracion en models/")
    ap.add_argument("--far", type=float, default=1e-3, help="FAR objetivo para el umbral")
    a = ap.parse_args()
    radii = tuple(int(r) for r in a.radii.split(","))
    grid = (a.grid, a.grid)

    z = np.load(a.cache)
    crops, names = z["crops"], z["names"]
    reserved = reserved_lfw_names(names)
    rng = np.random.default_rng(0)
    pool = sorted(set(names) - reserved)
    rng.shuffle(pool)
    train_ids = set(pool[: int(0.7 * len(pool))])
    tr = np.array([n in train_ids for n in names])
    te = np.array([(n not in train_ids) and (n not in reserved) for n in names])
    print(f"LFW: {len(names)} caras | train {tr.sum()} ({len(train_ids)} ids) | "
          f"test {te.sum()} | reservadas {len(reserved)} ids: {sorted(reserved)}")

    X = features(a.cache, crops, radii, grid, a.method, a.phot, False)
    Xtr, ytr = np.asarray(X[tr]), names[tr]
    if not a.no_flip:
        Xf = features(a.cache, crops, radii, grid, a.method, a.phot, True)
        Xtr = np.concatenate([Xtr, np.asarray(Xf[tr])])
        ytr = np.concatenate([ytr, ytr])
    t = time.time()
    mean, W = train_projection(Xtr, ytr, a.pca, a.lda)
    print(f"proyeccion {W.shape} entrenada en {time.time() - t:.0f}s")

    Xte, yte = np.asarray(X[te]), names[te]
    gen, imp = pair_scores(project(Xte, mean, W), yte, np.random.default_rng(1))
    m = metrics(gen, imp)
    print(f"[MLBP+LDA coseno] genuinos={len(gen)} impostores={len(imp)} "
          + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    res = {"config": vars(a), "mlbp_lda": m}
    if a.baseline:
        from bigbrother.matcher import chi2
        r = np.random.default_rng(2)
        yi = np.unique(yte, return_inverse=True)[1]
        g_pairs = [(i, j) for c in np.unique(yi) for k, i in enumerate(np.where(yi == c)[0])
                   for j in np.where(yi == c)[0][k + 1:]]
        g_pairs = [g_pairs[i] for i in r.choice(len(g_pairs), min(20000, len(g_pairs)), replace=False)]
        aa, bb = r.integers(0, len(Xte), (2, 40000))
        i_pairs = [(i, j) for i, j in zip(aa, bb) if yi[i] != yi[j]]
        g = np.array([chi2(Xte[i], Xte[j]) for i, j in g_pairs])
        im = np.array([chi2(Xte[i], Xte[j]) for i, j in i_pairs])
        mb = metrics(g, im, higher_better=False)
        print("[MLBP chi2 baseline] " + " ".join(f"{k}={v:.4f}" for k, v in mb.items()))
        res["baseline_chi2"] = mb

    if a.save:
        np.savez_compressed(config.PROJECTION_PATH, mean=mean, W=W.astype(np.float16),
                            radii=np.array(radii), grid=np.array(grid), method=np.array(a.method),
                            photometric=np.array(a.phot), size=np.array(config.FACE_SIZE))
        # Re-evaluar con W en float16 (lo que se distribuye) para calibrar exactamente
        W16 = W.astype(np.float16).astype(np.float32)
        gen, imp = pair_scores(project(Xte, mean, W16), yte, np.random.default_rng(1), n_imp=1_000_000)
        thr = float(np.quantile(imp, 1 - a.far))
        cal = {
            "threshold": thr,
            "target_far": a.far,
            "dedup_threshold": float(np.quantile(imp, 1 - a.far / 10)),
            "test_far": float((imp >= thr).mean()),
            "test_frr": float((gen < thr).mean()),
            "metrics": metrics(gen, imp),
            "n_genuine": int(len(gen)), "n_impostor": int(len(imp)),
            "train_ids": len(train_ids), "train_images": int(tr.sum()),
            "test_images": int(te.sum()),
        }
        if a.baseline:
            cal["baseline_chi2"] = res["baseline_chi2"]
        Path(config.CALIBRATION_PATH).write_text(json.dumps(cal, indent=2))
        split = {"train_ids": sorted(train_ids), "reserved": sorted(reserved)}
        (config.MODELS_DIR / "lfw_split.json").write_text(json.dumps(split))
        print("guardado:", config.PROJECTION_PATH, json.dumps(cal, indent=1))
    return res


if __name__ == "__main__":
    main()
