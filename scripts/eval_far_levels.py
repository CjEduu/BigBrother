"""Analisis exhaustivo de rendimiento y seguridad a lo largo de distintos niveles de FAR y FPIR.

Ejecucion reproducible dentro de la imagen Docker `bigbrother:eval`:
    docker run --rm --entrypoint python \\
        -v /home/cj/.cache/bigbrother:/cache:ro \\
        -v "$(pwd)/scripts:/app/scripts:ro" \\
        -v "$(pwd)/docs:/app/docs" \\
        -v "$(pwd)/data/ataques:/app/data/ataques:ro" \\
        bigbrother:eval scripts/eval_far_levels.py

Trabaja sobre una BD temporal con clave propia (no modifica data/db.csv, keys/ ni models/).
Genera:
  - docs/resultados/eval_far_levels.json
  - docs/resultados/eval_far_levels.png
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_projection as tp  # noqa: E402
from build_db import build  # noqa: E402

from bigbrother import config  # noqa: E402
from bigbrother.dataset import FOTOS_DIR  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402

VINI = "vinicius_junior"
RESERVED_MAP = {
    "Christian_Bale": "christian_bale",
    "Jackie_Chan": "jackie_chan",
    "James_Phelps": "james_phelps",
    "Keanu_Reeves": "keanu_reeves",
    "Mary-Kate_Olsen": "mary-kate_olsen",
    "Michael_Jackson": "michael_jackson",
    "Salma_Hayek": "salma_hayek",
}
TWINS_LFW = {
    "Ashley_Olsen": "mary-kate_olsen",
    "Oliver_Phelps": "james_phelps",
}
PAULA_VICTIM = "aitana_etxebarria"

FAR_LEVELS = [
    (1e-5, "0,001%"),
    (1e-4, "0,01%"),
    (5e-4, "0,05%"),
    (1e-3, "0,1% (actual)"),
    (2e-3, "0,2%"),
    (5e-3, "0,5%"),
    (1e-2, "1%"),
    (2e-2, "2%"),
    (5e-2, "5%"),
    (1e-1, "10%"),
    ("EER", "EER"),
]

FPIR_LEVELS = [
    (1e-3, "0,1%"),
    (5e-3, "0,5%"),
    (1e-2, "1% (actual)"),
    (2e-2, "2%"),
    (5e-2, "5%"),
    (1e-1, "10%"),
]

BLUFR_HDLBP_LDA_FAR = 1e-3
BLUFR_HDLBP_LDA_TAR = 0.3612


def log(*args):
    print(*args, flush=True)


def find_cache(explicit_cache=None):
    candidates = []
    if explicit_cache:
        candidates.append(Path(explicit_cache))
    candidates.extend([
        Path("/cache/lfw_crops.npz"),
        Path(os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz")),
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No se encuentra lfw_crops.npz en /cache ni en ~/.cache/bigbrother"
    )


def load_or_extract_features(cache_path, crops):
    key = (
        f"r{''.join(map(str, config.LBP_RADII))}"
        f"_g{config.GRID[0]}x{config.GRID[1]}"
        f"_{config.LBP_METHOD}_{config.PHOTOMETRIC}"
    )
    fname = f"lfw_feats_{key}.npy"
    ro_file = cache_path.parent / fname
    if ro_file.exists():
        log(f"  Cargando features cacheadas: {ro_file}")
        return np.load(ro_file, mmap_mode="r")

    rw_dir = Path(tempfile.gettempdir()) / "bb_eval_cache"
    rw_dir.mkdir(parents=True, exist_ok=True)
    rw_file = rw_dir / fname
    if rw_file.exists():
        log(f"  Cargando features cacheadas en tmp: {rw_file}")
        return np.load(rw_file, mmap_mode="r")

    t0 = time.time()
    log(f"  Extrayendo MLBP para {len(crops)} caras de LFW...")
    X = tp.extract_all(
        crops,
        config.LBP_RADII,
        config.GRID,
        config.LBP_METHOD,
        config.PHOTOMETRIC,
        flip=False,
    )
    # Intentar guardar en cache_path.parent si es escribible, si no en /tmp
    try:
        np.save(ro_file, X)
        log(f"  Features guardadas en {ro_file} ({time.time() - t0:.1f}s)")
        return np.load(ro_file, mmap_mode="r")
    except OSError:
        np.save(rw_file, X)
        log(f"  Features guardadas en {rw_file} ({time.time() - t0:.1f}s)")
        return np.load(rw_file, mmap_mode="r")


def build_test_pairs(yte):
    """Genera exactamente los mismos 4.300 pares genuinos y 980.849 pares impostores
    que train_projection.pair_scores(..., rng=default_rng(1), n_imp=1_000_000, max_per_id=10)."""
    yi = np.unique(yte, return_inverse=True)[1]
    gen_idx = []
    for c in np.unique(yi):
        idx = np.where(yi == c)[0]
        if len(idx) > 10:
            idx = np.random.default_rng(int(c)).choice(idx, 10, replace=False)
        if len(idx) > 1:
            ii, jj = np.triu_indices(len(idx), 1)
            gen_idx.append(np.column_stack([idx[ii], idx[jj]]))
    gen_idx = np.concatenate(gen_idx)
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, len(yte), (2, 1_000_000))
    m = yi[a] != yi[b]
    imp_idx = np.column_stack([a[m], b[m]])
    return gen_idx, imp_idx


def _chi2_worker(pairs, X):
    n = len(pairs)
    out = np.empty(n, dtype=np.float32)
    bs = 128
    buf1 = np.empty((bs, X.shape[1]), dtype=np.float32)
    buf2 = np.empty((bs, X.shape[1]), dtype=np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        k = e - s
        b1, b2 = buf1[:k], buf2[:k]
        p = pairs[s:e]
        np.take(X, p[:, 0], axis=0, out=b1)
        np.take(X, p[:, 1], axis=0, out=b2)
        np.add(b1, b2, out=b2)
        b1 *= 2.0
        b1 -= b2
        b1 *= b1
        b2 += 1e-10
        b1 /= b2
        out[s:e] = -b1.sum(axis=1)
    return out


def compute_chi2_scores(Xte, gen_idx, imp_idx):
    cache_file = Path(tempfile.gettempdir()) / "bb_eval_cache" / "chi2_lfw_test_scores.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        if len(z["gen"]) == len(gen_idx) and len(z["imp"]) == len(imp_idx):
            log(f"  Cargando scores chi2 cacheados: {cache_file}")
            return z["gen"], z["imp"]

    t0 = time.time()
    log(f"  Calculando baseline MLBP+chi2 sobre {len(gen_idx)} gen + {len(imp_idx)} imp...")
    n_jobs = min(16, os.cpu_count() or 4)
    gen = _chi2_worker(gen_idx, Xte)
    chunks = np.array_split(imp_idx, max(n_jobs * 4, 32))
    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_chi2_worker)(c, Xte) for c in chunks
    )
    imp = np.concatenate(parts)
    log(f"  chi2 completado en {time.time() - t0:.1f}s")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, gen=gen, imp=imp)
    return gen, imp


def dist_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def compute_global_metrics(gen, imp):
    y_true = np.concatenate([np.ones(len(gen), dtype=np.int8), np.zeros(len(imp), dtype=np.int8)])
    y_score = np.concatenate([gen, imp])
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thrs = roc_curve(y_true, y_score, drop_intermediate=False)
    fnr = 1.0 - tpr
    # Punto exacto donde FPR y FNR se cruzan por interpolacion lineal
    diff = fpr - fnr
    idx = int(np.argmin(np.abs(diff)))
    # Interpolacion fina entre idx-1 e idx si cambia de signo
    if idx > 0 and diff[idx - 1] * diff[idx] <= 0 and diff[idx] != diff[idx - 1]:
        w = abs(diff[idx - 1]) / (abs(diff[idx - 1]) + abs(diff[idx]))
        eer = float(fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1]))
        eer_thr = float(thrs[idx - 1] + w * (thrs[idx] - thrs[idx - 1]))
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_thr = float(thrs[idx])

    mu_g, sd_g = float(np.mean(gen)), float(np.std(gen))
    mu_i, sd_i = float(np.mean(imp)), float(np.std(imp))
    d_prime = float((mu_g - mu_i) / np.sqrt(0.5 * (sd_g ** 2 + sd_i ** 2)))

    return {
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "d_prime": d_prime,
        "genuine_stats": dist_stats(gen),
        "impostor_stats": dist_stats(imp),
    }


def eval_at_threshold(gen, imp, thr):
    n_gen = len(gen)
    n_imp = len(imp)
    fa = int((imp >= thr).sum())
    ta = int((gen >= thr).sum())
    fr = n_gen - ta
    far = fa / n_imp
    tar = ta / n_gen
    frr = 1.0 - tar

    # Prior 1:1 -> TP = TAR, FP = FAR, FN = FRR
    p1 = tar / (tar + far) if (tar + far) > 0 else 0.0
    f1_1 = (2.0 * tar) / (1.0 + tar + far) if (1.0 + tar + far) > 0 else 0.0

    # Prior 1:100 (1 genuino por cada 100 impostores) -> TP = TAR, FP = 100 * FAR, FN = FRR
    p100 = tar / (tar + 100.0 * far) if (tar + 100.0 * far) > 0 else 0.0
    f1_100 = (2.0 * tar) / (1.0 + tar + 100.0 * far) if (1.0 + tar + 100.0 * far) > 0 else 0.0

    return {
        "threshold": float(thr),
        "far": float(far),
        "fmr": float(far),
        "false_accepts": fa,
        "n_imp": n_imp,
        "tar": float(tar),
        "true_accepts": ta,
        "frr": float(frr),
        "fnmr": float(frr),
        "false_rejects": fr,
        "n_gen": n_gen,
        "precision_1_1": float(p1),
        "f1_1_1": float(f1_1),
        "precision_1_100": float(p100),
        "f1_1_100": float(f1_100),
    }


def sample_roc_curve(gen, imp):
    """Muestrea ~400 puntos representativos en escala log-FAR de 1e-5 a 1."""
    log_fars = np.geomspace(1e-5, 0.999, 300)
    lin_fars = np.linspace(0.01, 0.999, 100)
    fars_target = np.unique(np.sort(np.concatenate([[0.0], log_fars, lin_fars])))
    thrs = np.quantile(imp, 1.0 - fars_target)
    pts = []
    for t in np.unique(thrs)[::-1]:
        fa = float((imp >= t).mean())
        ta = float((gen >= t).mean())
        pts.append({"thr": float(t), "far": fa, "tar": ta, "frr": 1.0 - ta})
    return pts


def gallery_matrix(bb, exclude_vini=None):
    ids, mats = [], []
    k = 0
    for id_, r in bb.db.all():
        if id_ == VINI:
            if k == exclude_vini:
                k += 1
                continue
            k += 1
        ids.append(id_)
        mats.append(r)
    return np.array(ids), np.stack(mats)


def best_per_identity(ids, S):
    unique_ids = np.unique(ids)
    B = np.stack([S[:, ids == u].max(1) for u in unique_ids], axis=1)
    return unique_ids, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="Ruta a lfw_crops.npz")
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "resultados"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = find_cache(args.cache)
    log(f"[0] Inicializando entorno y BD temporal (cache: {cache_path})")
    tmp_dir = Path(tempfile.mkdtemp(prefix="bb_far_"))
    try:
        bb = build(tmp_dir / "db.csv", tmp_dir / "keys", verbose=False)
        cal = json.loads(config.CALIBRATION_PATH.read_text())

        z = np.load(cache_path)
        crops, names = z["crops"], z["names"]
        X_all = load_or_extract_features(cache_path, crops)

        split = json.loads((config.MODELS_DIR / "lfw_split.json").read_text())
        train_ids = set(split["train_ids"])
        reserved_ids = set(split["reserved"])
        train_mask = np.array([n in train_ids for n in names])
        test_mask = np.array([(n not in train_ids) and (n not in reserved_ids) for n in names])

        Xtr = np.asarray(X_all[train_mask], dtype=np.float32)
        Xte = np.asarray(X_all[test_mask], dtype=np.float32)
        yte = names[test_mask]

        log(f"  LFW split -> train: {len(Xtr)} ({len(train_ids)} ids), test: {len(Xte)}, reserved: {len(reserved_ids)} ids")

        # ====================================================================
        # EXPERIMENTO 1: Verificacion 1:1 en LFW test (MLBP+LDA vs MLBP+chi2)
        # ====================================================================
        log("[1] Evaluando verificacion 1:1 en LFW test...")
        Yte = bb.projection(Xte)
        gen_idx, imp_idx = build_test_pairs(yte)
        gen_lda = np.einsum("ij,ij->i", Yte[gen_idx[:, 0]], Yte[gen_idx[:, 1]])
        imp_lda = np.einsum("ij,ij->i", Yte[imp_idx[:, 0]], Yte[imp_idx[:, 1]])
        log(f"  Pares LFW test -> genuinos: {len(gen_lda)}, impostores: {len(imp_lda)}")

        gen_chi2, imp_chi2 = compute_chi2_scores(Xte, gen_idx, imp_idx)

        glob_lda = compute_global_metrics(gen_lda, imp_lda)
        glob_chi2 = compute_global_metrics(gen_chi2, imp_chi2)
        # En chi2, la distancia es -score
        glob_chi2["eer_distance_threshold"] = -glob_chi2["eer_threshold"]

        log(
            f"  MLBP+LDA  -> AUC={glob_lda['auc_roc']:.4f} | EER={glob_lda['eer']:.4%} "
            f"(thr={glob_lda['eer_threshold']:.4f}) | d'={glob_lda['d_prime']:.3f}"
        )
        log(
            f"  MLBP+chi2 -> AUC={glob_chi2['auc_roc']:.4f} | EER={glob_chi2['eer']:.4%} "
            f"(dist={glob_chi2['eer_distance_threshold']:.2f}) | d'={glob_chi2['d_prime']:.3f}"
        )

        verif_lda_table = []
        verif_chi2_table = []
        for target, label in FAR_LEVELS:
            if target == "EER":
                thr_lda = glob_lda["eer_threshold"]
                thr_chi2 = glob_chi2["eer_threshold"]
            elif abs(target - 1e-3) < 1e-12:
                # Usar exactamente el umbral calibrado de BigBrother (0.217241168) para el punto actual
                thr_lda = float(bb.threshold)
                thr_chi2 = float(np.quantile(imp_chi2, 1.0 - target))
            else:
                thr_lda = float(np.quantile(imp_lda, 1.0 - target))
                thr_chi2 = float(np.quantile(imp_chi2, 1.0 - target))

            row_lda = {"target_far": target, "label": label, **eval_at_threshold(gen_lda, imp_lda, thr_lda)}
            if abs((target if isinstance(target, float) else -1) - 1e-3) < 1e-12:
                row_lda["threshold_quantile_lfw"] = float(np.quantile(imp_lda, 1.0 - target))
            verif_lda_table.append(row_lda)

            row_chi2 = {"target_far": target, "label": label, **eval_at_threshold(gen_chi2, imp_chi2, thr_chi2)}
            row_chi2["chi2_distance_threshold"] = float(-thr_chi2)
            verif_chi2_table.append(row_chi2)

        # ====================================================================
        # EXPERIMENTO 2: Impacto real en la galeria de 50 identidades
        # ====================================================================
        log("[2] Evaluando impacto en la galeria real de 50 identidades...")
        ids_full, M_full = gallery_matrix(bb)
        unique_ids, B_test = best_per_identity(ids_full, Yte @ M_full.T)
        idx_vini = int(np.where(unique_ids == VINI)[0][0])
        max_test = B_test.max(axis=1)
        vini_test = B_test[:, idx_vini]

        # Entrenamiento de LFW contra la galeria (para calibrar FPIR 1:N)
        Ytr = bb.projection(Xtr)
        _, B_train = best_per_identity(ids_full, Ytr @ M_full.T)
        max_train = B_train.max(axis=1)

        # 2a. Vinicius LOO (5 fotos)
        vini_paths = sorted((FOTOS_DIR / VINI).iterdir())
        vini_templates = [bb.extraer(load_image(p)) for p in vini_paths]
        vini_projs = bb.projection(np.stack(vini_templates))
        vini_loo_scores = []
        vini_loo_ranks = []
        for i in range(len(vini_paths)):
            ids_ex, M_ex = gallery_matrix(bb, exclude_vini=i)
            u_ex, b_ex = best_per_identity(ids_ex, vini_projs[i : i + 1] @ M_ex.T)
            b_ex = b_ex[0]
            order = np.argsort(-b_ex)
            vini_idx_ex = int(np.where(u_ex == VINI)[0][0])
            vini_loo_scores.append(float(b_ex[vini_idx_ex]))
            vini_loo_ranks.append(int(np.where(order == vini_idx_ex)[0][0]) + 1)
        vini_loo_scores = np.array(vini_loo_scores)

        # 2b. Los 7 inscritos presentes en LFW (55 fotos)
        enrolled_lfw = []
        for lfw_name, id_ in RESERVED_MAP.items():
            m = names == lfw_name
            Y_res = bb.projection(np.asarray(X_all[m], dtype=np.float32))
            _, B_res = best_per_identity(ids_full, Y_res @ M_full.T)
            own_idx = int(np.where(unique_ids == id_)[0][0])
            top_idx = B_res.argmax(axis=1)
            enrolled_lfw.append({
                "id": id_,
                "lfw_name": lfw_name,
                "n": int(m.sum()),
                "own_scores": B_res[:, own_idx],
                "max_scores": B_res.max(axis=1),
                "rank1_match": (top_idx == own_idx),
                "top1_ids": [str(unique_ids[k]) for k in top_idx],
            })

        # 2c. Gemelos: Paula Etxeberria (data/ataques/gemela_paula) + LFW (Ashley Olsen, Oliver Phelps)
        paula_dir = ROOT / "data" / "ataques" / "gemela_paula"
        paula_data = {}
        for p in sorted(paula_dir.glob("*.jpg")):
            sc = bb._scores(bb._proj(bb.extraer(load_image(p))))
            paula_data[p.name] = sc

        lfw_twins_data = {}
        for lfw_twin, victim_id in TWINS_LFW.items():
            m = names == lfw_twin
            Y_tw = bb.projection(np.asarray(X_all[m], dtype=np.float32))
            _, B_tw = best_per_identity(ids_full, Y_tw @ M_full.T)
            # 1 foto cada uno en LFW
            sc = {str(unique_ids[k]): float(B_tw[0, k]) for k in range(len(unique_ids))}
            lfw_twins_data[f"{lfw_twin}->{victim_id}"] = {
                "victim": victim_id,
                "scores": sc,
            }

        # Barrido de impacto en la galeria para cada nivel de FAR de verificacion
        gallery_impact_table = []
        for row in verif_lda_table:
            thr = row["threshold"]
            label = row["label"]
            target_far = row["target_far"]

            # Open-set con umbral de verificacion
            fpir_cnt = int((max_test >= thr).sum())
            fpir_rate = float(fpir_cnt / len(max_test))

            # Sesgo Vinicius
            vini_fa = int((vini_test >= thr).sum())
            vini_far = float(vini_fa / len(vini_test))
            vini_bias_ratio = float(vini_far / row["far"]) if row["far"] > 0 else None

            # Efecto wolf (cuantas identidades distintas verifica cada probe de LFW test)
            n_verified_per_probe = (B_test >= thr).sum(axis=1)
            max_wolf_k = int(n_verified_per_probe.max())
            wolf_idx = int(np.argmax(n_verified_per_probe))
            wolf_verified_ids = [
                str(unique_ids[k])
                for k in np.argsort(-B_test[wolf_idx])
                if B_test[wolf_idx, k] >= thr
            ]
            # Top 3 wolves
            top_wolf_indices = np.argsort(-n_verified_per_probe)[:3]
            top_wolves = [
                {
                    "lfw_name": str(yte[idx]),
                    "n_identities_verified": int(n_verified_per_probe[idx]),
                    "max_score": float(B_test[idx].max()),
                }
                for idx in top_wolf_indices
                if n_verified_per_probe[idx] > 0
            ]

            # Vinicius LOO
            vini_acc = int((vini_loo_scores >= thr).sum())

            # 7 inscritos LFW
            per_person = {}
            tot_n = 0
            tot_acc = 0
            for entry in enrolled_lfw:
                acc_p = int((entry["own_scores"] >= thr).sum())
                tot_n += entry["n"]
                tot_acc += acc_p
                per_person[entry["id"]] = {
                    "n": entry["n"],
                    "accepted": acc_p,
                    "tar": float(acc_p / entry["n"]),
                }

            # Gemelos
            paula_eval = {}
            for pname, sc in paula_data.items():
                ordered = sorted(sc.items(), key=lambda kv: -kv[1])
                verified_ids = [k for k, v in ordered if v >= thr]
                paula_eval[pname] = {
                    "score_aitana": float(sc[PAULA_VICTIM]),
                    "accepted_as_aitana": bool(sc[PAULA_VICTIM] >= thr),
                    "n_identities_verified": len(verified_ids),
                    "verified_identities": verified_ids,
                }

            lfw_twins_eval = {}
            for tkey, tinfo in lfw_twins_data.items():
                sc = tinfo["scores"]
                vic = tinfo["victim"]
                ordered = sorted(sc.items(), key=lambda kv: -kv[1])
                verified_ids = [k for k, v in ordered if v >= thr]
                lfw_twins_eval[tkey] = {
                    "score_victim": float(sc[vic]),
                    "accepted_as_victim": bool(sc[vic] >= thr),
                    "n_identities_verified": len(verified_ids),
                    "verified_identities": verified_ids,
                }

            gallery_impact_table.append({
                "target_far": target_far,
                "label": label,
                "threshold": thr,
                "open_set_fpir": fpir_rate,
                "open_set_false_idents": fpir_cnt,
                "n_probes": len(max_test),
                "far_vinicius": vini_far,
                "false_accepts_vinicius": vini_fa,
                "vinicius_bias_ratio": vini_bias_ratio,
                "wolf_max_identities": max_wolf_k,
                "wolf_worst_probe": str(yte[wolf_idx]) if max_wolf_k > 0 else None,
                "wolf_worst_verified_ids": wolf_verified_ids[:10],
                "wolf_top3": top_wolves,
                "probes_verifying_ge1": int((n_verified_per_probe >= 1).sum()),
                "probes_verifying_ge2": int((n_verified_per_probe >= 2).sum()),
                "probes_verifying_ge5": int((n_verified_per_probe >= 5).sum()),
                "vinicius_loo_accepted": vini_acc,
                "vinicius_loo_total": len(vini_loo_scores),
                "vinicius_loo_tar": float(vini_acc / len(vini_loo_scores)),
                "enrolled_7_accepted": tot_acc,
                "enrolled_7_total": tot_n,
                "enrolled_7_tar": float(tot_acc / tot_n),
                "enrolled_7_by_person": per_person,
                "twins_paula": paula_eval,
                "twins_lfw": lfw_twins_eval,
            })

        # ====================================================================
        # 2d. Tabla propia de calibracion de identificacion 1:N (FPIR)
        # ====================================================================
        log("[2b] Generando tabla de calibracion de identificacion 1:N (FPIR)...")
        ident_calib_table = []
        for target_fpir, label in FPIR_LEVELS:
            if abs(target_fpir - 1e-2) < 1e-12:
                thr_id = float(bb.ident_threshold)
            else:
                thr_id = float(np.quantile(max_train, 1.0 - target_fpir))

            fpir_tr = float((max_train >= thr_id).mean())
            fa_tr = int((max_train >= thr_id).sum())
            fpir_te = float((max_test >= thr_id).mean())
            fa_te = int((max_test >= thr_id).sum())

            vini_tar_cnt = int((vini_loo_scores >= thr_id).sum())
            vini_dir_cnt = int(sum(
                (vini_loo_ranks[k] == 1) and (vini_loo_scores[k] >= thr_id)
                for k in range(len(vini_loo_scores))
            ))

            enr_tar_cnt = 0
            enr_dir_cnt = 0
            enr_false_id_cnt = 0
            enr_total = 0
            enr_by_person = {}
            for entry in enrolled_lfw:
                tar_p = int((entry["own_scores"] >= thr_id).sum())
                dir_p = int((entry["rank1_match"] & (entry["own_scores"] >= thr_id)).sum())
                fid_p = int((~entry["rank1_match"] & (entry["max_scores"] >= thr_id)).sum())
                enr_tar_cnt += tar_p
                enr_dir_cnt += dir_p
                enr_false_id_cnt += fid_p
                enr_total += entry["n"]
                enr_by_person[entry["id"]] = {
                    "n": entry["n"],
                    "dir_count": dir_p,
                    "dir": float(dir_p / entry["n"]),
                    "tar_count": tar_p,
                    "tar": float(tar_p / entry["n"]),
                    "misidentified_count": fid_p,
                }

            # Efecto en gemelos bajo umbral de identificacion
            paula_ident = {}
            for pname, sc in paula_data.items():
                best_id, best_sc = max(sc.items(), key=lambda kv: kv[1])
                paula_ident[pname] = {
                    "identificar": best_id if best_sc >= thr_id else None,
                    "best_score": float(best_sc),
                    "score_aitana": float(sc[PAULA_VICTIM]),
                }

            ident_row = {
                "target_fpir": float(target_fpir),
                "label": label,
                "threshold": thr_id,
                "fpir_train": fpir_tr,
                "false_idents_train": fa_tr,
                "n_train": len(max_train),
                "fpir_test": fpir_te,
                "false_idents_test": fa_te,
                "n_test": len(max_test),
                "vinicius_loo_dir_count": vini_dir_cnt,
                "vinicius_loo_tar_count": vini_tar_cnt,
                "vinicius_loo_total": len(vini_loo_scores),
                "vinicius_loo_dir": float(vini_dir_cnt / len(vini_loo_scores)),
                "vinicius_loo_tar": float(vini_tar_cnt / len(vini_loo_scores)),
                "enrolled_7_dir_count": enr_dir_cnt,
                "enrolled_7_tar_count": enr_tar_cnt,
                "enrolled_7_misidentified_count": enr_false_id_cnt,
                "enrolled_7_total": enr_total,
                "enrolled_7_dir": float(enr_dir_cnt / enr_total),
                "enrolled_7_tar": float(enr_tar_cnt / enr_total),
                "enrolled_7_by_person": enr_by_person,
                "twins_paula": paula_ident,
            }
            if abs(target_fpir - 1e-2) < 1e-12:
                ident_row["threshold_quantile_train"] = float(np.quantile(max_train, 1.0 - target_fpir))
            ident_calib_table.append(ident_row)

        # Detalles individuales de genuinos y gemelos
        enrolled_details = []
        for entry in enrolled_lfw:
            enrolled_details.append({
                "id": entry["id"],
                "lfw_name": entry["lfw_name"],
                "n": entry["n"],
                "own_score_mean": float(np.mean(entry["own_scores"])),
                "own_score_min": float(np.min(entry["own_scores"])),
                "own_score_max": float(np.max(entry["own_scores"])),
                "rank1_correct": int(entry["rank1_match"].sum()),
                "scores": [round(float(x), 4) for x in entry["own_scores"]],
            })

        # Guardar JSON completo
        results = {
            "metadata": {
                "n_gallery_identities": int(len(unique_ids)),
                "n_gallery_templates": int(len(ids_full)),
                "n_lfw_train_images": int(len(Xtr)),
                "n_lfw_train_ids": int(len(train_ids)),
                "n_lfw_test_images": int(len(Xte)),
                "n_lfw_test_genuine_pairs": int(len(gen_lda)),
                "n_lfw_test_impostor_pairs": int(len(imp_lda)),
                "current_verif_threshold": float(bb.threshold),
                "current_ident_threshold": float(bb.ident_threshold),
                "current_dedup_threshold": float(bb.dedup_threshold),
                "calibration_json": cal,
            },
            "global_metrics": {
                "mlbp_lda_cosine": glob_lda,
                "mlbp_chi2_baseline": glob_chi2,
                "blufr_reference": {
                    "method": "HD-LBP + LDA (BLUFR, Liao et al. 2014)",
                    "far": BLUFR_HDLBP_LDA_FAR,
                    "tar": BLUFR_HDLBP_LDA_TAR,
                },
            },
            "verification_far_sweep": {
                "mlbp_lda_cosine": verif_lda_table,
                "mlbp_chi2_baseline": verif_chi2_table,
            },
            "gallery_impact_by_verif_far": gallery_impact_table,
            "identification_fpir_calibration": ident_calib_table,
            "gallery_probe_details": {
                "vinicius_loo": {
                    "fotos": [p.name for p in vini_paths],
                    "scores": [round(float(s), 4) for s in vini_loo_scores],
                    "ranks": vini_loo_ranks,
                },
                "enrolled_7_lfw": enrolled_details,
                "paula_top12": {
                    pname: sorted(sc.items(), key=lambda kv: -kv[1])[:12]
                    for pname, sc in paula_data.items()
                },
                "lfw_twins": {
                    tkey: {
                        "victim": tinfo["victim"],
                        "score_victim": float(tinfo["scores"][tinfo["victim"]]),
                        "top5": sorted(tinfo["scores"].items(), key=lambda kv: -kv[1])[:5],
                    }
                    for tkey, tinfo in lfw_twins_data.items()
                },
            },
            "curves": {
                "roc_mlbp_lda": sample_roc_curve(gen_lda, imp_lda),
                "roc_mlbp_chi2": sample_roc_curve(gen_chi2, imp_chi2),
            },
        }

        json_path = out_dir / "eval_far_levels.json"
        json_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"[3] JSON guardado en {json_path}")

        # ====================================================================
        # Grafica multi-panel (4 paneles)
        # ====================================================================
        make_figure(
            out_dir / "eval_far_levels.png",
            results,
            gen_lda,
            imp_lda,
            vini_loo_scores,
            paula_data,
        )
        log(f"[4] Grafica guardada en {out_dir / 'eval_far_levels.png'}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def make_figure(out_path, results, gen_lda, imp_lda, vini_loo_scores, paula_data):
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 13.0,
        "figure.titleweight": "bold",
    })

    C_LDA = "#1f5a99"
    C_CHI = "#d95f02"
    C_GRN = "#1b9e77"
    C_PRP = "#7570b3"
    C_RED = "#c0392b"
    C_BRN = "#8c564b"

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    fig.suptitle(
        "BigBrother (MLBP + PCA/LDA) — Análisis de rendimiento y seguridad por nivel de FAR (LFW test y galería real)",
        y=0.985,
    )

    roc_lda = results["curves"]["roc_mlbp_lda"]
    roc_chi = results["curves"]["roc_mlbp_chi2"]
    glob_lda = results["global_metrics"]["mlbp_lda_cosine"]
    glob_chi = results["global_metrics"]["mlbp_chi2_baseline"]
    verif_lda = results["verification_far_sweep"]["mlbp_lda_cosine"]
    gal_tab = results["gallery_impact_by_verif_far"]

    # ------------------------------------------------------------------
    # Panel 1: ROC en escala log-FAR (1e-5 a 1)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    far_l = np.array([max(p["far"], 1e-6) for p in roc_lda])
    tar_l = np.array([p["tar"] for p in roc_lda])
    far_c = np.array([max(p["far"], 1e-6) for p in roc_chi])
    tar_c = np.array([p["tar"] for p in roc_chi])

    ax.plot(
        far_l, tar_l, color=C_LDA, lw=2.2,
        label=f"MLBP + PCA/LDA + coseno (AUC={glob_lda['auc_roc']:.3f}, EER={glob_lda['eer']:.2%})",
    )
    ax.plot(
        far_c, tar_c, color=C_CHI, lw=1.8, ls="--",
        label=f"MLBP + χ² clásico, Ahonen (AUC={glob_chi['auc_roc']:.3f}, EER={glob_chi['eer']:.2%})",
    )
    ax.scatter(
        [BLUFR_HDLBP_LDA_FAR], [BLUFR_HDLBP_LDA_TAR],
        marker="D", s=65, color=C_GRN, edgecolor="black", zorder=6,
        label="Ref. BLUFR HD-LBP+LDA (36,12 % @ FAR 0,1 %)",
    )

    key_labels = {"0,01%", "0,1% (actual)", "1%", "5%", "EER"}
    offsets = {
        "0,01%": (-22, 14),
        "0,1% (actual)": (16, -26),
        "1%": (10, -18),
        "5%": (10, -18),
        "EER": (-36, 10),
    }
    for row in verif_lda:
        if row["label"] in key_labels:
            is_cur = row["label"] == "0,1% (actual)"
            ax.scatter(
                [row["far"]], [row["tar"]],
                color=C_RED if is_cur else C_LDA,
                s=55 if is_cur else 34, zorder=5,
                edgecolor="white", lw=0.8,
            )
            short = row["label"].replace(" (actual)", " (actual)")
            dx, dy = offsets[row["label"]]
            ax.annotate(
                f"{short}\n({row['tar']:.1%})",
                xy=(row["far"], row["tar"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                color=C_RED if is_cur else "#222222",
                fontweight="bold" if is_cur else "normal",
                arrowprops=dict(arrowstyle="-", color=C_RED if is_cur else "#777777", lw=0.7)
                if is_cur or row["label"] == "0,01%"
                else None,
            )

    ax.axvline(1e-3, color=C_RED, ls=":", lw=1.0, alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("FAR (escala logarítmica)")
    ax.set_ylabel("TAR (1 − FRR)")
    ax.set_title("(1) Curva ROC en LFW test (4.300 gen. / 980.849 imp.)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel 2: Curva DET en escala normal-deviate (probit)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    clip_min, clip_max = 1e-4, 0.99

    def to_det(pts):
        f = np.array([p["far"] for p in pts])
        r = np.array([p["frr"] for p in pts])
        m = (f >= clip_min) & (f <= clip_max) & (r >= clip_min) & (r <= clip_max)
        return norm.ppf(f[m]), norm.ppf(r[m])

    dx_l, dy_l = to_det(roc_lda)
    dx_c, dy_c = to_det(roc_chi)
    ax.plot(dx_l, dy_l, color=C_LDA, lw=2.2, label=f"MLBP + PCA/LDA (d' = {glob_lda['d_prime']:.2f})")
    ax.plot(dx_c, dy_c, color=C_CHI, lw=1.8, ls="--", label=f"MLBP + χ² clásico (d' = {glob_chi['d_prime']:.2f})")

    # Diagonal EER (FAR = FRR)
    diag = np.linspace(norm.ppf(1e-4), norm.ppf(0.85), 100)
    ax.plot(diag, diag, color="#888888", ls=":", lw=1.1, label="Línea EER (FAR = FRR)")
    ax.scatter([norm.ppf(glob_lda["eer"])], [norm.ppf(glob_lda["eer"])], color=C_LDA, s=45, zorder=5)
    ax.annotate(
        f"EER {glob_lda['eer']:.1%}",
        xy=(norm.ppf(glob_lda["eer"]), norm.ppf(glob_lda["eer"])),
        xytext=(-48, -16), textcoords="offset points", fontsize=7.5, color=C_LDA,
    )
    ax.scatter([norm.ppf(glob_chi["eer"])], [norm.ppf(glob_chi["eer"])], color=C_CHI, s=45, zorder=5)
    ax.annotate(
        f"EER {glob_chi['eer']:.1%}",
        xy=(norm.ppf(glob_chi["eer"]), norm.ppf(glob_chi["eer"])),
        xytext=(8, 6), textcoords="offset points", fontsize=7.5, color=C_CHI,
    )

    # Punto actual FAR 0.1%
    cur_row = next(r for r in verif_lda if r["label"] == "0,1% (actual)")
    ax.scatter(
        [norm.ppf(cur_row["far"])], [norm.ppf(cur_row["frr"])],
        color=C_RED, s=55, zorder=6, edgecolor="white", label="Actual (FAR 0,1 %, FRR 65,2 %)",
    )

    ticks_p = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
    ticks_lbl = ["0,01", "0,1", "0,5", "1", "2", "5", "10", "20", "40", "60", "80", "90"]
    ticks_pos = [norm.ppf(p) for p in ticks_p]
    ax.xaxis.set_major_locator(FixedLocator(ticks_pos))
    ax.xaxis.set_major_formatter(FixedFormatter(ticks_lbl))
    ax.yaxis.set_major_locator(FixedLocator(ticks_pos))
    ax.yaxis.set_major_formatter(FixedFormatter(ticks_lbl))
    ax.set_xlim(norm.ppf(8e-5), norm.ppf(0.65))
    ax.set_ylim(norm.ppf(0.015), norm.ppf(0.95))
    ax.set_xlabel("FAR (%) — escala normal-deviate")
    ax.set_ylabel("FRR (%) — escala normal-deviate")
    ax.set_title("(2) Curva DET (separabilidad normal-deviate)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel 3: Evolucion en galeria real (50 id.) segun FAR objetivo
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    x_fars = np.array([
        r["far"] if r["target_far"] == "EER" else float(r["target_far"])
        for r in verif_lda
    ])
    y_tar_lfw = np.array([r["tar"] * 100.0 for r in verif_lda])
    y_tar_7 = np.array([g["enrolled_7_tar"] * 100.0 for g in gal_tab])
    y_tar_vini = np.array([g["vinicius_loo_tar"] * 100.0 for g in gal_tab])
    y_fpir = np.array([g["open_set_fpir"] * 100.0 for g in gal_tab])
    y_vini_far = np.array([g["far_vinicius"] * 100.0 for g in gal_tab])

    ax.plot(x_fars, y_tar_lfw, "o-", color=C_LDA, lw=2.0, ms=4.5, label="TAR LFW test (1:1)")
    ax.plot(x_fars, y_tar_vini, "^--", color=C_GRN, lw=1.8, ms=5.0, label="TAR Vinícius LOO (5 fotos)")
    ax.plot(x_fars, y_tar_7, "s-", color=C_PRP, lw=2.0, ms=4.5, label="TAR 7 inscritos en LFW (55 fotos)")
    ax.plot(x_fars, y_fpir, "D-", color=C_RED, lw=2.0, ms=4.5, label="FPIR open-set (50 id., umbral verif.)")
    ax.plot(x_fars, y_vini_far, "v--", color=C_CHI, lw=1.8, ms=4.5, label="FAR contra Vinícius (4.499 desc.)")

    ax.axvline(1e-3, color=C_RED, ls=":", lw=1.2, alpha=0.8)
    ax.text(1.18e-3, 83, "Actual\n(FAR 0,1 %)", fontsize=7.8, color=C_RED, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(7e-6, 0.25)
    ax.set_ylim(-2, 106)
    ax.set_xlabel("FAR objetivo de verificación (escala logarítmica)")
    ax.set_ylabel("Tasa (%)")
    ax.set_title("(3) Compromiso usabilidad vs. seguridad en galería real (50 id.)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", fontsize=7.6, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel 4: Distribucion de scores genuinos vs impostores + umbrales
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    bins = np.linspace(-0.25, 0.55, 110)
    ax.hist(imp_lda, bins=bins, density=True, alpha=0.45, color=C_CHI, label="Impostores LFW (n=980.849)")
    ax.hist(gen_lda, bins=bins, density=True, alpha=0.50, color=C_LDA, label="Genuinos LFW (n=4.300)")

    thr_map = {r["label"]: r["threshold"] for r in verif_lda}
    thr_lines = [
        ("EER", thr_map["EER"], "#555555", ":"),
        ("FAR 5%", thr_map["5%"], C_BRN, "-."),
        ("FAR 1%", thr_map["1%"], C_PRP, "--"),
        ("FAR 0,1% (actual)", thr_map["0,1% (actual)"], C_RED, "-"),
        ("FAR 0,01%", thr_map["0,01%"], "#111111", "--"),
    ]
    y_top = ax.get_ylim()[1]
    for i_t, (lbl, tval, col, ls) in enumerate(thr_lines):
        ax.axvline(tval, color=col, ls=ls, lw=1.7, label=f"{lbl}: {tval:.3f}")

    # Marcar rangos de Vinicius LOO y Paula
    ax.scatter(
        vini_loo_scores,
        np.full(len(vini_loo_scores), 0.25),
        marker="^", s=45, color=C_GRN, edgecolor="black", zorder=6,
        label=f"Vinícius LOO ({vini_loo_scores.min():.2f}–{vini_loo_scores.max():.2f})",
    )
    paula_s = [paula_data["paula_01.jpg"][PAULA_VICTIM], paula_data["paula_02.jpg"][PAULA_VICTIM]]
    ax.scatter(
        paula_s,
        [0.55, 0.55],
        marker="X", s=60, color=C_RED, edgecolor="black", zorder=7,
        label=f"Gemela Paula → Aitana ({paula_s[0]:.2f}, {paula_s[1]:.2f})",
    )

    ax.set_xlim(-0.22, 0.55)
    ax.set_xlabel("Similitud coseno (espacio MLBP + PCA/LDA)")
    ax.set_ylabel("Densidad")
    ax.set_title("(4) Distribuciones de scores y umbrales operativos")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=7.3, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
