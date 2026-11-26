"""Evaluacion rigurosa del efecto de la distancia interocular (IOD) en BigBrother.

Uso (dentro de bigbrother:eval con LFW montado en /cache):
    python scripts/eval_interocular.py

Genera:
    - docs/resultados/eval_interocular.json
    - docs/resultados/eval_interocular.png
"""
import os

# Evitar contencion de hilos BLAS/OpenMP dentro de multiprocessing.Pool
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_db import build  # noqa: E402
from bigbrother import config  # noqa: E402
from bigbrother.dataset import FOTOS_DIR, enrollment_photos  # noqa: E402
from bigbrother.detector import FaceDetector  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402
from bigbrother.matcher import Faceness, Projection  # noqa: E402
from bigbrother.mlbp import MLBPExtractor  # noqa: E402
from bigbrother.preprocess import align, normalize_face  # noqa: E402

OUT_DIR = ROOT / "docs" / "resultados"
CALIB_THR = float(config.THRESHOLD)
IDENT_THR = float(config.IDENT_THRESHOLD)
INTERNAL_IOD = float((0.69 - 0.31) * config.FACE_SIZE)  # 48.64 px
ISO_IOD = 60.0

LFW_TARGETS = [10, 12, 16, 20, 24, 30, 36, 48, 60]
HIRES_TARGETS = [10, 12, 16, 20, 24, 30, 36, 48, 60, 80, 100, 120]

# Paleta coherente con docs/resultados/
C_AZUL = "#4a6fa5"
C_NARANJA = "#d98e32"
C_VERDE = "#3d8b5f"
C_ROJO = "#c0392b"
C_MORADO = "#7d5ba6"
C_CIAN = "#2b8cbe"

# --- Globales por worker -----------------------------------------------------
_W_DET = None
_W_EX = None
_W_PROJ = None
_W_FACE = None


def _init_worker():
    global _W_DET, _W_EX, _W_PROJ, _W_FACE
    cv2.setNumThreads(1)
    _W_DET = FaceDetector()
    _W_EX = MLBPExtractor()
    _W_PROJ = Projection()
    _W_FACE = Faceness()


def _analyze_face(face):
    x, y, w, h = face.bbox
    min_side = float(min(w, h))
    re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_d = float(np.linalg.norm(le - re))
    axis = (le - re) / max(eye_d, 1e-9)
    yaw = abs(float(np.dot(nose - (le + re) / 2.0, axis))) / max(eye_d, 1e-9)
    return min_side, eye_d, yaw


def _process_lfw_item(args):
    """Procesa una imagen LFW a resolucion natural (target_iod=None) o reescalada."""
    path_str, nat_iod, target_iod = args
    img = cv2.imread(path_str)
    if img is None:
        return {"ok": False, "no_face": True, "seq_reason": "NoFaceDetected", "seq_reason_no_ambig": "NoFaceDetected"}

    if target_iod is None:
        resized = img
        scale = 1.0
    else:
        if nat_iod <= 0.0:
            return {"ok": False, "no_face": True, "seq_reason": "NoFaceDetected", "seq_reason_no_ambig": "NoFaceDetected"}
        scale = float(target_iod) / float(nat_iod)
        if abs(scale - 1.0) < 1e-6:
            resized = img
        else:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            h, w = img.shape[:2]
            nw = max(1, round(w * scale))
            nh = max(1, round(h * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=interp)

    h_r, w_r = resized.shape[:2]
    invalid_min_side = min(h_r, w_r) < config.MIN_IMAGE_SIDE

    faces = _W_DET.detect_all(resized)
    if not faces:
        return {
            "ok": False,
            "invalid_min_side": invalid_min_side,
            "no_face": True,
            "ambiguous": False,
            "fail_min_face": False,
            "fail_min_eye": False,
            "fail_yaw": False,
            "seq_reason": "NoFaceDetected",
            "seq_reason_no_ambig": "NoFaceDetected",
            "det_iod": None,
            "min_side": None,
            "yaw": None,
            "faceness_score": None,
            "faceness_ok": False,
            "y": None,
            "scale": scale,
        }

    f = faces[0]
    ambiguous = len(faces) > 1 and faces[1].area >= config.AMBIGUOUS_FACE_RATIO * f.area
    min_side, det_iod, yaw = _analyze_face(f)
    fail_min_face = min_side < config.MIN_FACE_PX
    fail_min_eye = det_iod < config.MIN_EYE_DIST_PX
    fail_yaw = yaw > config.MAX_YAW_RATIO

    if ambiguous:
        seq_reason = "AmbiguousFace"
    elif fail_min_face:
        seq_reason = "MIN_FACE_PX"
    elif fail_min_eye:
        seq_reason = "MIN_EYE_DIST_PX"
    elif fail_yaw:
        seq_reason = "MAX_YAW_RATIO"
    else:
        seq_reason = "OK"

    if fail_min_face:
        seq_no_ambig = "MIN_FACE_PX"
    elif fail_min_eye:
        seq_no_ambig = "MIN_EYE_DIST_PX"
    elif fail_yaw:
        seq_no_ambig = "MAX_YAW_RATIO"
    else:
        seq_no_ambig = "OK"

    crop = align(resized, f.landmarks)
    t = _W_EX.extract(normalize_face(crop))
    fscore = float(_W_FACE.score(t))
    y = _W_PROJ(t).astype(np.float32)

    return {
        "ok": True,
        "invalid_min_side": invalid_min_side,
        "no_face": False,
        "ambiguous": ambiguous,
        "fail_min_face": fail_min_face,
        "fail_min_eye": fail_min_eye,
        "fail_yaw": fail_yaw,
        "seq_reason": seq_reason,
        "seq_reason_no_ambig": seq_no_ambig,
        "det_iod": det_iod,
        "min_side": min_side,
        "yaw": yaw,
        "faceness_score": fscore,
        "faceness_ok": fscore >= _W_FACE.threshold,
        "y": y,
        "scale": scale,
    }


def pair_scores_with_indices(Y, labels, rng, n_imp=1_000_000, max_per_id=10):
    """Genera pares genuinos e impostores devolviendo scores e indices locales."""
    yi = np.unique(labels, return_inverse=True)[1]
    g_a, g_b = [], []
    for c in np.unique(yi):
        idx = np.where(yi == c)[0]
        if len(idx) > max_per_id:
            idx = np.random.default_rng(int(c)).choice(idx, max_per_id, replace=False)
        if len(idx) > 1:
            ii, jj = np.triu_indices(len(idx), 1)
            g_a.append(idx[ii])
            g_b.append(idx[jj])
    if g_a:
        ga = np.concatenate(g_a)
        gb = np.concatenate(g_b)
        gen = np.einsum("ij,ij->i", Y[ga], Y[gb])
    else:
        ga = np.array([], dtype=np.int64)
        gb = np.array([], dtype=np.int64)
        gen = np.array([], dtype=np.float32)

    if len(Y) >= 2 and len(np.unique(yi)) >= 2:
        a, b = rng.integers(0, len(Y), (2, n_imp))
        m = yi[a] != yi[b]
        ia, ib = a[m], b[m]
        imp = np.einsum("ij,ij->i", Y[ia], Y[ib])
    else:
        ia = np.array([], dtype=np.int64)
        ib = np.array([], dtype=np.int64)
        imp = np.array([], dtype=np.float32)
    return gen, imp, ga, gb, ia, ib


def compute_biometric_metrics(gen, imp, thr=CALIB_THR):
    """Calcula EER, TAR@FAR={0.1%, 1%, 5%}, y FAR/FRR al umbral calibrado."""
    if len(gen) < 10 or len(imp) < 100:
        return None
    ths = np.unique(np.quantile(imp, np.linspace(0, 1, 4001)))
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    idx = int(np.argmin(np.abs(far - frr)))
    out = {
        "n_gen": int(len(gen)),
        "n_imp": int(len(imp)),
        "eer": float((far[idx] + frr[idx]) / 2.0),
        "eer_thr": float(ths[idx]),
        "far_at_calib_thr": float((imp >= thr).mean()),
        "frr_at_calib_thr": float((gen < thr).mean()),
        "tar_at_calib_thr": float((gen >= thr).mean()),
        "gen_mean": float(np.mean(gen)),
        "gen_std": float(np.std(gen)),
        "gen_median": float(np.median(gen)),
        "imp_mean": float(np.mean(imp)),
        "imp_std": float(np.std(imp)),
        "imp_median": float(np.median(imp)),
    }
    for target in (1e-3, 1e-2, 5e-2):
        t = float(np.quantile(imp, 1.0 - target))
        out[f"tar@far{target:g}"] = float((gen >= t).mean())
        out[f"thr@far{target:g}"] = t
    return out


def summarize_fta(results):
    n = len(results)
    seq_counts = {
        "NoFaceDetected": 0,
        "AmbiguousFace": 0,
        "MIN_FACE_PX": 0,
        "MIN_EYE_DIST_PX": 0,
        "MAX_YAW_RATIO": 0,
        "OK": 0,
    }
    seq_no_ambig_counts = {
        "NoFaceDetected": 0,
        "MIN_FACE_PX": 0,
        "MIN_EYE_DIST_PX": 0,
        "MAX_YAW_RATIO": 0,
        "OK": 0,
    }
    marginal = {
        "invalid_min_side": 0,
        "no_face": 0,
        "ambiguous": 0,
        "min_face_px": 0,
        "min_eye_dist_px": 0,
        "max_yaw_ratio": 0,
        "faceness_rejected_among_detected": 0,
        "faceness_rejected_among_strict_ok": 0,
    }
    det_iods = []

    for r in results:
        seq_counts[r["seq_reason"]] += 1
        seq_no_ambig_counts[r["seq_reason_no_ambig"]] += 1
        if r.get("invalid_min_side"):
            marginal["invalid_min_side"] += 1
        if r.get("no_face"):
            marginal["no_face"] += 1
            continue
        if r["ambiguous"]:
            marginal["ambiguous"] += 1
        if r["fail_min_face"]:
            marginal["min_face_px"] += 1
        if r["fail_min_eye"]:
            marginal["min_eye_dist_px"] += 1
        if r["fail_yaw"]:
            marginal["max_yaw_ratio"] += 1
        if not r["faceness_ok"]:
            marginal["faceness_rejected_among_detected"] += 1
            if r["seq_reason"] == "OK":
                marginal["faceness_rejected_among_strict_ok"] += 1
        det_iods.append(r["det_iod"])

    n_det = n - marginal["no_face"]
    n_strict_det_ok = seq_counts["OK"]
    n_strict_full_ok = n_strict_det_ok - marginal["faceness_rejected_among_strict_ok"]
    n_no_ambig_ok = seq_no_ambig_counts["OK"]

    det_iods_arr = np.array(det_iods, dtype=np.float64) if det_iods else np.array([])
    return {
        "n_total": n,
        "n_detected": n_det,
        "n_strict_detector_ok": n_strict_det_ok,
        "n_strict_full_ok": n_strict_full_ok,
        "n_quality_no_ambig_ok": n_no_ambig_ok,
        "fta_detector_strict": 1.0 - n_strict_det_ok / n,
        "fta_full_strict": 1.0 - n_strict_full_ok / n,
        "fta_quality_no_ambig": 1.0 - n_no_ambig_ok / n,
        "seq_counts": seq_counts,
        "seq_rates": {k: v / n for k, v in seq_counts.items()},
        "seq_no_ambig_counts": seq_no_ambig_counts,
        "seq_no_ambig_rates": {k: v / n for k, v in seq_no_ambig_counts.items()},
        "marginal_counts": marginal,
        "marginal_rates": {
            "invalid_min_side": marginal["invalid_min_side"] / n,
            "no_face": marginal["no_face"] / n,
            "ambiguous": marginal["ambiguous"] / n,
            "min_face_px": marginal["min_face_px"] / n,
            "min_eye_dist_px": marginal["min_eye_dist_px"] / n,
            "max_yaw_ratio": marginal["max_yaw_ratio"] / n,
            "faceness_among_detected": (
                marginal["faceness_rejected_among_detected"] / max(n_det, 1)
            ),
            "faceness_among_strict_ok": (
                marginal["faceness_rejected_among_strict_ok"] / max(n_strict_det_ok, 1)
            ),
        },
        "detected_iod_stats": (
            {
                "mean": float(det_iods_arr.mean()),
                "std": float(det_iods_arr.std()),
                "median": float(np.median(det_iods_arr)),
                "p10": float(np.percentile(det_iods_arr, 10)),
                "p90": float(np.percentile(det_iods_arr, 90)),
            }
            if len(det_iods_arr)
            else None
        ),
    }


def eval_subset_metrics(results, names, mask):
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return None
    Y = np.stack([results[i]["y"] for i in idx])
    y_lbl = names[idx]
    gen, imp, *_ = pair_scores_with_indices(Y, y_lbl, np.random.default_rng(1), n_imp=1_000_000)
    return compute_biometric_metrics(gen, imp)


def run_experiment_a_and_c(lfw_dir, workers):
    print("[Exp A & C] Cargando split LFW test...", flush=True)
    split = json.loads((config.MODELS_DIR / "lfw_split.json").read_text(encoding="utf-8"))
    train_ids, reserved = set(split["train_ids"]), set(split["reserved"])
    all_paths = sorted(Path(lfw_dir).glob("*/*.jpg"))
    test_paths = [
        p for p in all_paths if p.parent.name not in train_ids and p.parent.name not in reserved
    ]
    test_names = np.array([p.parent.name for p in test_paths])
    path_strs = [str(p) for p in test_paths]
    print(f"  LFW test: {len(test_paths)} imagenes, {len(set(test_names))} identidades", flush=True)

    with Pool(workers, initializer=_init_worker) as pool:
        t0 = time.time()
        nat_res = pool.map(
            _process_lfw_item, [(p, 0.0, None) for p in path_strs], chunksize=64
        )
        print(f"  Pasada natural completada en {time.time() - t0:.1f}s", flush=True)

        nat_iods = np.array(
            [r["det_iod"] if r["ok"] else np.nan for r in nat_res], dtype=np.float64
        )
        valid_nat = nat_iods[np.isfinite(nat_iods)]
        nat_summary = {
            "n_total": len(nat_res),
            "n_detected": int(np.isfinite(nat_iods).sum()),
            "iod_stats": {
                "min": float(valid_nat.min()),
                "p5": float(np.percentile(valid_nat, 5)),
                "p10": float(np.percentile(valid_nat, 10)),
                "p25": float(np.percentile(valid_nat, 25)),
                "median": float(np.median(valid_nat)),
                "mean": float(valid_nat.mean()),
                "std": float(valid_nat.std()),
                "p75": float(np.percentile(valid_nat, 75)),
                "p90": float(np.percentile(valid_nat, 90)),
                "p95": float(np.percentile(valid_nat, 95)),
                "max": float(valid_nat.max()),
                "frac_ge_48_64": float((valid_nat >= INTERNAL_IOD).mean()),
                "frac_ge_60": float((valid_nat >= ISO_IOD).mean()),
                "count_ge_60": int((valid_nat >= ISO_IOD).sum()),
            },
            "fta": summarize_fta(nat_res),
            "metrics_no_size_cut_with_faceness": eval_subset_metrics(
                nat_res, test_names, np.array([r["ok"] and r["faceness_ok"] for r in nat_res])
            ),
            "metrics_no_size_cut_all_detected": eval_subset_metrics(
                nat_res, test_names, np.array([r["ok"] for r in nat_res])
            ),
            "metrics_strict_quality": eval_subset_metrics(
                nat_res,
                test_names,
                np.array([r["ok"] and r["seq_reason"] == "OK" and r["faceness_ok"] for r in nat_res]),
            ),
        }

        # Experimento C: estratificacion por IOD natural en LFW test
        print("[Exp C] Estratificacion por IOD natural...", flush=True)
        ok_nat_mask = np.array([r["ok"] and r["faceness_ok"] for r in nat_res])
        idx_ok = np.where(ok_nat_mask)[0]
        Y_nat = np.stack([nat_res[i]["y"] for i in idx_ok])
        lbl_nat = test_names[idx_ok]
        iod_ok = nat_iods[idx_ok]
        gen, imp, ga, gb, ia, ib = pair_scores_with_indices(
            Y_nat, lbl_nat, np.random.default_rng(1), n_imp=1_000_000
        )
        gen_min_iod = np.minimum(iod_ok[ga], iod_ok[gb])
        gen_diff_iod = np.abs(iod_ok[ga] - iod_ok[gb])
        imp_min_iod = np.minimum(iod_ok[ia], iod_ok[ib])

        r_min_p, p_min_p = pearsonr(gen_min_iod, gen)
        r_min_s, p_min_s = spearmanr(gen_min_iod, gen)
        r_diff_p, p_diff_p = pearsonr(gen_diff_iod, gen)
        r_diff_s, p_diff_s = spearmanr(gen_diff_iod, gen)

        def eval_stratum(g_mask, i_mask):
            g_sub = gen[g_mask]
            i_sub = imp[i_mask] if i_mask.sum() >= 5000 else imp
            m_all_imp = compute_biometric_metrics(g_sub, imp)
            m_matched_imp = compute_biometric_metrics(g_sub, i_sub) if i_mask.sum() >= 5000 else None
            return {
                "n_gen": int(g_mask.sum()),
                "n_imp_matched": int(i_mask.sum()),
                "min_iod_mean": float(gen_min_iod[g_mask].mean()) if g_mask.any() else None,
                "gen_score_mean": float(g_sub.mean()) if len(g_sub) else None,
                "gen_score_std": float(g_sub.std()) if len(g_sub) else None,
                "tar_at_calib_thr": float((g_sub >= CALIB_THR).mean()) if len(g_sub) else None,
                "metrics_vs_all_imp": m_all_imp,
                "metrics_vs_matched_imp": m_matched_imp,
            }

        bands = [(0.0, 35.0, "<35"), (35.0, 40.0, "35-40"), (40.0, 45.0, "40-45"), (45.0, 50.0, "45-50"), (50.0, 1e9, ">=50")]
        band_results = {}
        for lo, hi, label in bands:
            gm = (gen_min_iod >= lo) & (gen_min_iod < hi)
            im = (imp_min_iod >= lo) & (imp_min_iod < hi)
            band_results[label] = eval_stratum(gm, im)

        q1, q2 = np.percentile(gen_min_iod, [33.3333, 66.6667])
        terciles = [
            (0.0, float(q1), f"T1 (<={q1:.2f} px)"),
            (float(q1), float(q2), f"T2 ({q1:.2f}-{q2:.2f} px)"),
            (float(q2), 1e9, f"T3 (>{q2:.2f} px)"),
        ]
        tercile_results = {}
        for lo, hi, label in terciles:
            if lo == 0.0:
                gm = gen_min_iod <= hi
                im = imp_min_iod <= hi
            else:
                gm = (gen_min_iod > lo) & (gen_min_iod <= hi)
                im = (imp_min_iod > lo) & (imp_min_iod <= hi)
            tercile_results[label] = eval_stratum(gm, im)

        exp_c = {
            "n_genuine_pairs": int(len(gen)),
            "n_impostor_pairs": int(len(imp)),
            "correlations_with_genuine_score": {
                "min_iod_pearson_r": float(r_min_p),
                "min_iod_pearson_p": float(p_min_p),
                "min_iod_spearman_rho": float(r_min_s),
                "min_iod_spearman_p": float(p_min_s),
                "abs_iod_diff_pearson_r": float(r_diff_p),
                "abs_iod_diff_pearson_p": float(p_diff_p),
                "abs_iod_diff_spearman_rho": float(r_diff_s),
                "abs_iod_diff_spearman_p": float(p_diff_s),
            },
            "tercile_cutoffs_px": [float(q1), float(q2)],
            "by_tercile": tercile_results,
            "by_band": band_results,
        }

        # Experimento A: barrido controlado de IOD
        print("[Exp A] Barrido controlado de IOD en LFW test...", flush=True)
        exp_a_sweep = {}
        for target in LFW_TARGETS:
            t1 = time.time()
            tasks = [
                (p, float(nat_iods[i]) if np.isfinite(nat_iods[i]) else 0.0, float(target))
                for i, p in enumerate(path_strs)
            ]
            res = pool.map(_process_lfw_item, tasks, chunksize=64)

            n_tot = len(res)
            n_down = int(np.sum(np.isfinite(nat_iods) & (nat_iods >= target)))
            n_up = int(np.sum(np.isfinite(nat_iods) & (nat_iods < target)))

            mask_no_cut_face = np.array([r["ok"] and r["faceness_ok"] for r in res])
            mask_no_cut_all = np.array([r["ok"] for r in res])
            mask_strict = np.array(
                [r["ok"] and r["seq_reason"] == "OK" and r["faceness_ok"] for r in res]
            )
            mask_down_only = mask_no_cut_face & np.isfinite(nat_iods) & (nat_iods >= target)

            fta = summarize_fta(res)
            m_no_cut_face = eval_subset_metrics(res, test_names, mask_no_cut_face)
            m_no_cut_all = eval_subset_metrics(res, test_names, mask_no_cut_all)
            m_strict = eval_subset_metrics(res, test_names, mask_strict)
            m_down = eval_subset_metrics(res, test_names, mask_down_only)

            dt = time.time() - t1
            eer_nc = m_no_cut_face["eer"] if m_no_cut_face else float("nan")
            eer_st = m_strict["eer"] if m_strict else float("nan")
            print(
                f"  target={target:2d}px ({dt:4.1f}s) | down={100*n_down/n_tot:5.1f}% up={100*n_up/n_tot:5.1f}% | "
                f"FTA_det={100*fta['fta_detector_strict']:5.1f}% FTA_noamb={100*fta['fta_quality_no_ambig']:5.1f}% "
                f"face_rej={100*fta['marginal_rates']['faceness_among_detected']:4.2f}% | "
                f"EER(no_cut)={100*eer_nc:5.2f}% EER(strict)={100*eer_st:5.2f}%",
                flush=True,
            )

            exp_a_sweep[str(target)] = {
                "target_iod_px": target,
                "runtime_s": round(dt, 2),
                "rescaling": {
                    "n_downscaled": n_down,
                    "pct_downscaled": 100.0 * n_down / n_tot,
                    "n_upscaled": n_up,
                    "pct_upscaled": 100.0 * n_up / n_tot,
                },
                "fta": fta,
                "metrics_no_size_cut_with_faceness": m_no_cut_face,
                "metrics_no_size_cut_all_detected": m_no_cut_all,
                "metrics_strict_quality": m_strict,
                "metrics_downscaled_only": m_down,
            }

    return {
        "natural_baseline": nat_summary,
        "sweep": exp_a_sweep,
        "valid_nat_iods": valid_nat,
    }, exp_c


def run_experiment_b():
    print("[Exp B] Evaluando galeria real (data/fotos/) y saturacion en vinicius_junior...", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="bb_iod_"))
    try:
        bb = build(tmp / "db.csv", tmp / "keys", verbose=False)
        det = bb.detector
        ex = bb.extractor
        proj = bb.projection
        face_clf = bb.faceness

        photos = enrollment_photos(FOTOS_DIR)
        gallery_records = []
        vini_items = []
        other_items = []

        for id_, p in photos:
            img = load_image(p)
            f = det.detect(
                img,
                strict=False,
                is_face=lambda face, _im=img: (
                    face_clf.score(ex.extract(normalize_face(align(_im, face.landmarks))))
                    >= face_clf.threshold
                ),
            )
            min_side, nat_iod, yaw = _analyze_face(f)
            crop = align(img, f.landmarks)
            t = ex.extract(normalize_face(crop))
            y = proj(t)
            rec = {
                "id": id_,
                "filename": p.name,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "natural_iod_px": round(nat_iod, 2),
                "min_bbox_px": round(min_side, 2),
                "yaw_ratio": round(yaw, 4),
                "faceness_score": round(float(face_clf.score(t)), 4),
            }
            gallery_records.append(rec)
            item = {"id": id_, "path": p, "img": img, "nat_iod": nat_iod, "y_nat": y}
            if id_ == "vinicius_junior":
                vini_items.append(item)
            else:
                other_items.append(item)

        all_photo_iods = np.array([r["natural_iod_px"] for r in gallery_records])
        by_id = {}
        for r in gallery_records:
            by_id.setdefault(r["id"], []).append(r["natural_iod_px"])
        per_id_iods = np.array([float(np.mean(vals)) for _, vals in sorted(by_id.items())])

        def stats_dict(arr):
            return {
                "n": int(len(arr)),
                "min": float(arr.min()),
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(arr.max()),
                "count_ge_60": int((arr >= ISO_IOD).sum()),
                "pct_ge_60": float(100.0 * (arr >= ISO_IOD).mean()),
            }

        vini_refs = np.stack(bb.db.get("vinicius_junior"))
        all_db_items = bb.db.all()

        def gallery_matrix_excluding_vini(excl_idx):
            ids, vecs = [], []
            k = 0
            for id_, vec in all_db_items:
                if id_ == "vinicius_junior":
                    if k == excl_idx:
                        k += 1
                        continue
                    k += 1
                ids.append(id_)
                vecs.append(vec)
            return np.array(ids), np.stack(vecs)

        hires_single_face = []
        for it in other_items:
            if it["nat_iod"] >= 120.0:
                s = 48.0 / it["nat_iod"]
                h, w = it["img"].shape[:2]
                rz = cv2.resize(it["img"], (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                fs = det.detect_all(rz)
                if fs:
                    f0 = det.detect_all(it["img"])[0]
                    c0 = np.array([f0.bbox[0] + f0.bbox[2] / 2, f0.bbox[1] + f0.bbox[3] / 2]) * s
                    c2 = np.array([fs[0].bbox[0] + fs[0].bbox[2] / 2, fs[0].bbox[1] + fs[0].bbox[3] / 2])
                    if np.linalg.norm(c0 - c2) < 0.35 * 48.0:
                        hires_single_face.append(it)

        print(f"  Galeria: 54 fotos / 50 IDs | IOD mediana={np.median(per_id_iods):.1f}px | "
              f"hires_single_face (>=120px)={len(hires_single_face)}", flush=True)

        hires_120_vecs = {}
        for it in hires_single_face:
            s120 = 120.0 / it["nat_iod"]
            h, w = it["img"].shape[:2]
            rz120 = cv2.resize(it["img"], (round(w * s120), round(h * s120)), interpolation=cv2.INTER_AREA)
            f120 = det.detect_all(rz120)[0]
            t120 = ex.extract(normalize_face(align(rz120, f120.landmarks)))
            hires_120_vecs[it["id"]] = proj(t120)

        sweep_b = {}
        for target in HIRES_TARGETS:
            vini_vecs = []
            vini_det_ok = 0
            vini_qual_ok = 0
            vini_face_ok = 0
            vini_fail_reasons = []
            vini_det_iods = []
            vini_asym_scores = []
            vini_ranks = []
            vini_ident_ok = 0
            vini_downscaled = 0

            for i, it in enumerate(vini_items):
                if it["nat_iod"] >= target:
                    vini_downscaled += 1
                s = float(target) / it["nat_iod"]
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                h, w = it["img"].shape[:2]
                rz = cv2.resize(it["img"], (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
                faces = det.detect_all(rz)
                if not faces:
                    vini_vecs.append(None)
                    vini_fail_reasons.append("NoFaceDetected")
                    continue
                vini_det_ok += 1
                f = faces[0]
                min_side, det_iod, yaw = _analyze_face(f)
                vini_det_iods.append(det_iod)
                amb = len(faces) > 1 and faces[1].area >= config.AMBIGUOUS_FACE_RATIO * f.area
                if amb:
                    vini_fail_reasons.append("AmbiguousFace")
                elif min_side < config.MIN_FACE_PX:
                    vini_fail_reasons.append("MIN_FACE_PX")
                elif det_iod < config.MIN_EYE_DIST_PX:
                    vini_fail_reasons.append("MIN_EYE_DIST_PX")
                elif yaw > config.MAX_YAW_RATIO:
                    vini_fail_reasons.append("MAX_YAW_RATIO")
                else:
                    vini_qual_ok += 1
                    vini_fail_reasons.append("OK")

                t = ex.extract(normalize_face(align(rz, f.landmarks)))
                if float(face_clf.score(t)) >= face_clf.threshold:
                    vini_face_ok += 1
                y = proj(t)
                vini_vecs.append(y)

                sc_asym = float(np.delete(vini_refs @ y, i).max())
                vini_asym_scores.append(sc_asym)

                g_ids, g_M = gallery_matrix_excluding_vini(i)
                sims = g_M @ y
                best_per_id = {u: float(sims[g_ids == u].max()) for u in np.unique(g_ids)}
                order = sorted(best_per_id, key=lambda u: -best_per_id[u])
                vini_ranks.append(order.index("vinicius_junior") + 1)
                if order[0] == "vinicius_junior" and best_per_id["vinicius_junior"] >= IDENT_THR:
                    vini_ident_ok += 1

            vini_sym_scores = []
            if all(v is not None for v in vini_vecs):
                V = np.stack(vini_vecs)
                S_sym = V @ V.T
                for i in range(len(V)):
                    vini_sym_scores.append(float(np.delete(S_sym[i], i).max()))

            other_vecs = []
            other_vecs_aa = []
            hires_cos_vs_120 = []
            hires_cos_vs_nat = []
            hires_ids = {it["id"] for it in hires_single_face}

            for it in other_items:
                s = float(target) / it["nat_iod"]
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                h, w = it["img"].shape[:2]
                rz = cv2.resize(it["img"], (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
                faces = det.detect_all(rz)
                if not faces:
                    continue
                f = faces[0]
                t = ex.extract(normalize_face(align(rz, f.landmarks)))
                y = proj(t)
                other_vecs.append(y)
                if float(target) > INTERNAL_IOD:
                    k_aa = INTERNAL_IOD / float(target)
                    rz_aa = cv2.resize(
                        rz,
                        (max(1, round(rz.shape[1] * k_aa)), max(1, round(rz.shape[0] * k_aa))),
                        interpolation=cv2.INTER_AREA,
                    )
                    y_aa = proj(ex.extract(normalize_face(align(rz_aa, f.landmarks * k_aa))))
                    other_vecs_aa.append(y_aa)
                else:
                    other_vecs_aa.append(y)
                if it["id"] in hires_ids:
                    hires_cos_vs_120.append(float(y @ hires_120_vecs[it["id"]]))
                    hires_cos_vs_nat.append(float(y @ it["y_nat"]))

            O = np.stack(other_vecs)
            imp_vs_vini = (O @ vini_refs.T).max(axis=1)
            S_oth = O @ O.T
            iu, ju = np.triu_indices(len(O), 1)
            imp_pairs_49 = S_oth[iu, ju]

            O_aa = np.stack(other_vecs_aa)
            S_aa = O_aa @ O_aa.T
            imp_pairs_49_aa = S_aa[iu, ju]

            sweep_b[str(target)] = {
                "target_iod_px": target,
                "vinicius_loo": {
                    "n_photos": len(vini_items),
                    "n_downscaled": vini_downscaled,
                    "det_ok": vini_det_ok,
                    "strict_qual_ok": vini_qual_ok,
                    "faceness_ok": vini_face_ok,
                    "fail_reasons": vini_fail_reasons,
                    "detected_iod_mean": float(np.mean(vini_det_iods)) if vini_det_iods else None,
                    "asym_scores": [round(x, 4) for x in vini_asym_scores],
                    "asym_mean": float(np.mean(vini_asym_scores)) if vini_asym_scores else None,
                    "asym_min": float(np.min(vini_asym_scores)) if vini_asym_scores else None,
                    "asym_max": float(np.max(vini_asym_scores)) if vini_asym_scores else None,
                    "asym_accepted_02172": int(sum(x >= CALIB_THR for x in vini_asym_scores)),
                    "asym_ranks": vini_ranks,
                    "asym_rank1_count": int(sum(r == 1 for r in vini_ranks)),
                    "asym_identified_02847": int(vini_ident_ok),
                    "sym_scores": [round(x, 4) for x in vini_sym_scores],
                    "sym_mean": float(np.mean(vini_sym_scores)) if vini_sym_scores else None,
                    "sym_accepted_02172": int(sum(x >= CALIB_THR for x in vini_sym_scores)),
                },
                "gallery_impostors_49": {
                    "n_detected": len(other_vecs),
                    "vs_vinicius_mean": float(imp_vs_vini.mean()),
                    "vs_vinicius_std": float(imp_vs_vini.std()),
                    "vs_vinicius_max": float(imp_vs_vini.max()),
                    "vs_vinicius_far_02172": float((imp_vs_vini >= CALIB_THR).mean()),
                    "pairwise_49_mean": float(imp_pairs_49.mean()),
                    "pairwise_49_std": float(imp_pairs_49.std()),
                    "pairwise_49_p99": float(np.percentile(imp_pairs_49, 99)),
                    "pairwise_49_far_02172": float((imp_pairs_49 >= CALIB_THR).mean()),
                    "pairwise_49_antialias_mean": float(imp_pairs_49_aa.mean()),
                    "pairwise_49_antialias_far_02172": float((imp_pairs_49_aa >= CALIB_THR).mean()),
                },
                "hires_ge120_self_similarity": {
                    "n": len(hires_cos_vs_120),
                    "cos_vs_120px_mean": float(np.mean(hires_cos_vs_120)) if hires_cos_vs_120 else None,
                    "cos_vs_120px_min": float(np.min(hires_cos_vs_120)) if hires_cos_vs_120 else None,
                    "cos_vs_nat_mean": float(np.mean(hires_cos_vs_nat)) if hires_cos_vs_nat else None,
                },
            }
            print(
                f"  target={target:3d}px | vini_asym={np.mean(vini_asym_scores):.4f} "
                f"({sum(x >= CALIB_THR for x in vini_asym_scores)}/5) "
                f"vini_sym={np.mean(vini_sym_scores):.4f} | "
                f"imp49_mean={imp_pairs_49.mean():.4f} (aa={imp_pairs_49_aa.mean():.4f}) | "
                f"cos_vs_120={np.mean(hires_cos_vs_120):.4f}",
                flush=True,
            )

        return {
            "gallery_stats_all_54_photos": stats_dict(all_photo_iods),
            "gallery_stats_50_identities": stats_dict(per_id_iods),
            "vinicius_natural_iods": [
                {
                    "filename": r["filename"],
                    "natural_iod_px": r["natural_iod_px"],
                    "width": r["width"],
                    "height": r["height"],
                }
                for r in gallery_records
                if r["id"] == "vinicius_junior"
            ],
            "gallery_photos": gallery_records,
            "all_photo_iods": all_photo_iods,
            "sweep": sweep_b,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_figure(res_a, res_b, res_c, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # ---------------- Panel 1: EER y TAR@FAR={0.1%, 1%, 5%} vs IOD (LFW test)
    xs = LFW_TARGETS
    sw_a = res_a["sweep"]

    def series(regime, key, scale=100.0):
        vals = []
        for t in xs:
            m = sw_a[str(t)][regime]
            vals.append(m[key] * scale if m is not None else np.nan)
        return np.array(vals, dtype=float)

    ax1.axvspan(8, config.MIN_EYE_DIST_PX, color=C_ROJO, alpha=0.07, label=f"Zona bloqueada (<{config.MIN_EYE_DIST_PX} px)")
    ax1.plot(xs, series("metrics_no_size_cut_with_faceness", "eer"), "o-", color=C_ROJO, lw=2.2, label="EER (%) [sin corte tamaño]")
    ax1.plot(xs, series("metrics_no_size_cut_with_faceness", "tar@far0.05"), "^-", color=C_VERDE, lw=2.0, label="TAR @ FAR=5 % [sin corte]")
    ax1.plot(xs, series("metrics_no_size_cut_with_faceness", "tar@far0.01"), "s-", color=C_AZUL, lw=2.0, label="TAR @ FAR=1 % [sin corte]")
    ax1.plot(xs, series("metrics_no_size_cut_with_faceness", "tar@far0.001"), "d-", color=C_NARANJA, lw=2.0, label="TAR @ FAR=0,1 % [sin corte]")

    ax1.plot(xs, series("metrics_strict_quality", "eer"), "o--", color=C_ROJO, alpha=0.55, lw=1.2, ms=4, label="EER (%) [calidad estricta]")
    ax1.plot(xs, series("metrics_strict_quality", "tar@far0.001"), "d--", color=C_NARANJA, alpha=0.55, lw=1.2, ms=4, label="TAR @ FAR=0,1 % [estricto]")

    for xv, txt, col, ha, dx in (
        (config.MIN_EYE_DIST_PX, "16 px\n(MIN_EYE)", "#444444", "left", 0.7),
        (INTERNAL_IOD, "48,6 px\n(128×128)", C_MORADO, "right", -0.7),
        (ISO_IOD, "60 px\n(ISO 19794-5)", "#111111", "right", -0.7),
    ):
        ax1.axvline(xv, ls="--", lw=1.2, color=col, alpha=0.85)
        ax1.text(xv + dx, 2.0, txt, fontsize=7.5, color=col, va="bottom", ha=ha, fontweight="bold")

    ax1.set(
        xlim=(8, 64),
        ylim=(0, 102),
        xlabel="Distancia interocular objetivo (px)",
        ylabel="Porcentaje (%)",
        title="(1) Rendimiento en LFW test vs IOD (degradación MLBP+LDA)",
    )
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, 0.99), ncol=2, fontsize=6.8, framealpha=0.93)

    # ---------------- Panel 2: FTA desglosado + FAR/FRR al umbral fijo 0.2172
    x_idx = np.arange(len(xs))
    width = 0.62

    def fta_rate(k):
        return np.array([100.0 * sw_a[str(t)]["fta"]["seq_rates"][k] for t in xs])

    r_nof = fta_rate("NoFaceDetected")
    r_minf = fta_rate("MIN_FACE_PX")
    r_mine = fta_rate("MIN_EYE_DIST_PX")
    r_yaw = fta_rate("MAX_YAW_RATIO")
    r_amb = fta_rate("AmbiguousFace")
    r_face = np.array(
        [100.0 * sw_a[str(t)]["fta"]["marginal_counts"]["faceness_rejected_among_strict_ok"] / sw_a[str(t)]["fta"]["n_total"] for t in xs]
    )

    bottom = np.zeros(len(xs))
    stacks = [
        ("NoFaceDetected", r_nof, "#2c3e50"),
        ("MIN_FACE_PX < 40", r_minf, "#c0392b"),
        ("MIN_EYE_DIST_PX < 16", r_mine, "#e67e22"),
        ("MAX_YAW_RATIO > 0,45", r_yaw, "#9b59b6"),
        ("AmbiguousFace", r_amb, "#95a5a6"),
        ("Faceness", r_face, "#16a085"),
    ]
    for lbl, vals, col in stacks:
        ax2.bar(x_idx, vals, width, bottom=bottom, label=lbl, color=col, alpha=0.82)
        bottom += vals

    frr_nc = series("metrics_no_size_cut_with_faceness", "frr_at_calib_thr")
    far_nc = series("metrics_no_size_cut_with_faceness", "far_at_calib_thr")

    ax2.plot(x_idx, frr_nc, "o-", color="#1a5276", lw=2.2, label="FRR @ 0,2172 (sin corte, %)")
    ax2_r = ax2.twinx()
    ax2_r.plot(x_idx, far_nc, "s--", color="#d35400", lw=2.0, label="FAR @ 0,2172 (sin corte, %)")
    ax2_r.axhline(0.1, color="#d35400", ls=":", lw=1.0, alpha=0.7)
    ax2_r.set_ylabel("FAR @ umbral 0,2172 (%)", color="#d35400")
    ax2_r.tick_params(axis="y", colors="#d35400")
    ax2_r.set_ylim(0, max(0.35, float(np.nanmax(far_nc)) * 1.45))

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([str(t) for t in xs])
    ax2.set(
        ylim=(0, 118),
        xlabel="Distancia interocular objetivo (px)",
        ylabel="FTA / FRR (%)",
        title="(2) FTA desglosado y FAR/FRR al umbral fijo (0,2172)",
    )
    ax2.grid(axis="y", alpha=0.25)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.6, framealpha=0.93, ncol=2)

    # ---------------- Panel 3: Alta resolucion vinicius_junior (10..120 px)
    xb = HIRES_TARGETS
    sw_b = res_b["sweep"]
    v_asym_mean = np.array([sw_b[str(t)]["vinicius_loo"]["asym_mean"] for t in xb])
    v_asym_min = np.array([sw_b[str(t)]["vinicius_loo"]["asym_min"] for t in xb])
    v_asym_max = np.array([sw_b[str(t)]["vinicius_loo"]["asym_max"] for t in xb])
    v_sym_mean = np.array([sw_b[str(t)]["vinicius_loo"]["sym_mean"] for t in xb])
    imp_vini_mean = np.array([sw_b[str(t)]["gallery_impostors_49"]["vs_vinicius_mean"] for t in xb])
    hires_cos = np.array([sw_b[str(t)]["hires_ge120_self_similarity"]["cos_vs_120px_mean"] for t in xb])

    ax3.fill_between(xb, v_asym_min, v_asym_max, color=C_AZUL, alpha=0.16, label="Vinicius LOO mín–máx (asimétrico)")
    ax3.plot(xb, v_asym_mean, "o-", color=C_AZUL, lw=2.2, label="Vinicius LOO media (asimétrico vs BD)")
    ax3.plot(xb, v_sym_mean, "s--", color=C_CIAN, lw=1.7, label="Vinicius LOO media (simétrico)")
    ax3.plot(xb, imp_vini_mean, "^-", color=C_ROJO, lw=1.7, label="Impostores galería (49 IDs) vs Vinicius")
    ax3.plot(xb, hires_cos, "d:", color=C_VERDE, lw=1.8, label="Auto-similitud cos vs 120 px (galería ≥120 px)")

    ax3.axhline(CALIB_THR, color="#222222", ls="--", lw=1.2, label=f"Umbral verificación ({CALIB_THR:.4f})")
    ax3.axhline(IDENT_THR, color=C_NARANJA, ls="-.", lw=1.1, label=f"Umbral identificación ({IDENT_THR:.4f})")

    for xv, txt, col, ha, dx in (
        (config.MIN_EYE_DIST_PX, "16 px", "#444444", "left", 1.2),
        (INTERNAL_IOD, "48,6 px\n(saturación)", C_MORADO, "right", -1.2),
        (ISO_IOD, "60 px\n(ISO)", "#111111", "left", 1.2),
    ):
        ax3.axvline(xv, ls="--", lw=1.1, color=col, alpha=0.85)
        ax3.text(xv + dx, 0.98, txt, fontsize=7.4, color=col, va="top", ha=ha, fontweight="bold")

    ax3.set(
        xlim=(5, 125),
        ylim=(-0.05, 1.04),
        xlabel="Distancia interocular objetivo (px)",
        ylabel="Similitud coseno",
        title="(3) Saturación en alta resolución: Vinicius LOO y galería (10–120 px)",
    )
    ax3.grid(alpha=0.25)
    ax3.legend(loc="center right", bbox_to_anchor=(0.99, 0.57), fontsize=6.9, framealpha=0.93)

    # ---------------- Panel 4: Distribucion IOD natural (LFW vs Galeria) + Exp C
    lfw_iods = res_a["valid_nat_iods"]
    gal_iods = res_b["all_photo_iods"]
    bins = np.linspace(0, 215, 44)
    ax4.hist(lfw_iods, bins=bins, density=True, alpha=0.55, color=C_AZUL, label=f"LFW test (n={len(lfw_iods)}, med={np.median(lfw_iods):.1f} px)")
    ax4.hist(gal_iods, bins=bins, density=True, alpha=0.55, color=C_NARANJA, label=f"Galería data/fotos (n={len(gal_iods)}, med={np.median(gal_iods):.1f} px)")

    ylim_top = ax4.get_ylim()[1]
    for xv, txt, col, ha, dx in (
        (config.MIN_EYE_DIST_PX, "16 px", "#444444", "left", 1.5),
        (INTERNAL_IOD, "48,6 px", C_MORADO, "right", -1.5),
        (ISO_IOD, "60 px ISO", "#111111", "left", 1.5),
    ):
        ax4.axvline(xv, ls="--", lw=1.2, color=col, alpha=0.85)
        ax4.text(xv + dx, ylim_top * 0.95, txt, fontsize=7.4, color=col, va="top", ha=ha, fontweight="bold")

    ax4.set(
        xlim=(0, 215),
        xlabel="Distancia interocular natural (px)",
        ylabel="Densidad",
        title="(4) Distribución de IOD natural y estratificación en LFW test",
    )
    ax4.grid(alpha=0.25)
    ax4.legend(loc="upper right", fontsize=7.3, framealpha=0.92)

    ins = ax4.inset_axes([0.46, 0.28, 0.51, 0.43])
    band_keys = ["<35", "35-40", "40-45", "45-50", ">=50"]
    tars = [100.0 * (res_c["by_band"][k]["tar_at_calib_thr"] or 0.0) for k in band_keys]
    ns = [res_c["by_band"][k]["n_gen"] for k in band_keys]
    bars = ins.bar(np.arange(len(band_keys)), tars, color=C_VERDE, alpha=0.85, width=0.6)
    ins.set_xticks(np.arange(len(band_keys)))
    ins.set_xticklabels(band_keys, fontsize=7)
    ins.set_ylim(0, max(tars) * 1.40)
    ins.set_title("Exp. C: TAR (%) @ 0,2172 por IOD mín. del par", fontsize=7.8)
    ins.tick_params(axis="y", labelsize=7)
    ins.grid(axis="y", alpha=0.25)
    for b, val, n_p in zip(bars, tars, ns):
        ins.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.1f}%\n(n={n_p})", ha="center", va="bottom", fontsize=6.0)

    fig.suptitle(
        "BigBrother (MLBP + PCA/LDA): Análisis completo por distancia interocular (IOD)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    default_lfw = "/cache/lfw_home/lfw" if Path("/cache/lfw_home/lfw").exists() else os.path.expanduser("~/.cache/bigbrother/lfw_home/lfw")
    ap.add_argument("--lfw", default=default_lfw)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    t_start = time.time()
    res_a, res_c = run_experiment_a_and_c(args.lfw, args.workers)
    res_b = run_experiment_b()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "eval_interocular.png"
    make_figure(res_a, res_b, res_c, out_png)

    res_a_json = {k: v for k, v in res_a.items() if k != "valid_nat_iods"}
    res_b_json = {k: v for k, v in res_b.items() if k != "all_photo_iods"}

    payload = {
        "metadata": {
            "calib_threshold": CALIB_THR,
            "ident_threshold": IDENT_THR,
            "min_face_px": config.MIN_FACE_PX,
            "min_eye_dist_px": config.MIN_EYE_DIST_PX,
            "max_yaw_ratio": config.MAX_YAW_RATIO,
            "ambiguous_face_ratio": config.AMBIGUOUS_FACE_RATIO,
            "face_size": config.FACE_SIZE,
            "internal_eye_dist_px": INTERNAL_IOD,
            "iso_19794_5_min_iod_px": ISO_IOD,
            "lbp_radii": list(config.LBP_RADII),
            "grid": list(config.GRID),
            "photometric": config.PHOTOMETRIC,
            "total_runtime_s": round(time.time() - t_start, 2),
        },
        "experiment_a_lfw_sweep": res_a_json,
        "experiment_b_hires_gallery": res_b_json,
        "experiment_c_lfw_natural_stratification": res_c,
    }

    out_json = OUT_DIR / "eval_interocular.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK -> {out_json} & {out_png} ({time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
