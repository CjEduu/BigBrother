"""Evaluacion longitudinal y de cambios drasticos de apariencia en BigBrother.

    python scripts/eval_cambios_apariencia.py
    -> docs/resultados/eval_cambios_apariencia.json
    -> docs/resultados/eval_cambios_apariencia.png

Evalua el sistema real (BD de 50 identidades en data/db.csv) sobre las 45 fotos reales
catalogadas en data/pruebas_cambios/manifest.json sin modificar la BD, keys/, models/ ni
data/fotos/.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bigbrother import config  # noqa: E402
from bigbrother.dataset import FOTOS_DIR  # noqa: E402
from bigbrother.detector import check_quality  # noqa: E402
from bigbrother.errors import BigBrotherError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402
from bigbrother.preprocess import align  # noqa: E402
from bigbrother.system import BigBrother  # noqa: E402

PRUEBAS_DIR = ROOT / "data" / "pruebas_cambios"
MANIFEST_PATH = PRUEBAS_DIR / "manifest.json"
OUT_DIR = ROOT / "docs" / "resultados"
OUT_JSON = OUT_DIR / "eval_cambios_apariencia.json"
OUT_PNG = OUT_DIR / "eval_cambios_apariencia.png"

# Umbrales de referencia calibrados en LFW test (scripts/train_projection.py)
THR_VERIF = float(config.THRESHOLD)          # 0.217241 (FAR 0.1%)
THR_IDENT = float(config.IDENT_THRESHOLD)    # 0.284694 (FPIR 1%)
THR_FAR_1 = float(config._CAL.get("metrics", {}).get("thr@far0.01", 0.15703514))
THR_FAR_5 = 0.108374   # cuantil 95% de pares impostores LFW test (1M pares)
THR_FAR_10 = 0.083431  # cuantil 90% de pares impostores LFW test (1M pares)

CAT_ORDER = [
    "pigmentacion_cirugia",
    "transicion_genero",
    "envejecimiento",
    "apariencia_estilismo",
]
CAT_LABELS = {
    "pigmentacion_cirugia": "1. Pigmentación / cirugía",
    "transicion_genero": "2. Transición de género",
    "envejecimiento": "3. Envejecimiento extremo",
    "apariencia_estilismo": "4. Piercings / maquillaje / vello / peso",
}
CAT_COLORS = {
    "pigmentacion_cirugia": "#8e44ad",
    "transicion_genero": "#2980b9",
    "envejecimiento": "#d35400",
    "apariencia_estilismo": "#27ae60",
}


def face_geometry(face):
    """Calcula eye_dist_px y yaw_ratio de un objeto Face."""
    re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_d = float(np.linalg.norm(le - re))
    axis = (le - re) / (eye_d + 1e-12)
    yaw = abs(float(np.dot(nose - (le + re) / 2, axis))) / (eye_d + 1e-12)
    return eye_d, yaw


def eval_sample(bb, item):
    img_path = ROOT / item["path"]
    img = load_image(img_path)
    id_real = item["id"]

    # 1. Deteccion YuNet y metricas geometricas
    faces = bb.detector.detect_all(img)
    n_faces = len(faces)
    det_score = float(faces[0].score) if n_faces else None
    bbox_w = float(faces[0].bbox[2]) if n_faces else None
    bbox_h = float(faces[0].bbox[3]) if n_faces else None
    eye_dist_px, yaw_ratio = face_geometry(faces[0]) if n_faces else (None, None)

    detector_status = "OK"
    detector_reason = None
    try:
        crop = bb.recortar(img)
    except BigBrotherError as e:
        detector_status = type(e).__name__
        detector_reason = str(e)
        crop = None

    rec = {
        **item,
        "detector_status": detector_status,
        "detector_reason": detector_reason,
        "n_faces": n_faces,
        "detector_score": round(det_score, 4) if det_score is not None else None,
        "bbox_px": [round(bbox_w, 1), round(bbox_h, 1)] if bbox_w is not None else None,
        "eye_dist_px": round(eye_dist_px, 2) if eye_dist_px is not None else None,
        "yaw_ratio": round(yaw_ratio, 4) if yaw_ratio is not None else None,
    }

    if crop is not None:
        # 2. API publica
        verif_pub = bool(bb.verificar(id_real, img))
        ident_pub = bb.identificar(img)

        # 3. Diagnostico interno (caja blanca)
        t = bb.extraer_sd(crop)
        faceness_val = float(bb.faceness.score(t))
        y = bb._proj(t)
        scores = bb._scores(y)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        sim_real = float(scores[id_real])
        rank_real = [k for k, _ in ordered].index(id_real) + 1
        cross_fp = [(k, round(float(v), 4)) for k, v in ordered if k != id_real and v >= THR_VERIF]

        rec.update({
            "verificar_publico": verif_pub,
            "identificar_publico": ident_pub,
            "identificacion_correcta": bool(ident_pub == id_real),
            "faceness_score": round(faceness_val, 4),
            "sim_id_real": round(sim_real, 4),
            "margen_verif": round(sim_real - THR_VERIF, 4),
            "margen_ident": round(sim_real - THR_IDENT, 4),
            "rank_id_real": rank_real,
            "top3": [(k, round(float(v), 4)) for k, v in ordered[:3]],
            "top1_id": ordered[0][0],
            "top1_sim": round(float(ordered[0][1]), 4),
            "cross_fp": cross_fp,
            "n_cross_fp": len(cross_fp),
            "pasa_far_01pct": bool(sim_real >= THR_VERIF),
            "pasa_far_1pct": bool(sim_real >= THR_FAR_1),
            "pasa_far_5pct": bool(sim_real >= THR_FAR_5),
            "pasa_far_10pct": bool(sim_real >= THR_FAR_10),
            "nivel_impostor": bool(sim_real < 0.10),
            "sim_sin_filtro_calidad": round(sim_real, 4),
        })
    else:
        # Diagnostico adicional si hay al menos una cara detectada (aunque rechazada por calidad/ambiguedad)
        sim_unfiltered = None
        rank_unfiltered = None
        top1_unfiltered = None
        if n_faces > 0:
            try:
                crop_u = align(img, faces[0].landmarks)
                t_u = bb.extraer_sd(crop_u)
                y_u = bb._proj(t_u)
                sc_u = bb._scores(y_u)
                ord_u = sorted(sc_u.items(), key=lambda kv: -kv[1])
                sim_unfiltered = round(float(sc_u[id_real]), 4)
                rank_unfiltered = [k for k, _ in ord_u].index(id_real) + 1
                top1_unfiltered = (ord_u[0][0], round(float(ord_u[0][1]), 4))
            except BigBrotherError:
                pass

        rec.update({
            "verificar_publico": None,
            "identificar_publico": None,
            "identificacion_correcta": False,
            "faceness_score": None,
            "sim_id_real": None,
            "margen_verif": None,
            "margen_ident": None,
            "rank_id_real": None,
            "top3": None,
            "top1_id": None,
            "top1_sim": None,
            "cross_fp": [],
            "n_cross_fp": 0,
            "pasa_far_01pct": False,
            "pasa_far_1pct": False,
            "pasa_far_5pct": False,
            "pasa_far_10pct": False,
            "nivel_impostor": None,
            "sim_sin_filtro_calidad": sim_unfiltered,
            "rank_sin_filtro_calidad": rank_unfiltered,
            "top1_sin_filtro_calidad": top1_unfiltered,
        })
    return rec


def diagnose_elisabeth_enrollment(bb, elisabeth_records):
    """Diagnostico especial de la inscripcion de elisabeth_II_1945.jpg:
    YuNet detecta dos regiones: faces[0] (mayor area, falso positivo en el neumatico/mano)
    y faces[1] (la cara real de Isabel II en 1945). Como area(1)/area(0) = 0.304 < 0.45,
    no salta AmbiguousFace y la BD real contiene el neumatico como template de elisabeth_ii.
    """
    enr_path = FOTOS_DIR / "dataset" / "elisabeth_II_1945.jpg"
    img = load_image(enr_path)
    faces = bb.detector.detect_all(img)
    info = {
        "foto_inscripcion": str(enr_path.relative_to(ROOT)),
        "imagen_px": [int(img.shape[1]), int(img.shape[0])],
        "detecciones_yunet": [
            {
                "idx": i,
                "bbox": [round(v, 1) for v in f.bbox],
                "area": round(f.area, 1),
                "score": round(f.score, 4),
            }
            for i, f in enumerate(faces)
        ],
    }
    if len(faces) >= 2:
        info["ratio_area_f1_f0"] = round(faces[1].area / faces[0].area, 4)
        t_tire = bb.extraer_sd(align(img, faces[0].landmarks))
        t_face = bb.extraer_sd(align(img, faces[1].landmarks))
        info["faceness_f0_neumatico"] = round(float(bb.faceness.score(t_tire)), 4)
        info["faceness_f1_cara_real"] = round(float(bb.faceness.score(t_face)), 4)
        y_tire = bb.projection(t_tire)
        y_face = bb._proj(t_face)
        db_vec = bb.db.get("elisabeth_ii")[0]
        info["sim_db_con_f0_neumatico"] = round(float(np.dot(db_vec, y_tire)), 4)
        info["sim_db_con_f1_cara_real"] = round(float(np.dot(db_vec, y_face)), 4)

        re_eval = []
        for rec in elisabeth_records:
            p_img = load_image(ROOT / rec["path"])
            try:
                y_p = bb._proj(bb.extraer(p_img))
            except BigBrotherError:
                continue
            sim_real_1945 = float(np.dot(y_p, y_face))
            re_eval.append({
                "label": rec["label"],
                "probe_year": rec["probe_year"],
                "sim_contra_bd_real_neumatico": rec["sim_id_real"],
                "sim_contra_cara_real_1945": round(sim_real_1945, 4),
                "verificaria_con_cara_real_far01": bool(sim_real_1945 >= THR_VERIF),
                "verificaria_con_cara_real_far1": bool(sim_real_1945 >= THR_FAR_1),
                "verificaria_con_cara_real_far5": bool(sim_real_1945 >= THR_FAR_5),
            })
        info["comparativa_pruebas"] = re_eval
    return info


def summarize_group(rows):
    n = len(rows)
    ok = [r for r in rows if r["detector_status"] == "OK"]
    fta = [r for r in rows if r["detector_status"] != "OK"]
    sims = [r["sim_id_real"] for r in ok]
    ranks = [r["rank_id_real"] for r in ok]
    return {
        "n_total": n,
        "n_detectadas_ok": len(ok),
        "n_fta": len(fta),
        "fta_rate": round(len(fta) / n, 4) if n else 0.0,
        "verif_ok_total": sum(1 for r in ok if r["verificar_publico"]),
        "tar_sobre_total": round(sum(1 for r in ok if r["verificar_publico"]) / n, 4) if n else 0.0,
        "tar_sobre_detectadas": round(sum(1 for r in ok if r["verificar_publico"]) / len(ok), 4) if ok else 0.0,
        "ident_ok": sum(1 for r in ok if r["identificacion_correcta"]),
        "rank1_count": sum(1 for r in ok if r["rank_id_real"] == 1),
        "rank1_rate_detectadas": round(sum(1 for r in ok if r["rank_id_real"] == 1) / len(ok), 4) if ok else 0.0,
        "pasa_far_1pct": sum(1 for r in ok if r["pasa_far_1pct"]),
        "pasa_far_5pct": sum(1 for r in ok if r["pasa_far_5pct"]),
        "pasa_far_10pct": sum(1 for r in ok if r["pasa_far_10pct"]),
        "nivel_impostor_count": sum(1 for r in ok if r["nivel_impostor"]),
        "cross_fp_probes": sum(1 for r in ok if r["n_cross_fp"] > 0),
        "cross_fp_total_matches": sum(r["n_cross_fp"] for r in ok),
        "sim_media": round(float(np.mean(sims)), 4) if sims else None,
        "sim_mediana": round(float(np.median(sims)), 4) if sims else None,
        "sim_min": round(float(np.min(sims)), 4) if sims else None,
        "sim_max": round(float(np.max(sims)), 4) if sims else None,
        "rank_medio": round(float(np.mean(ranks)), 2) if ranks else None,
    }


def make_figure(records, by_cat):
    # Ordenar: por categoria (CAT_ORDER), dentro por id y probe_year
    cat_rank = {c: i for i, c in enumerate(CAT_ORDER)}
    ordered = sorted(records, key=lambda r: (cat_rank[r["category"]], r["id"], r["probe_year"], r["label"]))

    fig = plt.figure(figsize=(17.0, 14.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.40)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --- Panel izquierdo: barras horizontales por foto ---
    n = len(ordered)
    y_pos = np.arange(n)[::-1]

    # Zonas de fondo
    ax.axvspan(-0.10, 0.10, color="#fbeaea", alpha=0.55, zorder=0)
    ax.axvspan(THR_VERIF, 0.90, color="#eaf7ee", alpha=0.55, zorder=0)

    prev_cat = None
    for idx, (r, y) in enumerate(zip(ordered, y_pos)):
        cat = r["category"]
        col = CAT_COLORS[cat]
        if prev_cat is not None and cat != prev_cat:
            ax.axhline(y + 0.5, color="#bbbbbb", lw=0.8, ls="-", zorder=1)
        prev_cat = cat

        sim = r["sim_id_real"]
        if sim is not None:
            alpha = 0.92 if r["verificar_publico"] else 0.72
            edge = "#111111" if r["verificar_publico"] else "none"
            ax.barh(y, sim, height=0.68, color=col, alpha=alpha, edgecolor=edge, lw=0.9, zorder=3)
            # Marcador adicional si hubo falso positivo cruzado
            if r["n_cross_fp"] > 0:
                ax.scatter([sim], [y], marker="D", s=24, color="#c0392b", edgecolor="white", lw=0.6, zorder=5)
            # Etiqueta derecha con resultado y rank
            marks = []
            if r["identificacion_correcta"]:
                marks.append("IDENT✓")
            elif r["verificar_publico"]:
                marks.append("VERIF✓")
            else:
                marks.append("RECHAZ✗")
            marks.append(f"#{r['rank_id_real']}")
            if r["rank_id_real"] != 1:
                marks.append(f"(top1:{r['top1_id'][:11]} {r['top1_sim']:.2f})")
            if r["n_cross_fp"] > 0:
                marks.append(f"[FPx:{r['n_cross_fp']}]")
            x_txt = max(sim, 0.0) + 0.009
            ax.text(x_txt, y, " ".join(marks), va="center", ha="left", fontsize=6.4,
                    color="#1a1a1a" if r["verificar_publico"] else "#444444",
                    fontweight="bold" if r["verificar_publico"] else "normal")
        else:
            # FTA en YuNet
            sim_u = r.get("sim_sin_filtro_calidad")
            if sim_u is not None:
                ax.plot([sim_u], [y], marker="x", ms=5.5, color=col, mew=1.4, zorder=4)
            ax.text(0.01, y, f"[FTA: {r['detector_status']}]" + (f" (sin filtro: {sim_u:.2f})" if sim_u is not None else ""),
                    va="center", ha="left", fontsize=6.4, color="#b03a2e", style="italic")

    # Lineas de umbral
    ax.axvline(0.0, color="#666666", lw=0.8, zorder=2)
    ax.axvline(THR_FAR_5, color="#7f8c8d", ls=":", lw=1.1, zorder=4)
    ax.axvline(THR_FAR_1, color="#e67e22", ls="-.", lw=1.2, zorder=4)
    ax.axvline(THR_VERIF, color="#c0392b", ls="--", lw=1.5, zorder=4)
    ax.axvline(THR_IDENT, color="#1e8449", ls="-", lw=1.5, zorder=4)

    # Etiquetas de umbrales arriba
    y_top = n - 0.1
    ax.text(-0.085, y_top, "Zona impostor\n(<0,10)", fontsize=6.3, color="#922b21", va="bottom", ha="left")
    ax.text(THR_FAR_5, y_top, f"FAR 5%\n({THR_FAR_5:.3f})", fontsize=6.2, color="#555555", ha="center", va="bottom")
    ax.text(THR_FAR_1, y_top, f"FAR 1%\n({THR_FAR_1:.3f})", fontsize=6.2, color="#b9770e", ha="center", va="bottom")
    ax.text(THR_VERIF, y_top, f"Verif.\n({THR_VERIF:.3f})", fontsize=6.4, color="#922b21", ha="center", va="bottom", fontweight="bold")
    ax.text(THR_IDENT, y_top, f"Ident.\n({THR_IDENT:.3f})", fontsize=6.4, color="#196f3d", ha="center", va="bottom", fontweight="bold")

    ylabels = []
    for r in ordered:
        dy = r["delta_years"]
        dy_s = f"{dy:+d}a" if dy != 0 else "0a"
        lbl_short = r["label"].replace("_", " ")
        ylabels.append(f"{r['id']} · {lbl_short} ({r['probe_year']}, {dy_s})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=7.0)
    ax.set_xlim(-0.095, 0.85)
    ax.set_ylim(-0.8, n + 1.4)
    ax.set_xlabel("Similitud coseno contra el template inscrito de su identidad real", fontsize=9)
    ax.set_title("Evaluación por caso (45 fotos reales · BD de 50 identidades)", fontsize=10.5, pad=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)

    legend_elems = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABELS[c]) for c in CAT_ORDER] + [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b", ms=6, label="Falso positivo cruzado (≥0,2172)"),
        Line2D([0], [0], marker="x", color="#555555", lw=0, ms=6, mew=1.4, label="Rechazada en YuNet (score sin filtro)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=6.7, framealpha=0.94)

    # --- Panel derecho: resumen por categoria ---
    cats_rev = CAT_ORDER[::-1]
    yc = np.arange(len(cats_rev))
    h = 0.21
    tar_01 = [by_cat[c]["tar_sobre_detectadas"] * 100 for c in cats_rev]
    tar_1 = [100 * by_cat[c]["pasa_far_1pct"] / max(by_cat[c]["n_detectadas_ok"], 1) for c in cats_rev]
    tar_5 = [100 * by_cat[c]["pasa_far_5pct"] / max(by_cat[c]["n_detectadas_ok"], 1) for c in cats_rev]
    r1_rate = [by_cat[c]["rank1_rate_detectadas"] * 100 for c in cats_rev]

    ax2.barh(yc + 1.5 * h, tar_01, height=h, color="#c0392b", label="TAR @FAR 0,1% (umbral BigBrother)")
    ax2.barh(yc + 0.5 * h, tar_1, height=h, color="#e67e22", label="TAR @FAR 1% (0,1570)")
    ax2.barh(yc - 0.5 * h, tar_5, height=h, color="#7f8c8d", label="TAR @FAR 5% (0,1084)")
    ax2.barh(yc - 1.5 * h, r1_rate, height=h, color="#2980b9", label="Rank-1 entre 50 identidades")

    for i, c in enumerate(cats_rev):
        s = by_cat[c]
        for off, val in ((1.5 * h, tar_01[i]), (0.5 * h, tar_1[i]), (-0.5 * h, tar_5[i]), (-1.5 * h, r1_rate[i])):
            ax2.text(val + 1.5, yc[i] + off, f"{val:.0f}%", va="center", ha="left", fontsize=7.0)
        ax2.text(2, yc[i] + 2.22 * h,
                 f"n={s['n_detectadas_ok']}/{s['n_total']} OK · sim. media={s['sim_media']:.3f} · FPx={s['cross_fp_probes']}",
                 fontsize=7.0, color="#333333", fontweight="bold")

    short_cat_labels = {
        "pigmentacion_cirugia": "1. Pigmentación\ny cirugía",
        "transicion_genero": "2. Transición\nde género",
        "envejecimiento": "3. Envejecimiento\nextremo",
        "apariencia_estilismo": "4. Piercings, maq.,\nvello y peso",
    }
    ax2.set_yticks(yc)
    ax2.set_yticklabels([short_cat_labels[c] for c in cats_rev], fontsize=8.3)
    ax2.set_xlim(0, 118)
    ax2.set_ylim(-0.6, len(cats_rev) - 0.05)
    ax2.set_xlabel("Porcentaje sobre caras detectadas OK (%)", fontsize=9)
    ax2.set_title("Resumen por categoría de cambio", fontsize=10.5, pad=10, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7.0, framealpha=0.94)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("BigBrother (MLBP + PCA/LDA): evaluación longitudinal y frente a cambios drásticos de apariencia",
                 fontsize=12.5, fontweight="bold", y=0.935)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bb = BigBrother()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    records = []
    for item in manifest:
        rec = eval_sample(bb, item)
        records.append(rec)
        print(f"{rec['id']:18s} | {rec['label']:24s} | det={rec['detector_status']:14s} | "
              f"sim={str(rec['sim_id_real']):>7s} | rank={str(rec['rank_id_real']):>3s} | "
              f"verif={str(rec['verificar_publico']):>5s} | ident={str(rec['identificar_publico'])}",
              flush=True)

    by_cat = {}
    for c in CAT_ORDER:
        by_cat[c] = summarize_group([r for r in records if r["category"] == c])

    by_id = {}
    for pid in sorted({r["id"] for r in records}):
        by_id[pid] = summarize_group([r for r in records if r["id"] == pid])

    elisa_diag = diagnose_elisabeth_enrollment(bb, [r for r in records if r["id"] == "elisabeth_ii"])

    summary_all = summarize_group(records)
    # Resumen adicional excluyendo el fallo de inscripcion de elisabeth_ii
    summary_sin_elisa = summarize_group([r for r in records if r["id"] != "elisabeth_ii"])

    out = {
        "thresholds": {
            "verificacion_far_0_1pct": THR_VERIF,
            "identificacion_fpir_1pct": THR_IDENT,
            "far_1pct_lfw": THR_FAR_1,
            "far_5pct_lfw": THR_FAR_5,
            "far_10pct_lfw": THR_FAR_10,
            "faceness_threshold": float(bb.faceness.threshold),
        },
        "n_enrolled_ids": len(bb.identidades()),
        "resumen_global": summary_all,
        "resumen_sin_elisabeth_ii": summary_sin_elisa,
        "resumen_por_categoria": by_cat,
        "resumen_por_identidad": by_id,
        "elisabeth_ii_diagnostico_inscripcion": elisa_diag,
        "resultados": records,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figure(records, by_cat)
    print("\nRESUMEN GLOBAL:", json.dumps(summary_all, indent=2, ensure_ascii=False))
    print("Guardado:", OUT_JSON, "y", OUT_PNG)


if __name__ == "__main__":
    main()
