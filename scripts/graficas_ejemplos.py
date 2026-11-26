"""Genera montajes visuales (hojas de ejemplos reales) para los informes de BigBrother.

Cada figura combina fotos reales de inscripcion, sondas de prueba, transformaciones de
robustez/PAD, gemelos, parecidos de LFW, degradacion de IOD y ataques (morphing/dodging),
anotadas con las puntuaciones y decisiones exactas del pipeline real.

Ejecucion reproducible dentro de bigbrother:eval:
    docker run --rm --entrypoint python \
      -v /home/cj/.cache/bigbrother:/cache:ro \
      -v "$(pwd)/scripts:/app/scripts:ro" \
      -v "$(pwd)/docs:/app/docs" \
      -v "$(pwd)/data/pruebas_cambios:/app/data/pruebas_cambios:ro" \
      -v "$(pwd)/data/ataques:/app/data/ataques:ro" \
      bigbrother:eval scripts/graficas_ejemplos.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from bigbrother import BigBrother, config
from bigbrother.errors import BigBrotherError
from bigbrother.imageio import load_image
from bigbrother.preprocess import align, normalize_face

ROOT = Path(__file__).resolve().parent.parent
FOTOS = ROOT / "data" / "fotos"
DATASET = FOTOS / "dataset"
VINI_DIR = FOTOS / "vinicius_junior"
PAULA_DIR = ROOT / "data" / "ataques" / "gemela_paula"
CAMBIOS = ROOT / "data" / "pruebas_cambios"
OUT = ROOT / "docs" / "resultados"
CACHE = Path(os.environ.get("BIGBROTHER_CACHE", "/cache"))
LFW_DIR = CACHE / "lfw_home" / "lfw"

THR_VERIF = config.THRESHOLD
THR_IDENT = config.IDENT_THRESHOLD

# Paleta de bordes y etiquetas
COL_ENROLL = "#1f618d"   # azul: foto de inscripcion en BD
COL_OK = "#1e8449"       # verde: verificada / identificada ok
COL_FAIL = "#c0392b"     # rojo: rechazada / fallo / ataque exitoso
COL_WARN = "#d35400"     # naranja: impostor aceptado / advertencia / ranking sin umbral
COL_NEUTRAL = "#566573"  # gris: referencia neutra


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def portrait_crop(img: np.ndarray, bb: BigBrother | None = None,
                  size: int = 220, pad: float = 0.55) -> np.ndarray:
    """Recorte cuadrado centrado en la cara (con contexto de pelo/gafas/barba)."""
    h, w = img.shape[:2]
    cx, cy, side = w / 2.0, h / 2.0, min(h, w) * 0.85
    if bb is not None:
        try:
            faces = bb.detector.detect_all(img)
            if faces:
                # filtrar con faceness si hay varias detecciones (como en recortar)
                if len(faces) > 1:
                    valid = []
                    for f in faces:
                        try:
                            t = bb.extractor.extract(normalize_face(align(img, f.landmarks)))
                            if bb.faceness.is_face(t):
                                valid.append(f)
                        except Exception:
                            pass
                    if valid:
                        faces = valid
                f = faces[0]
                x, y, bw, bh = f.bbox
                cx = x + bw / 2.0
                cy = y + bh * 0.45
                side = max(bw, bh) * (1.0 + 2.0 * pad)
        except Exception:
            pass
    half = side / 2.0
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))
    # Pad con gris oscuro si sale fuera de la imagen
    pl = max(0, -x1)
    pt = max(0, -y1)
    pr = max(0, x2 - w)
    pb = max(0, y2 - h)
    if pl or pt or pr or pb:
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(28, 28, 32))
        x1 += pl
        x2 += pl
        y1 += pt
        y2 += pt
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def find_enrolled_photo(person_id: str) -> Path:
    if person_id == "vinicius_junior":
        return VINI_DIR / "vini_01.jpg"
    for p in sorted(DATASET.iterdir()):
        if p.name.lower().startswith(person_id.lower()):
            return p
    raise FileNotFoundError(f"No enrolled photo for {person_id}")


def draw_tile(ax, img_bgr: np.ndarray, title: str, subtitle: str,
              border_color: str = COL_NEUTRAL, badge: str | None = None,
              badge_color: str | None = None, title_size: float = 8.2,
              sub_size: float = 7.4):
    ax.imshow(bgr_to_rgb(img_bgr))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(2.8)
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=4, color="#1a1a1a")
    ax.set_xlabel(subtitle, fontsize=sub_size, labelpad=3, color="#222222")
    if badge:
        ax.text(
            0.04, 0.94, badge,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=6.9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=badge_color or border_color, edgecolor="none", alpha=0.92),
        )


def _find_rec(recs: list[dict], pid: str, label: str) -> dict:
    for r in recs:
        if r["id"] == pid and r["label"] == label:
            return r
    raise KeyError(f"{pid}/{label}")


# ============================================================================
# 1. Cambios de apariencia — Parte 1: Michael Jackson, Envejecimiento y Elisabeth II
# ============================================================================
def fig_cambios_mj_envejecimiento(bb: BigBrother, cam: dict):
    recs = cam["resultados"]
    fig = plt.figure(figsize=(15.5, 9.6), dpi=160)
    gs = fig.add_gridspec(3, 7, hspace=0.52, wspace=0.14, top=0.90, bottom=0.05, left=0.02, right=0.98)

    # --- Fila 1: Michael Jackson (1982 inscrita + 6 sondas) ---
    mj_enr = portrait_crop(load_image(str(find_enrolled_photo("michael_jackson"))), bb)
    draw_tile(fig.add_subplot(gs[0, 0]), mj_enr,
              "Michael Jackson (1982)", "Inscrita en BD (Thriller)",
              border_color=COL_ENROLL, badge="INSCRITA", badge_color=COL_ENROLL)

    mj_labels = ["1974_joven", "1977_joven", "1988_bad", "1992_dangerous", "1997_cannes", "2005_licencia"]
    for idx, lbl in enumerate(mj_labels, start=1):
        r = _find_rec(recs, "michael_jackson", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        ok = bool(r["verificar_publico"])
        col = COL_OK if ok else COL_FAIL
        badge = f"VERIF ✓ #{r['rank_id_real']}" if ok else f"RECHAZ ✗ #{r['rank_id_real']}"
        dy = f"{r['delta_years']:+d} a"
        draw_tile(fig.add_subplot(gs[0, idx]), im,
                  f"{r['probe_year']} ({dy})",
                  f"sim = {r['sim_id_real']:.3f}",
                  border_color=col, badge=badge)

    # --- Fila 2: Envejecimiento natural (Jordi Hurtado) vs Cirugía (Leticia Sabater) ---
    jh_enr = portrait_crop(load_image(str(find_enrolled_photo("jordi_hurtado"))), bb)
    draw_tile(fig.add_subplot(gs[1, 0]), jh_enr,
              "Jordi Hurtado (1985)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    for idx, lbl in enumerate(["2017_maduro", "2025_rtve"], start=1):
        r = _find_rec(recs, "jordi_hurtado", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        ok = bool(r["verificar_publico"])
        col = COL_OK if ok else COL_WARN
        badge = f"VERIF ✓ #{r['rank_id_real']}" if ok else f"RECHAZ ✗ #{r['rank_id_real']}"
        draw_tile(fig.add_subplot(gs[1, idx]), im,
                  f"{r['probe_year']} (+{r['delta_years']} a)",
                  f"sim = {r['sim_id_real']:.3f}",
                  border_color=col, badge=badge)

    # Separador visual sutil usando columna 3 como texto o relleno? Mejor meter 3 + 3 + 1 vacio con nota
    ax_mid = fig.add_subplot(gs[1, 3])
    ax_mid.axis("off")
    ax_mid.text(0.5, 0.5, "Envejecimiento\ngradual (+40 a)\n\nvs.\n\nCirugía estética\nmúltiple (+28 a)",
                ha="center", va="center", fontsize=8.8, fontweight="bold", color="#34495e",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f7", edgecolor="#bdc3c7"))

    ls_enr = portrait_crop(load_image(str(find_enrolled_photo("leticia_sabater"))), bb)
    draw_tile(fig.add_subplot(gs[1, 4]), ls_enr,
              "Leticia Sabater (1996)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    for idx, lbl in enumerate(["2024_logrono", "2025_pride"], start=5):
        r = _find_rec(recs, "leticia_sabater", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        draw_tile(fig.add_subplot(gs[1, idx]), im,
                  f"{r['probe_year']} (+{r['delta_years']} a)",
                  f"sim = {r['sim_id_real']:.3f}",
                  border_color=COL_FAIL, badge=f"IMPOSTOR #{r['rank_id_real']}")

    # --- Fila 3: Defecto resuelto de elisabeth_ii (1945 foto con cajas + neumático vs cara real + 4 sondas) ---
    el_path = find_enrolled_photo("elisabeth_ii")
    el_full = load_image(str(el_path))
    faces = bb.detector.detect_all(el_full)
    f0, f1 = faces[0], faces[1]
    # Recorte de contexto que engloba ambas cajas (cara arriba + neumatico abajo)
    x_min = max(0, int(min(f0.bbox[0], f1.bbox[0]) - 260))
    x_max = min(el_full.shape[1], int(max(f0.bbox[0] + f0.bbox[2], f1.bbox[0] + f1.bbox[2]) + 260))
    y_min = max(0, int(min(f0.bbox[1], f1.bbox[1]) - 180))
    y_max = min(el_full.shape[0], int(max(f0.bbox[1] + f0.bbox[3], f1.bbox[1] + f1.bbox[3]) + 180))
    ctx = el_full[y_min:y_max, x_min:x_max].copy()
    # Dibujar bboxes en ctx
    for f, color, lbl_box in [(f0, (40, 40, 220), "f0: neumatico"), (f1, (40, 175, 55), "f1: cara real")]:
        bx, by, bw, bh = [int(v) for v in f.bbox]
        cv2.rectangle(ctx, (bx - x_min, by - y_min), (bx + bw - x_min, by + bh - y_min), color, 16)
    ctx_sq = cv2.resize(ctx, (220, 220), interpolation=cv2.INTER_AREA)

    ax_ctx = fig.add_subplot(gs[2, 0])
    draw_tile(ax_ctx, ctx_sq,
              "elisabeth_II_1945.jpg", "2 detecciones YuNet",
              border_color=COL_ENROLL, badge="ORIGEN")

    crop_tire = align(el_full, f0.landmarks)
    crop_face = align(el_full, f1.landmarks)
    diag = cam["elisabeth_ii_diagnostico_inscripcion"]
    draw_tile(fig.add_subplot(gs[2, 1]), crop_tire,
              "f0: Neumático (ANTES)", f"faceness = {diag['faceness_f0_neumatico']:.2f}",
              border_color=COL_FAIL, badge="DESCARTADA ✗")
    draw_tile(fig.add_subplot(gs[2, 2]), crop_face,
              "f1: Cara real (AHORA)", f"faceness = +{diag['faceness_f1_cara_real']:.1f}",
              border_color=COL_OK, badge="INSCRITA ✓")

    for idx, lbl in enumerate(["1959_joven", "1986_madura", "1992_madura", "2015_anciana"], start=3):
        r = _find_rec(recs, "elisabeth_ii", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        is_r1 = r["rank_id_real"] == 1
        col = COL_WARN if is_r1 else COL_FAIL
        badge = f"RANK #1 ({r['sim_id_real']:.2f})" if is_r1 else f"RECHAZ ✗ #{r['rank_id_real']}"
        draw_tile(fig.add_subplot(gs[2, idx]), im,
                  f"{r['probe_year']} (+{r['delta_years']} a)",
                  f"sim = {r['sim_id_real']:.3f}",
                  border_color=col, badge=badge)

    fig.suptitle(
        "Ejemplos visuales reales — Pigmentación/cirugía (Michael Jackson), envejecimiento y arreglo de elisabeth_ii\n"
        f"(Umbral verificación = {THR_VERIF:.4f} · Identificación = {THR_IDENT:.4f} · BD de 50 identidades)",
        fontsize=12.2, fontweight="bold", y=0.975,
    )
    out = OUT / "ejemplos_cambios_mj_envejecimiento.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


# ============================================================================
# 2. Cambios de apariencia — Parte 2: Transición, Peso/Barba/Multi-enroll y Oclusiones
# ============================================================================
def fig_cambios_transicion_estilo(bb: BigBrother, cam: dict):
    recs = cam["resultados"]
    fig = plt.figure(figsize=(15.5, 9.6), dpi=160)
    gs = fig.add_gridspec(3, 7, hspace=0.52, wspace=0.14, top=0.90, bottom=0.05, left=0.02, right=0.98)

    # --- Fila 1: Transición de género (Ellen -> Elliot Page + Josie Totah) ---
    ep_enr = portrait_crop(load_image(str(find_enrolled_photo("ellen_page"))), bb)
    draw_tile(fig.add_subplot(gs[0, 0]), ep_enr,
              "Ellen Page (2006)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    for idx, (lbl, tag) in enumerate([
        ("2015_control_pre", "2015 pre (+9 a)"),
        ("2021_elliot_post", "2021 Elliot (+15 a)"),
        ("2023_elliot_post", "2023 Elliot (+17 a)"),
    ], start=1):
        r = _find_rec(recs, "ellen_page", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        is_r1 = r["rank_id_real"] == 1
        col = COL_WARN if is_r1 else COL_FAIL
        badge = f"RANK #1" if is_r1 else f"RECHAZ ✗ #{r['rank_id_real']}"
        draw_tile(fig.add_subplot(gs[0, idx]), im, tag,
                  f"sim = {r['sim_id_real']:.3f}",
                  border_color=col, badge=badge)

    jt_enr = portrait_crop(load_image(str(find_enrolled_photo("josie_totah"))), bb)
    draw_tile(fig.add_subplot(gs[0, 4]), jt_enr,
              "Josie Totah (2017)", "Inscrita (pre-trans.)",
              border_color=COL_ENROLL, badge="INSCRITA")
    for idx, (lbl, tag) in enumerate([
        ("2023_tmdb_post", "2023 post (+6 a)"),
        ("2025_post_transicion", "2025 post (+8 a)"),
    ], start=5):
        r = _find_rec(recs, "josie_totah", lbl)
        im = portrait_crop(load_image(str(ROOT / r["path"])), bb)
        ok = bool(r["verificar_publico"])
        col = COL_OK if ok else COL_FAIL
        badge = f"VERIF ✓ #{r['rank_id_real']}" if ok else f"RECHAZ ✗ #{r['rank_id_real']}"
        draw_tile(fig.add_subplot(gs[0, idx]), im, tag,
                  f"sim={r['sim_id_real']:.3f} (ident:{r['identificar_publico']})",
                  border_color=col, badge=badge, sub_size=6.5)

    # --- Fila 2: Qué SÍ tolera (Adele peso, Keanu barba, Vinícius multi-inscripción) ---
    ad_enr = portrait_crop(load_image(str(find_enrolled_photo("adele"))), bb)
    draw_tile(fig.add_subplot(gs[1, 0]), ad_enr, "Adele (2016)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    r_ad = _find_rec(recs, "adele", "2023_perdida_peso")
    draw_tile(fig.add_subplot(gs[1, 1]), portrait_crop(load_image(str(ROOT / r_ad["path"])), bb),
              "Adele 2023 (peso)", f"sim = {r_ad['sim_id_real']:.3f}",
              border_color=COL_OK, badge="IDENT ✓ #1")

    kr_enr = portrait_crop(load_image(str(find_enrolled_photo("keanu_reeves"))), bb)
    draw_tile(fig.add_subplot(gs[1, 2]), kr_enr, "Keanu (Matrix 1999)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    r_kr = _find_rec(recs, "keanu_reeves", "2019_barba_larga")
    draw_tile(fig.add_subplot(gs[1, 3]), portrait_crop(load_image(str(ROOT / r_kr["path"])), bb),
              "Keanu 2019 (+20 a)", f"sim = {r_kr['sim_id_real']:.3f} (barba)",
              border_color=COL_OK, badge="IDENT ✓ #1")

    vn_enr = portrait_crop(load_image(str(VINI_DIR / "vini_01.jpg")), bb)
    draw_tile(fig.add_subplot(gs[1, 4]), vn_enr, "Vinícius (5 fotos BD)", "Multi-inscripción",
              border_color=COL_ENROLL, badge="5 FOTOS BD")
    r_v18 = _find_rec(recs, "vinicius_junior", "2018_joven")
    draw_tile(fig.add_subplot(gs[1, 5]), portrait_crop(load_image(str(ROOT / r_v18["path"])), bb),
              "Vinícius 2018 (−6 a)", f"sim = {r_v18['sim_id_real']:.3f} (32px)",
              border_color=COL_OK, badge="VERIF ✓ #1")
    r_v25 = _find_rec(recs, "vinicius_junior", "2025_reciente")
    draw_tile(fig.add_subplot(gs[1, 6]), portrait_crop(load_image(str(ROOT / r_v25["path"])), bb),
              "Vinícius 2025 (trenzas)", f"sim = {r_v25['sim_id_real']:.3f}",
              border_color=COL_OK, badge="IDENT ✓ #1")

    # --- Fila 3: Qué ROMPE al sistema (oclusión en inscripción, giro+sonrisa, FTA gafas/barba) ---
    lg_enr = portrait_crop(load_image(str(find_enrolled_photo("lady_gaga"))), bb)
    draw_tile(fig.add_subplot(gs[2, 0]), lg_enr, "Lady Gaga (2009)", "Flequillo tapa ojo!",
              border_color=COL_ENROLL, badge="OCLUSIÓN BD", badge_color=COL_WARN)
    r_lg18 = _find_rec(recs, "lady_gaga", "2018_look_natural")
    draw_tile(fig.add_subplot(gs[2, 1]), portrait_crop(load_image(str(ROOT / r_lg18["path"])), bb),
              "Gaga 2018 (natural)", f"sim = {r_lg18['sim_id_real']:.3f}",
              border_color=COL_FAIL, badge=f"RECHAZ ✗ #{r_lg18['rank_id_real']}")
    r_lg24 = _find_rec(recs, "lady_gaga", "2024_venecia")
    draw_tile(fig.add_subplot(gs[2, 2]), portrait_crop(load_image(str(ROOT / r_lg24["path"])), bb),
              "Gaga 2024 (antifaz)", f"sim = {r_lg24['sim_id_real']:.3f}",
              border_color=COL_FAIL, badge=f"RECHAZ ✗ #{r_lg24['rank_id_real']}")

    ce_enr = portrait_crop(load_image(str(find_enrolled_photo("cynthia_erivo"))), bb)
    draw_tile(fig.add_subplot(gs[2, 3]), ce_enr, "Cynthia Erivo (2025)", "Inscrita en BD",
              border_color=COL_ENROLL, badge="INSCRITA")
    r_ce25 = _find_rec(recs, "cynthia_erivo", "2025_wicked_frontal")
    draw_tile(fig.add_subplot(gs[2, 4]), portrait_crop(load_image(str(ROOT / r_ce25["path"])), bb),
              "Erivo 2025 (giro+sonr.)", f"sim={r_ce25['sim_id_real']:.3f} yaw=0.45",
              border_color=COL_FAIL, badge=f"RECHAZ ✗ #{r_ce25['rank_id_real']}")
    r_cefta = _find_rec(recs, "cynthia_erivo", "2025_world_pride")
    draw_tile(fig.add_subplot(gs[2, 5]), portrait_crop(load_image(str(ROOT / r_cefta["path"])), None),
              "Erivo 2025 (gafas)", "YuNet no detecta cara",
              border_color=COL_FAIL, badge="FTA: NoFace")

    r_cbfta = _find_rec(recs, "christian_bale", "2022_barba_reciente")
    draw_tile(fig.add_subplot(gs[2, 6]), portrait_crop(load_image(str(ROOT / r_cbfta["path"])), None),
              "Ch. Bale 2022 (barba)", "YuNet no detecta cara",
              border_color=COL_FAIL, badge="FTA: NoFace")

    fig.suptitle(
        "Ejemplos visuales reales — Transición de género, cambios tolerados (peso/barba/multi-enroll) y fallos (oclusión/FTA)\n"
        f"(Umbral verificación = {THR_VERIF:.4f} · Identificación = {THR_IDENT:.4f})",
        fontsize=12.2, fontweight="bold", y=0.975,
    )
    out = OUT / "ejemplos_cambios_transicion_estilo.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


# ============================================================================
# 3. Robustez y ataques de presentación (PAD) sobre Vinícius (ej. gafas, mascarilla, pantalla...)
# ============================================================================
def _apply_robust_transforms(bb: BigBrother, img: np.ndarray):
    def _lm(im):
        return bb.detector.detect(im, strict=False).landmarks

    def sunglasses(im):
        im = im.copy()
        L = _lm(im)
        d = np.linalg.norm(L[1] - L[0])
        c = ((L[0] + L[1]) / 2).astype(int)
        cv2.rectangle(im, tuple((c - [int(1.1 * d), int(.35 * d)]).tolist()),
                      tuple((c + [int(1.1 * d), int(.35 * d)]).tolist()), (10, 10, 10), -1)
        return im

    def mask(im):
        im = im.copy()
        L = _lm(im)
        d = np.linalg.norm(L[1] - L[0])
        top = int(L[2][1] + .1 * d)
        xc = int(L[2][0])
        cv2.rectangle(im, (xc - int(1.3 * d), top), (xc + int(1.3 * d), top + int(1.8 * d)), (230, 200, 160), -1)
        return im

    def screen(im):
        h, w = im.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[.04 * w, .02 * h], [.97 * w, .05 * h], [.95 * w, .97 * h], [.02 * w, .94 * h]])
        out = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(20, 20, 20))
        yy, xx = np.mgrid[:h, :w]
        moire = 12 * np.sin(2 * np.pi * (xx * .9 + yy * .35) / 3.1)
        out = np.clip(255 * (out / 255.0) ** 0.8 + moire[..., None], 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (0, 0), 1.0)

    def printed(im):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= .6
        out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        out = .75 * out + 40 + [0, 5, 15]
        out += np.random.default_rng(0).normal(0, 6, out.shape)
        return out.clip(0, 255).astype(np.uint8)

    def rot(a):
        return lambda im: cv2.warpAffine(im, cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, 1),
                                         (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REFLECT)

    def jpeg(q):
        return lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

    def lowres(f):
        return lambda im: cv2.resize(cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA),
                                     (im.shape[1], im.shape[0]))

    def gamma(g):
        return lambda im: (255 * (im / 255.0) ** g).astype(np.uint8)

    return [
        ("original", "Original", lambda im: im),
        ("PAD: foto de pantalla", "PAD: pantalla (V2)", screen),
        ("PAD: foto impresa", "PAD: impresa (V2)", printed),
        ("rotacion 15", "Rotación 15°", rot(15)),
        ("oscura (gamma 2.2)", "Oscura (γ=2,2)", gamma(2.2)),
        ("baja resolucion x0.25", "Baja res. ×0,25", lowres(0.25)),
        ("jpeg q=10", "JPEG q=10", jpeg(10)),
        ("ruido s=20", "Ruido σ=20", lambda im: np.clip(im + np.random.default_rng(0).normal(0, 20, im.shape), 0, 255).astype(np.uint8)),
        ("desenfoque s=2", "Desenfoque σ=2", lambda im: cv2.GaussianBlur(im, (0, 0), 2)),
        ("desenfoque s=4", "Desenfoque σ=4", lambda im: cv2.GaussianBlur(im, (0, 0), 4)),
        ("mascarilla", "Mascarilla (V7)", mask),
        ("gafas de sol", "Gafas de sol (V7)", sunglasses),
    ]


def fig_robustez_vinicius(bb: BigBrother, res: dict):
    vini_paths = sorted(VINI_DIR.glob("*.jpg"))
    # Calcular embeddings de las 5 fotos de Vinicius para evaluar LOO real sobre vini_01 (idx=0)
    idx_demo = 0
    vini_refs = [bb.projection(bb.extraer(str(p))) for p in vini_paths]
    other_refs = np.stack([vini_refs[i] for i in range(len(vini_refs)) if i != idx_demo])
    img_orig = load_image(str(vini_paths[idx_demo]))

    items = _apply_robust_transforms(bb, img_orig)
    fig = plt.figure(figsize=(15.0, 6.4), dpi=160)
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.14, top=0.86, bottom=0.07, left=0.02, right=0.98)

    rob = res["robustez"]
    for k, (key, title, fn) in enumerate(items):
        row, col_i = divmod(k, 6)
        im_t = fn(img_orig)
        # Evaluar LOO sobre esta foto transformada
        try:
            t = bb.extraer(im_t)
            score = float((bb.projection(t) @ other_refs.T).max())
            ok = score >= THR_VERIF
            status = f"sim={score:.3f}"
        except BigBrotherError as e:
            score = None
            ok = False
            status = f"FTA: {type(e).__name__}"

        agg = rob[key]
        rate = f"{agg['vini_aceptadas']}/{agg['vini_n']} acept."
        if key.startswith("PAD:"):
            border = COL_WARN if ok else COL_OK
            badge = f"ENTRA ({rate})"
        else:
            border = COL_OK if ok else COL_FAIL
            badge = f"VERIF ✓ ({rate})" if ok else f"RECHAZ ✗ ({rate})"

        # Recortar centrado sobre la cara original para que se vean gafas/mascarilla/pantalla bien encuadradas
        crop = portrait_crop(im_t, bb if score is not None else None, size=220, pad=0.45)
        draw_tile(fig.add_subplot(gs[row, col_i]), crop,
                  title, f"{status} · med {agg['vini_score_medio']:.2f}",
                  border_color=border, badge=badge, title_size=8.3, sub_size=7.0)

    fig.suptitle(
        "Robustez, oclusiones (V7) y ataques de presentación PAD (V2) sobre vinicius_junior (vini_01, LOO)\n"
        f"(Verde = acepta al genuino · Naranja = ataque PAD entra sin liveness · Rojo = rechazado; umbral = {THR_VERIF:.4f})",
        fontsize=11.6, fontweight="bold", y=0.965,
    )
    out = OUT / "ejemplos_robustez_vinicius.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


# ============================================================================
# 4. Gemelos (V5, V19), Parecidos / Sesgo demográfico (V4) y 1-shot vs LFW (V6)
# ============================================================================
def _lfw_crop_by_name(lfw_names: np.ndarray, lfw_crops: np.ndarray, name: str,
                      best_against: np.ndarray | None = None, bb: BigBrother | None = None,
                      pick: str = "max") -> np.ndarray:
    idxs = np.where(lfw_names == name)[0]
    if len(idxs) == 0:
        raise KeyError(f"LFW person not found: {name}")
    if best_against is None or bb is None or len(idxs) == 1:
        return lfw_crops[idxs[0]]
    scores = []
    for i in idxs:
        t = bb.extractor.extract(lfw_crops[i])
        y = bb.projection(t)
        s = float((best_against @ y).max()) if best_against.ndim == 2 else float(best_against @ y)
        scores.append(s)
    chosen = int(np.argmax(scores) if pick == "max" else np.argmin(scores))
    return lfw_crops[idxs[chosen]]


def fig_gemelos_parecidos(bb: BigBrother, res: dict, gemela: dict, far: dict):
    npz = np.load(CACHE / "lfw_crops.npz", allow_pickle=True)
    lfw_crops = npz["crops"]
    lfw_names = npz["names"]

    fig = plt.figure(figsize=(15.5, 9.6), dpi=160)
    gs = fig.add_gridspec(3, 6, hspace=0.52, wspace=0.14, top=0.90, bottom=0.05, left=0.02, right=0.98)

    # --- Fila 1: Gemelos (V5) y super-wolf (V19) ---
    ait_enr = portrait_crop(load_image(str(find_enrolled_photo("aitana_etxebarria"))), bb)
    draw_tile(fig.add_subplot(gs[0, 0]), ait_enr,
              "Aitana Etxebarria", "Inscrita en BD",
              border_color=COL_ENROLL, badge="VÍCTIMA BD")

    p1 = portrait_crop(load_image(str(PAULA_DIR / "paula_01.jpg")), bb)
    s_p1 = gemela["fotos"]["paula_01.jpg"]["score_victima"]
    draw_tile(fig.add_subplot(gs[0, 1]), p1,
              "Paula Etxeberria (01)", f"vs Aitana = {s_p1:.3f} (2/50 id)",
              border_color=COL_FAIL, badge="ENTRA IDENT ✓")

    p2 = portrait_crop(load_image(str(PAULA_DIR / "paula_02.jpg")), bb)
    s_p2 = gemela["fotos"]["paula_02.jpg"]["score_victima"]
    draw_tile(fig.add_subplot(gs[0, 2]), p2,
              "Paula (02 · filtro)", f"vs Aitana = {s_p2:.3f} (8/50 id!)",
              border_color=COL_FAIL, badge="SUPER-WOLF 8/50")

    # Mostrar 2 de las mujeres que paula_02 también abre (V19)
    kj_enr = portrait_crop(load_image(str(find_enrolled_photo("kim_jisoo"))), bb)
    draw_tile(fig.add_subplot(gs[0, 3]), kj_enr,
              "Kim Jisoo (ajena)", "paula_02 → 0.406 ✓",
              border_color=COL_WARN, badge="FALSO POS. V19")

    # Gemelos LFW: Mary-Kate inscrita vs Ashley Olsen (gemela)
    mk_enr = portrait_crop(load_image(str(find_enrolled_photo("mary-kate_olsen"))), bb)
    draw_tile(fig.add_subplot(gs[0, 4]), mk_enr,
              "Mary-Kate Olsen", "Inscrita (genuina LFW ~0.174)",
              border_color=COL_ENROLL, badge="INSCRITA BD")

    ash_crop = _lfw_crop_by_name(lfw_names, lfw_crops, "Ashley_Olsen")
    s_ash = res["gemelos"]["Ashley_Olsen->mary-kate_olsen"]["scores"][0]
    draw_tile(fig.add_subplot(gs[0, 5]), ash_crop,
              "Ashley Olsen (gemela)", f"vs MK = {s_ash:.3f} (= genuina!)",
              border_color=COL_WARN, badge="MISMO SCORE")

    # --- Fila 2: Sesgo demográfico / Parecidos de Vinícius en LFW (V4) + peor wolf LFW ---
    vn_enr = portrait_crop(load_image(str(VINI_DIR / "vini_01.jpg")), bb)
    draw_tile(fig.add_subplot(gs[1, 0]), vn_enr,
              "Vinícius Júnior", "Genuino LOO: 0.28–0.36",
              border_color=COL_ENROLL, badge="VÍCTIMA BD")

    vini_vecs = np.stack(bb.db.get("vinicius_junior"))
    top4_vini = res["open_set"]["top10_parecidos_vinicius"][:4]
    for idx, (lfw_name, sc) in enumerate(top4_vini, start=1):
        crop = _lfw_crop_by_name(lfw_names, lfw_crops, lfw_name, vini_vecs, bb, pick="max")
        draw_tile(fig.add_subplot(gs[1, idx]), crop,
                  lfw_name.replace("_", " "), f"sim vs Vini = {sc:.3f}",
                  border_color=COL_FAIL, badge="IMPOSTOR ✓ V4")

    # Peor wolf LFW a FAR 0.1%: Gina_Gershon (7/50)
    gg_crop = _lfw_crop_by_name(lfw_names, lfw_crops, "Gina_Gershon")
    draw_tile(fig.add_subplot(gs[1, 5]), gg_crop,
              "Gina Gershon (LFW)", "Abre 7/50 identidades",
              border_color=COL_WARN, badge="PEOR WOLF 7/50")

    # --- Fila 3: SSPP — 1 foto inscrita vs fotos reales en LFW (V6) ---
    jc_enr = portrait_crop(load_image(str(find_enrolled_photo("jackie_chan"))), bb)
    draw_tile(fig.add_subplot(gs[2, 0]), jc_enr, "Jackie Chan (2000)", "1 foto inscrita (0/13 LFW)",
              border_color=COL_ENROLL, badge="INSCRITA 1-SHOT")
    jc_vec = np.stack(bb.db.get("jackie_chan"))
    jc_lfw = _lfw_crop_by_name(lfw_names, lfw_crops, "Jackie_Chan", jc_vec, bb, pick="max")
    draw_tile(fig.add_subplot(gs[2, 1]), jc_lfw, "Jackie Chan (LFW)", "Mejor de 13: sim = 0.106",
              border_color=COL_FAIL, badge="RECHAZ 0/13")

    mj_enr = portrait_crop(load_image(str(find_enrolled_photo("michael_jackson"))), bb)
    draw_tile(fig.add_subplot(gs[2, 2]), mj_enr, "Michael Jackson (1982)", "1 foto inscrita (0/12 LFW)",
              border_color=COL_ENROLL, badge="INSCRITA 1-SHOT")
    mj_vec = np.stack(bb.db.get("michael_jackson"))
    mj_lfw = _lfw_crop_by_name(lfw_names, lfw_crops, "Michael_Jackson", mj_vec, bb, pick="max")
    draw_tile(fig.add_subplot(gs[2, 3]), mj_lfw, "M. Jackson (LFW ~2003)", "Mejor de 12: sim = 0.136",
              border_color=COL_FAIL, badge="RECHAZ 0/12")

    sh_enr = portrait_crop(load_image(str(find_enrolled_photo("salma_hayek"))), bb)
    draw_tile(fig.add_subplot(gs[2, 4]), sh_enr, "Salma Hayek (2012)", "1 foto inscrita (4/13 LFW)",
              border_color=COL_ENROLL, badge="INSCRITA 1-SHOT")
    sh_vec = np.stack(bb.db.get("salma_hayek"))
    sh_ok = _lfw_crop_by_name(lfw_names, lfw_crops, "Salma_Hayek", sh_vec, bb, pick="max")
    draw_tile(fig.add_subplot(gs[2, 5]), sh_ok, "Salma Hayek (LFW)", "Mejor de 13: sim = 0.256",
              border_color=COL_OK, badge="VERIF ✓ (4/13)")

    fig.suptitle(
        "Ejemplos visuales reales — Gemelos (V5/V19), sesgo demográfico y wolves en LFW (V4), y 1-shot vs LFW (V6)\n"
        f"(Umbral verificación = {THR_VERIF:.4f} · Identificación = {THR_IDENT:.4f})",
        fontsize=11.8, fontweight="bold", y=0.975,
    )
    out = OUT / "ejemplos_gemelos_parecidos.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


# ============================================================================
# 5. Distancia interocular (IOD) y efecto anti-aliasing (INTER_LINEAR vs INTER_AREA)
# ============================================================================
def _align_legacy_no_aa(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Version anterior de align() sin pre-reduccion INTER_AREA (para comparacion visual)."""
    from bigbrother.preprocess import _TEMPLATE
    dst = _TEMPLATE * config.FACE_SIZE
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (config.FACE_SIZE, config.FACE_SIZE),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fig_interocular_escala(bb: BigBrother):
    # Fila 1: vini_06 reescalada a IOD = 10, 12, 16, 20, 30, 48, 93 px (natural)
    vini_paths = sorted(VINI_DIR.glob("*.jpg"))
    idx_demo = 4  # vini_06.jpg (IOD natural ~92.8 px)
    vini_refs = [bb.projection(bb.extraer(str(p))) for p in vini_paths]
    other_refs = np.stack([vini_refs[i] for i in range(len(vini_refs)) if i != idx_demo])
    img_v6 = load_image(str(vini_paths[idx_demo]))
    f_nat = bb.detector.detect(img_v6, strict=False)
    iod_nat = float(np.linalg.norm(f_nat.landmarks[1] - f_nat.landmarks[0]))

    iods = [10, 12, 16, 20, 30, 48, round(iod_nat)]
    fig = plt.figure(figsize=(15.5, 6.2), dpi=160)
    gs = fig.add_gridspec(2, 7, hspace=0.50, wspace=0.14, top=0.86, bottom=0.06, left=0.02, right=0.98)

    for col_i, target_iod in enumerate(iods):
        scale = target_iod / iod_nat
        if abs(scale - 1.0) > 1e-3:
            nw, nh = max(16, int(round(img_v6.shape[1] * scale))), max(16, int(round(img_v6.shape[0] * scale)))
            im_s = cv2.resize(img_v6, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        else:
            im_s = img_v6
        f_s = bb.detector.detect(im_s, strict=False)
        crop = align(im_s, f_s.landmarks)
        t = bb.extractor.extract(normalize_face(crop))
        sc = float((bb.projection(t) @ other_refs.T).max())
        # Verificar si pasa control estricto
        strict_ok = True
        try:
            bb.detector.detect(im_s, strict=True)
        except BigBrotherError:
            strict_ok = False
        if not strict_ok:
            border, badge = COL_FAIL, "FTA <40px"
        elif sc >= THR_VERIF:
            border, badge = COL_OK, "OK · VERIF ✓"
        else:
            border, badge = COL_WARN, "OK · RECHAZ"
        lbl = f"IOD = {target_iod} px" + (" (nat.)" if col_i == len(iods) - 1 else "")
        draw_tile(fig.add_subplot(gs[0, col_i]), crop, lbl, f"LOO sim = {sc:.3f}",
                  border_color=border, badge=badge, title_size=8.2, sub_size=7.2)

    # Fila 2: Comparativa anti-aliasing en alta resolucion (antes INTER_LINEAR vs ahora INTER_AREA)
    hi_cases = [
        ("eva_hache", "Eva Hache"),
        ("adele", "Adele"),
        ("james_phelps", "James Phelps"),
    ]
    for idx_h, (pid, pretty) in enumerate(hi_cases):
        im_h = load_image(str(find_enrolled_photo(pid)))
        fh = bb.detector.detect(im_h, strict=False)
        iod_h = float(np.linalg.norm(fh.landmarks[1] - fh.landmarks[0]))
        c_old = _align_legacy_no_aa(im_h, fh.landmarks)
        c_new = align(im_h, fh.landmarks)
        # Mostrar versión equalized en escala de grises que ve realmente MLBP (donde el aliasing salta a la vista)
        g_old = cv2.equalizeHist(cv2.cvtColor(c_old, cv2.COLOR_BGR2GRAY))
        g_new = cv2.equalizeHist(cv2.cvtColor(c_new, cv2.COLOR_BGR2GRAY))
        c0 = idx_h * 2
        draw_tile(fig.add_subplot(gs[1, c0]), g_old,
                  f"{pretty} (ANTES)", f"IOD {iod_h:.0f}px · INTER_LINEAR",
                  border_color=COL_FAIL, badge="ALIASING ✗", title_size=8.0, sub_size=6.8)
        draw_tile(fig.add_subplot(gs[1, c0 + 1]), g_new,
                  f"{pretty} (AHORA)", f"IOD {iod_h:.0f}px · INTER_AREA",
                  border_color=COL_OK, badge="LIMPIO ✓", title_size=8.0, sub_size=6.8)

    ax_note = fig.add_subplot(gs[1, 6])
    ax_note.axis("off")
    ax_note.text(
        0.5, 0.5,
        "Efecto de\nINTER_AREA\n(entrada real\na MLBP tras\nequalizeHist):\n\nFAR galería 120px:\n6,91% → 2,13%\n\nCross-FPs:\n234 → 36",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1b4f72",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eaf2f8", edgecolor="#85c1e9"),
    )

    fig.suptitle(
        "Lo que realmente ve MLBP (recorte 128×128): degradación por baja IOD (arriba) y corrección anti-aliasing en alta resolución (abajo)",
        fontsize=11.4, fontweight="bold", y=0.96,
    )
    out = OUT / "ejemplos_interocular_escala.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


# ============================================================================
# 6. Ataques empíricos visuales: A3 Morphing (α=0..1) y A5 Dodging del detector
# ============================================================================
def fig_ataques_morph_dodging(bb: BigBrother, atk: dict):
    fig = plt.figure(figsize=(15.5, 5.8), dpi=160)
    gs = fig.add_gridspec(2, 6, hspace=0.48, wspace=0.14, top=0.85, bottom=0.07, left=0.02, right=0.98)

    # --- Fila 1: A3 Morphing (vini_01 + paula_01) ---
    cv_crop = bb.recortar(str(VINI_DIR / "vini_01.jpg"))
    ca_crop = bb.recortar(str(PAULA_DIR / "paula_01.jpg"))
    a3_by_alpha = {round(r["alpha"], 1): r for r in atk["a3_morphing_inscripcion"]}

    alphas = [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]
    for col_i, a in enumerate(alphas):
        m = cv2.addWeighted(cv_crop, 1.0 - a, ca_crop, a, 0.0)
        row_a = a3_by_alpha[a]
        sv, sa = row_a["score_morph_victima"], row_a["score_morph_atacante"]
        if a == 0.0:
            border, badge = COL_ENROLL, "VÍCTIMA (0%)"
        elif a == 1.0:
            border, badge = COL_NEUTRAL, "ATACANTE (100%)"
        elif row_a.get("poisoning_aceptado"):
            border, badge = COL_FAIL, "POISONING ✓"
        elif row_a.get("atacante_verifica_como_sospechoso"):
            border, badge = COL_WARN, "ID NUEVO + ATAC ✓"
        else:
            border, badge = COL_WARN, "ID NUEVO ✓"
        draw_tile(fig.add_subplot(gs[0, col_i]), m,
                  f"Morph α = {a:.1f}",
                  f"s(víct)={sv:.2f} · s(atac)={sa:.2f}",
                  border_color=border, badge=badge, title_size=8.3, sub_size=7.2)

    # --- Fila 2: A5 Dodging del detector (vini_06) + muestra de A2 / morfología ---
    img_v6 = load_image(str(VINI_DIR / "vini_06.jpg"))
    face = bb.detector.detect(img_v6)
    re_, le = face.landmarks[0], face.landmarks[1]
    cx, cy = (re_[0] + le[0]) / 2, (re_[1] + le[1]) / 2
    w = abs(le[0] - re_[0])

    p_ojos = img_v6.copy()
    cv2.rectangle(p_ojos, (int(cx - w * 1.2), int(cy - w * 0.7)),
                  (int(cx + w * 1.2), int(cy + w * 0.7)), (0, 0, 0), -1)

    p_frente = img_v6.copy()
    cv2.rectangle(p_frente, (int(cx - w), int(cy - w * 2.0)),
                  (int(cx + w), int(cy - w * 1.1)), (0, 0, 0), -1)

    # Usar el mismo recorte centrado en img_v6 para los 3
    def _crop_like_v6(im):
        x, y, bw, bh = face.bbox
        side = max(bw, bh) * 2.1
        hc, vc = x + bw / 2, y + bh * 0.45
        x1, y1 = max(0, int(hc - side / 2)), max(0, int(vc - side / 2))
        x2, y2 = min(im.shape[1], int(hc + side / 2)), min(im.shape[0], int(vc + side / 2))
        return cv2.resize(im[y1:y2, x1:x2], (220, 220), interpolation=cv2.INTER_AREA)

    draw_tile(fig.add_subplot(gs[1, 0]), _crop_like_v6(img_v6),
              "A5: Sin parche", "verificar = True",
              border_color=COL_OK, badge="DETECTADO ✓")
    draw_tile(fig.add_subplot(gs[1, 1]), _crop_like_v6(p_ojos),
              "A5: Parche ojos", "YuNet → NoFaceDetected",
              border_color=COL_FAIL, badge="EVADIDO ✗")
    draw_tile(fig.add_subplot(gs[1, 2]), _crop_like_v6(p_frente),
              "A5: Parche frente (ctrl)", "verificar = True",
              border_color=COL_OK, badge="SIGUE ENTRANDO ✓")

    # Celdas 3-5 de la fila 2: ataque de presentación V3 (morph con cómplice parecido de LFW: Shawn_Kemp)
    npz = np.load(CACHE / "lfw_crops.npz", allow_pickle=True)
    lfw_crops, lfw_names = npz["crops"], npz["names"]
    vini_vecs = np.stack(bb.db.get("vinicius_junior"))
    sk_crop = _lfw_crop_by_name(lfw_names, lfw_crops, "Shawn_Kemp", vini_vecs, bb, pick="max")
    sk_bgr = cv2.cvtColor(sk_crop, cv2.COLOR_GRAY2BGR) if sk_crop.ndim == 2 else sk_crop
    morph_sk = cv2.addWeighted(cv_crop, 0.5, sk_bgr, 0.5, 0.0)
    s_sk_morph = float((bb.projection(bb.extractor.extract(normalize_face(morph_sk))) @ vini_vecs.T).max())

    draw_tile(fig.add_subplot(gs[1, 3]), cv_crop,
              "V3: Vinícius (50%)", "Víctima alineada",
              border_color=COL_ENROLL, badge="VÍCTIMA")
    draw_tile(fig.add_subplot(gs[1, 4]), sk_crop,
              "V3: Shawn Kemp (50%)", "Cómplice parecido (LFW)",
              border_color=COL_NEUTRAL, badge="CÓMPLICE")
    draw_tile(fig.add_subplot(gs[1, 5]), morph_sk,
              "V3: Morph 50/50", f"sim={s_sk_morph:.3f} (24/25 acept.)",
              border_color=COL_FAIL, badge="ENTRA ✓ V3")

    fig.suptitle(
        "Ejemplos visuales reales — Ataques de morphing por inscripción / presentación (A3, V3) y evasión del detector (A5)\n"
        f"(Umbral verificación = {THR_VERIF:.4f} · Deduplicación = {config.DEDUP_THRESHOLD:.4f})",
        fontsize=11.6, fontweight="bold", y=0.96,
    )
    out = OUT / "ejemplos_ataques_morph_dodging.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Guardado:", out)


def main():
    bb = BigBrother()
    cam = json.loads((OUT / "eval_cambios_apariencia.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / "resultados.json").read_text(encoding="utf-8"))
    gemela = json.loads((OUT / "gemela_paula.json").read_text(encoding="utf-8"))
    far = json.loads((OUT / "eval_far_levels.json").read_text(encoding="utf-8"))
    atk = json.loads((OUT / "ataques_literatura.json").read_text(encoding="utf-8"))

    fig_cambios_mj_envejecimiento(bb, cam)
    fig_cambios_transicion_estilo(bb, cam)
    fig_robustez_vinicius(bb, res)
    fig_gemelos_parecidos(bb, res, gemela, far)
    fig_interocular_escala(bb)
    fig_ataques_morph_dodging(bb, atk)


if __name__ == "__main__":
    main()
