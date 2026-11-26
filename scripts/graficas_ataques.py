"""Figuras ilustrativas de docs/ATAQUES_EMPIRICOS.md (generadas con matplotlib).

    python scripts/graficas_ataques.py
    -> docs/resultados/{pipeline_ataques,a1_integridad,a2_hillclimb,a3_morphing}.png

F1 pipeline_ataques : el pipeline del sistema + los tres canales de la API + los puntos de
                     Ratha + donde golpea cada ataque A1-A5.
F2 a1_integridad    : que cubre la huella firmada (BD y models/ SI, codigo NO) y los dos
                     caminos de manipulacion (detectado / no detectado) + mitigacion.
F3 a2_hillclimb     : fases del ataque con las cifras medidas + curva real mezcla en el
                     espacio de templates frente a morphing de pixeles.
F4 a3_morphing      : barrido alpha del morphing con los umbrales y las zonas
                     (dedup bloquea / poisoning aceptado / probes de ambos padres).

Los datos de F3-F4 salen de docs/resultados/ataques_literatura.json; la curva de mezcla en
el espacio de templates se calcula aqui (caja blanca, solo para la grafica).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from ataques_literatura import (PAULA_01, VINI_DIR, VICTIM, blend_counts,  # noqa: E402
                                from_counts, open_bb, to_counts)

OUT = ROOT / "docs" / "resultados"
RESULTS = OUT / "ataques_literatura.json"

# paleta (coherente con la de las figuras del INFORME)
C_AZUL, C_NARANJA, C_VERDE, C_ROJO, C_GRIS = "#4a6fa5", "#d98e32", "#3d8b5f", "#c0392b", "#888888"
THRESH, DEDUP = 0.2172, 0.2782


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=C_AZUL, fs=7.0, lw=1.1, tc="#222222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.3, ls="-", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=1, shrinkB=1,
                                connectionstyle=f"arc3,rad={rad}"))


def marker(ax, x, y, label, dx=0, dy=0):
    """Marcador circular de ataque (A1..A5) con pequeño offset."""
    ax.add_patch(Circle((x + dx, y + dy), 2.1, fc=C_ROJO, ec="white", lw=1.2, zorder=6))
    ax.text(x + dx, y + dy, label, ha="center", va="center", fontsize=6.2,
            color="white", weight="bold", zorder=7)


# ============================================================ F1: pipeline y puntos de ataque
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-11, 68)
    ax.axis("off")

    # telon de fondo: el CODIGO desplegado (no cubierto por la huella -> A1)
    ax.add_patch(Rectangle((1, 7), 98, 53, fc="#f4f4f2", ec="#bbbbbb", lw=1.1, ls="--", zorder=0))
    ax.text(50, 3.8, "todo lo anterior lo ejecuta CÓDIGO desplegado — fuera de la huella de integridad (A1: Troyano)",
            fontsize=7.4, color="#777777", style="italic", ha="center", va="bottom")

    # fila 1: captura -> deteccion -> normalizacion -> extraccion
    y1 = 44
    box(ax, 2, y1, 13, 9, "foto\n(JPG/PNG/…)", fc="#fdf6e9", ec=C_NARANJA)
    box(ax, 19, y1, 13, 9, "YuNet\ndetección +\ncalidad", fc="#eef3fb")
    box(ax, 36, y1, 13, 9, "alineado\n128×128 +\necualización", fc="#eef3fb")
    box(ax, 53, y1, 13, 9, "MLBP\nhistogramas\n384×59", fc="#fdeedd", ec=C_NARANJA)
    for x1, x2 in ((15, 19), (32, 36), (49, 53)):
        arrow(ax, x1, y1 + 4.5, x2, y1 + 4.5)

    # fila 2 (derecha -> izquierda): faceness -> proyeccion -> coseno -> decision
    y2 = 26
    box(ax, 53, y2, 13, 9, "faceness\n«¿es una\ncara?»", fc="#eef3fb")
    box(ax, 36, y2, 13, 9, "proyección\n√ → PCA →\nLDA (300)", fc="#eef3fb")
    box(ax, 19, y2, 13, 9, "coseno vs\ngalería +\numbral", fc="#eef3fb")
    box(ax, 2, y2, 13, 9, "decisión\nbool / id\n(sin score)", fc="#e8f4ec", ec=C_VERDE)
    arrow(ax, 59.5, y1, 59.5, y2 + 9)
    arrow(ax, 53, y2 + 4.5, 49, y2 + 4.5)
    arrow(ax, 36, y2 + 4.5, 32, y2 + 4.5)
    arrow(ax, 19, y2 + 4.5, 15, y2 + 4.5)

    # zona de datos: BD, zona protegida y modelos
    box(ax, 17, 8, 17, 10, "data/db.csv\ntemplates proyectados\nAES-256-GCM (id = AAD)",
        fc="#fdeedd", ec=C_NARANJA, fs=6.4)
    box(ax, 39, 8, 21, 10, "keys/ (zona protegida)\nclave maestra + huella HMAC\nde la BD y de models/",
        fc="#e8f4ec", ec=C_VERDE, fs=6.2)
    box(ax, 65, 8, 27, 10, "models/  ✔ cubierto por la huella\nYuNet · proyección LDA ·\nfaceness · umbrales",
        fc="#eef3fb", fs=6.2)
    arrow(ax, 25.5, 18, 25.5, y2, color=C_NARANJA)
    ax.text(27.2, 21.5, "descifra y compara", fontsize=6.0, color="#777777", ha="left")
    arrow(ax, 39, 13, 34, 13, color=C_VERDE, ls=":")
    arrow(ax, 60, 13, 65, 13, color=C_VERDE, ls=":")

    # canales de entrada (API)
    ax.text(8.5, 63.5, "API completa: inscribir / verificar / identificar (foto)", fontsize=6.8,
            ha="center", color="#333333")
    arrow(ax, 8.5, 62.5, 8.5, y1 + 9)
    ax.text(59.5, 63.5, "API *_sd (recorte, sin detector)", fontsize=6.8, ha="center", color="#333333")
    arrow(ax, 59.5, 62.5, 59.5, y1 + 9)
    ax.text(88, 63.5, "API *_t (template, sin detector ni extractor)", fontsize=6.8,
            ha="center", color="#333333")
    arrow(ax, 88, 62.5, 67, y2 + 9.5, rad=-0.12)

    # marcadores de ataque
    marker(ax, 19, y1 + 9, "A5", -2.2, 2.2)     # parche -> YuNet
    marker(ax, 68.5, 33.5, "A2", 0, 0)             # hill-climb por el canal _t (entra en faceness)
    marker(ax, 17, 18, "A3", 0, -2.4)           # morph/poisoning en la BD
    marker(ax, 60, 18, "A4", 1.6, -2.4)         # rotacion de clave (keys)
    marker(ax, 99, 60, "A1", -2.4, 0)           # troyano de codigo

    # leyenda
    leyenda = ("A1 troyano de código (Ratha 3/5/8) · A2 hill-climbing binario por el canal *_t · "
               "A3 morphing + poisoning de la galería · A4 rotación de clave no revoca · "
               "A5 parche sobre los ojos → NoFaceDetected")
    ax.text(50, -8.6, leyenda, fontsize=7.2, ha="center", va="center", color="#444444", wrap=True)

    ax.set_title("BigBrother: pipeline, canales de la API y dónde golpea cada ataque medido",
                 fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_ataques.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================ F2: A1 cobertura de la huella
def fig_a1():
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    ax.text(50, 44.5, "¿Qué cubre la huella firmada en keys/db.state?", fontsize=11, ha="center")
    box(ax, 10, 35, 22, 7, "data/db.csv  ✔", fc="#e8f4ec", ec=C_VERDE, fs=8.5)
    box(ax, 39, 35, 22, 7, "models/  ✔", fc="#e8f4ec", ec=C_VERDE, fs=8.5)
    box(ax, 68, 35, 22, 7, "bigbrother/*.py  ✗", fc="#fdeaea", ec=C_ROJO, fs=8.5)

    # camino izquierdo: manipular models -> detectado
    box(ax, 8, 19, 20, 9, "editar el umbral en\nmodels/calibration.json", fc="#fdeedd", ec=C_NARANJA, fs=6.8)
    arrow(ax, 18, 19, 18, 13.5, color=C_ROJO)
    box(ax, 8, 4.5, 20, 8, "arranque bloqueado:\nIntegrityError (rc 2)", fc="#e8f4ec", ec=C_VERDE, fs=7)

    # camino derecho: manipular codigo -> NO detectado
    box(ax, 46, 19, 24, 9, "parchear system.py\n(anular decisión / extractor\ntroyano: 1 línea)",
        fc="#fdeaea", ec=C_ROJO, fs=6.6)
    arrow(ax, 58, 19, 58, 13.5, color=C_ROJO)
    box(ax, 46, 4.5, 24, 8, "arranque rc 0:\nla huella sigue «válida»", fc="#fdeaea", ec=C_ROJO, fs=7)
    arrow(ax, 70, 8.5, 88, 8.5, color=C_ROJO)
    box(ax, 80, 4.5, 18, 8, "el atacante verifica\ncomo la víctima", fc="#fdeaea", ec=C_ROJO, fs=7)

    # mitigacion (validada en el paso 5 de A1)
    box(ax, 72, 19, 26, 10, "mitigación validada (paso 5):\ndigest de bigbrother/*.py en la huella\n→ el mismo troyano da IntegrityError",
        fc="#e8f4ec", ec=C_VERDE, fs=6.6)
    arrow(ax, 72, 24, 70, 24, color=C_VERDE, ls=":")

    fig.tight_layout()
    fig.savefig(OUT / "a1_integridad.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================ F3: A2 hill-climbing
def _curva_mezcla_templates():
    """Caja blanca (solo para la figura): score de la mezcla uniforme de histogramas de
    (victima, atacante) contra la galeria, para comparar con el morphing de pixeles."""
    bb = open_bb(*_fresh())
    fotos = sorted(str(p) for p in VINI_DIR.glob("*.jpg"))
    ex = bb.extractor
    Cv, Ca = to_counts(bb.extraer(fotos[0]), ex), to_counts(bb.extraer(str(PAULA_01)), ex)
    alphas = np.round(np.arange(0.0, 1.001, 0.02), 3)
    scores = []
    for a in alphas:
        y = bb.projection(from_counts(blend_counts(Cv, Ca, float(a))))
        scores.append(float(bb._score_id(VICTIM, y)))
    return alphas, np.array(scores)


def _fresh():
    import ataques_literatura as AL
    return AL.fresh_db_copy("gfx")


def fig_a2():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    a2 = res["a2_hillclimb_binario"]
    alphas, tpl_scores = _curva_mezcla_templates()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), width_ratios=[1.05, 1.25])

    # panel izquierdo: fases (esquema con las cifras medidas)
    a = ax[0]
    a.set_xlim(0, 100)
    a.set_ylim(0, 46)
    a.axis("off")
    a.set_title("Fases del ataque (oráculo: solo el bool de verificar_t)", fontsize=9)
    box(a, 2, 30, 46, 11, "fase 0 (2 consultas)\nfoto pública de la víctima\n→ template ACEPTADO",
        fc="#eef3fb", fs=7)
    box(a, 54, 30, 44, 11, "cara del atacante\n→ RECHAZADO", fc="#fdeedd", ec=C_NARANJA, fs=7)
    box(a, 2, 15, 44, 11,
        f"fase 1 · bisección de mezcla uniforme\n(55 consultas)\nfrontera: α* = {a2['alpha_frontera_uniforme']:.3f}",
        fc="#eef3fb", fs=7)
    box(a, 52, 15, 46, 11,
        f"fase 2 · hill-climb por regiones\n(4.001 consultas, {a2['pasos_aceptados_fase2']} pasos)\n"
        f"α medio final = {a2['alpha_medio_final']:.3f}", fc="#eef3fb", fs=7)
    box(a, 2, 1, 96, 10,
        f"template final: coseno {a2['sim_template_atacante']:.3f} con la CARA DEL ATACANTE "
        f"(base {a2['sim_base_victima_atacante']:.3f}) y score {a2['score_white_box_final']:.4f} "
        f"(umbral {THRESH}) → verificar_t = true", fc="#e8f4ec", ec=C_VERDE, fs=7.4)
    arrow(a, 24, 30, 24, 26, lw=1.1)
    arrow(a, 46, 20.5, 52, 20.5, lw=1.1)  # fase 1 -> fase 2
    arrow(a, 24, 15, 24, 11, lw=1.1)
    arrow(a, 75, 15, 75, 11, lw=1.1)

    # panel derecho: mezcla en el espacio de templates vs pixeles (datos reales)
    b = ax[1]
    m = {r["alpha"]: r for r in res["a3_morphing_inscripcion"]}
    ma = np.array(sorted(m))
    b.plot(alphas, tpl_scores, color=C_ROJO, lw=2.0,
           label="mezcla de TEMPLATES (histogramas, canal *_t)")
    b.plot(ma, [m[a]["score_morph_victima"] for a in ma], color=C_NARANJA, lw=2.0, marker="o", ms=3.5,
           label="morphing de PÍXELES (foto, canal completo)")
    b.axhline(THRESH, color="#555555", ls="--", lw=1.1)
    b.text(1.0, THRESH + 0.012, f"umbral de verificación ({THRESH})", fontsize=6.8, color="#555555")
    b.axvline(a2["alpha_frontera_uniforme"], color=C_ROJO, ls=":", lw=1.2)
    b.text(a2["alpha_frontera_uniforme"] - 0.015, 0.62,
           f"α* = {a2['alpha_frontera_uniforme']:.3f}:\naún aceptada", fontsize=7,
           color=C_ROJO, ha="right")
    b.set_xlabel("α de la mezcla  (0 = víctima pura · 1 = atacante puro)")
    b.set_ylabel("similitud con la galería de la víctima")
    b.set_title("La mezcla en el espacio de templates es mucho más peligrosa\n"
                "que el morphing de píxeles (misma pareja de caras)", fontsize=9)
    b.legend(fontsize=7, loc="lower left")
    b.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "a2_hillclimb.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================ F4: A3 morphing
def fig_a3():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = sorted(res["a3_morphing_inscripcion"], key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    sv = np.array([r["score_morph_victima"] for r in rows])
    pv = np.array([r["score_probe_victima_vini06"] for r in rows])
    pa = np.array([r["score_probe_atacante_paula02"] for r in rows])

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.axvspan(0, 0.55, color=C_ROJO, alpha=0.07)
    ax.axvspan(0, 0.6, color=C_VERDE, alpha=0.08, hatch="//", ec=C_VERDE, lw=0.0)

    ax.plot(a, sv, marker="o", ms=4, color=C_AZUL, lw=1.8,
            label="s(morph, galería víctima) — máx. de las 5")
    ax.plot(a, pv, marker="s", ms=4, color="#1f4e79", lw=1.6, ls="-.",
            label="s(morph, foto hold-out de la víctima: vini_06)")
    ax.plot(a, pa, marker="^", ms=4.5, color=C_NARANJA, lw=1.6, ls="--",
            label="s(morph, foto hold-out del atacante: paula_02)")
    ax.axhline(THRESH, color="#555555", ls="--", lw=1.1)
    ax.text(1.01, THRESH, f" umbral verificación ({THRESH})", fontsize=7, va="center", color="#555555")
    ax.axhline(DEDUP, color=C_ROJO, ls=":", lw=1.3)
    ax.text(1.01, DEDUP, f" umbral dedup ({DEDUP})", fontsize=7, va="center", color=C_ROJO)

    ax.annotate("α ≤ 0,5: inscribir como id nuevo\n→ RECHAZADO por deduplicación",
                xy=(0.35, 0.78), xytext=(0.36, 0.88), fontsize=7.2, color=C_ROJO,
                ha="center", arrowprops=dict(arrowstyle="-", color=C_ROJO, lw=0.9))
    ax.annotate("poisoning aceptado (α ≤ 0,6):\nel morph queda en la galería de la víctima\n"
                "(actualizar=True SIN dedup) y la foto del\nmorph pasa el sensor completo",
                xy=(0.28, DEDUP + 0.015), xytext=(0.05, 0.50), fontsize=7.2, color=C_VERDE)
    ax.annotate("atacante hold-out verifica\ncontra el id del morph (0,223)",
                xy=(0.8, 0.223), xytext=(0.72, 0.42), fontsize=7.2, color=C_NARANJA,
                arrowprops=dict(arrowstyle="-", color=C_NARANJA, lw=0.9))
    ax.annotate("víctima hold-out: 0,02\n(magic passport no cuadra)",
                xy=(0.6, 0.02), xytext=(0.47, 0.18), fontsize=7.2, color="#1f4e79",
                arrowprops=dict(arrowstyle="-", color="#1f4e79", lw=0.9))

    ax.set_xlabel("α del morph  (0 = víctima pura · 1 = atacante puro; mezcla lineal de recortes alineados)")
    ax.set_ylabel("similitud coseno")
    ax.set_title("A3 — magic passport y poisoning: qué bloquea la deduplicación y qué no "
                 "(generador de morphing débil: límite inferior)")
    ax.legend(fontsize=7.2, loc="center left")
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.05)

    fig.tight_layout()
    fig.savefig(OUT / "a3_morphing.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_a1()
    fig_a2()
    fig_a3()
    print("figuras generadas en", OUT)
