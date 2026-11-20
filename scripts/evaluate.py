"""Evaluacion del sistema: rendimiento + bateria de ataques.

    python scripts/evaluate.py            # genera docs/resultados/{resultados.json, *.png}

Requiere la cache de LFW (scripts/prepare_lfw.py). Trabaja sobre una BD temporal con clave
propia: no toca data/db.csv.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_projection as tp  # noqa: E402
from build_db import build  # noqa: E402

from bigbrother import config  # noqa: E402
from bigbrother.dataset import FOTOS_DIR  # noqa: E402
from bigbrother.errors import BigBrotherError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402
from bigbrother.mlbp import MLBPExtractor  # noqa: E402
from bigbrother.preprocess import normalize_face  # noqa: E402

OUT = ROOT / "docs" / "resultados"
CACHE = Path(os.path.expanduser("~/.cache/bigbrother/lfw_crops.npz"))
VINI = "vinicius_junior"
RESERVED_MAP = {"Christian_Bale": "christian_bale", "Jackie_Chan": "jackie_chan",
                "James_Phelps": "james_phelps", "Keanu_Reeves": "keanu_reeves",
                "Mary-Kate_Olsen": "mary-kate_olsen", "Michael_Jackson": "michael_jackson",
                "Salma_Hayek": "salma_hayek"}
TWINS = {"Ashley_Olsen": "mary-kate_olsen", "Oliver_Phelps": "james_phelps"}

R = {}  # resultados


def log(*a):
    print(*a, flush=True)


# ============================================================================ utilidades
class Ctx:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bb_eval_"))
        t = time.time()
        self.bb = build(self.tmp / "db.csv", self.tmp / "keys", verbose=False)
        R["build_db_s"] = round(time.time() - t, 2)
        self.thr = self.bb.threshold
        self.ex = MLBPExtractor()
        self.vini_paths = sorted((FOTOS_DIR / VINI).iterdir())
        self.vini_refs = self.bb.db.get(VINI)
        assert len(self.vini_refs) == len(self.vini_paths)
        z = np.load(CACHE)
        self.crops, self.names = z["crops"], z["names"]
        self.X = tp.features(CACHE, self.crops, config.LBP_RADII, config.GRID,
                             config.LBP_METHOD, config.PHOTOMETRIC, False)
        split = json.loads((config.MODELS_DIR / "lfw_split.json").read_text())
        train = set(split["train_ids"])
        reserved = set(split["reserved"])
        self.test_mask = np.array([(n not in train) and (n not in reserved) for n in self.names])

    def proj(self, T):
        return self.bb.projection(np.asarray(T, np.float32))

    def gallery(self, exclude_vini=None):
        """(ids, matriz) de la BD, opcionalmente sin el template exclude_vini de Vinicius."""
        ids, M = [], []
        k = 0
        for i, r in self.bb.db.all():
            if i == VINI:
                if k == exclude_vini:
                    k += 1
                    continue
                k += 1
            ids.append(i)
            M.append(r)
        return np.array(ids), np.stack(M)

    def best_per_id(self, ids, S):
        """S: (n_probes, n_templates) -> dict id -> (n_probes,) max."""
        return {u: S[:, ids == u].max(1) for u in np.unique(ids)}

    def vini_loo(self, templates):
        """Puntuacion LOO contra Vinicius: la foto i se compara con los otros templates."""
        Y = self.proj(np.stack(templates))
        S = Y @ np.stack(self.vini_refs).T
        return np.array([np.delete(S[i], i).max() for i in range(len(Y))])


def template_from_crop(ex, crop_gray):
    return ex.extract(normalize_face(crop_gray))


# ============================================================================ 1. LFW
def eval_lfw(c):
    log("[1] LFW test (identidades disjuntas del entrenamiento)")
    Xte, yte = np.asarray(c.X[c.test_mask]), c.names[c.test_mask]
    Y = c.proj(Xte)
    gen, imp = tp.pair_scores(Y, yte, np.random.default_rng(1), n_imp=1_000_000)
    m = tp.metrics(gen, imp)
    m.update(far_at_thr=float((imp >= c.thr).mean()), frr_at_thr=float((gen < c.thr).mean()),
             n_gen=int(len(gen)), n_imp=int(len(imp)))
    # baseline chi2 (vectorizado por bloques)
    rng = np.random.default_rng(2)
    yi = np.unique(yte, return_inverse=True)[1]
    gp = []
    for u in np.unique(yi):
        idx = np.where(yi == u)[0][:10]
        gp += [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
    gp = np.array(gp)
    ip = rng.integers(0, len(Xte), (60000, 2))
    ip = ip[yi[ip[:, 0]] != yi[ip[:, 1]]]

    def chi(P):
        out = []
        for s in range(0, len(P), 2000):
            a, b = Xte[P[s:s + 2000, 0]], Xte[P[s:s + 2000, 1]]
            out.append(((a - b) ** 2 / (a + b + 1e-10)).sum(1))
        return -np.concatenate(out)
    gb, ib = chi(gp), chi(ip)
    mb = tp.metrics(gb, ib)
    R["lfw"] = {"mlbp_lda_coseno": m, "mlbp_chi2_baseline": mb}
    log("   MLBP+LDA:", m)
    log("   chi2    :", mb)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (g, i_, lab) in ((gen, imp, "MLBP + PCA/LDA + coseno (nuestro)"), (gb, ib, "MLBP + chi2 (baseline)")):
        th = np.quantile(i_, np.linspace(0, 1, 2000))
        ax[0].plot([(i_ >= t).mean() for t in th], [(g >= t).mean() for t in th], label=lab)
    ax[0].axvline(m["far_at_thr"], ls="--", c="gray", lw=1, label=f"umbral (FAR={m['far_at_thr']:.2%})")
    ax[0].set(xscale="log", xlim=(1e-4, 1), xlabel="FAR", ylabel="TAR (1-FRR)", title="ROC en LFW (test)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].hist(imp, 100, density=True, alpha=.6, label="impostores")
    ax[1].hist(gen, 100, density=True, alpha=.6, label="genuinos")
    ax[1].axvline(c.thr, c="k", ls="--", label="umbral")
    ax[1].set(xlabel="similitud coseno", title="Distribucion de puntuaciones (LFW test)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "roc_lfw.png", dpi=120)
    plt.close(fig)
    return gen, imp


# ============================================================================ 2. Vinicius
def eval_vinicius(c):
    log("[2] Vinicius (leave-one-out sobre las fotos de inscripcion)")
    T = [c.bb.extraer(load_image(p)) for p in c.vini_paths]
    loo = c.vini_loo(T)
    # identificacion LOO
    ranks, ident_ok = [], []
    for i, t in enumerate(T):
        ids, M = c.gallery(exclude_vini=i)
        best = c.best_per_id(ids, c.proj(t)[None] @ M.T)
        order = sorted(best, key=lambda u: -best[u][0])
        ranks.append(order.index(VINI) + 1)
        ident_ok.append(bool(order[0] == VINI and best[VINI][0] >= c.bb.ident_threshold))
    R["vinicius_loo"] = {"fotos": [p.name for p in c.vini_paths], "scores": loo.round(4).tolist(),
                         "aceptadas": int((loo >= c.thr).sum()), "total": len(loo), "rank": ranks,
                         "identificadas": int(sum(ident_ok))}
    log("   scores LOO:", loo.round(3), "rank:", ranks)
    return T, loo


# ============================================================================ 3. genuinos LFW de inscritos
def eval_reserved(c):
    log("[3] Genuinos cruzados: fotos LFW de personas inscritas con la foto del docente")
    ids, M = c.gallery()
    out, scores = {}, []
    for lfw_name, id_ in RESERVED_MAP.items():
        m = c.names == lfw_name
        if not m.any():
            continue
        S = c.proj(np.asarray(c.X[m])) @ M.T
        best = c.best_per_id(ids, S)
        own = best[id_]
        top = np.array([max(best, key=lambda u: best[u][k]) for k in range(len(own))])
        out[id_] = {"n": int(m.sum()), "aceptadas": int((own >= c.thr).sum()),
                    "rank1": int((top == id_).sum()), "media": float(own.mean())}
        scores += own.tolist()
    R["genuinos_lfw_inscritos"] = out
    tot = sum(v["n"] for v in out.values())
    acc = sum(v["aceptadas"] for v in out.values())
    R["genuinos_lfw_inscritos_total"] = {"n": tot, "aceptadas": acc, "tar": acc / max(tot, 1)}
    log("   ", R["genuinos_lfw_inscritos_total"])
    return np.array(scores)


# ============================================================================ 4. gemelos
def eval_twins(c):
    log("[4] Ataque con gemelo")
    ids, M = c.gallery()
    out = {}
    for twin, victim in TWINS.items():
        m = c.names == twin
        S = c.proj(np.asarray(c.X[m])) @ M.T
        best = c.best_per_id(ids, S)
        v = best[victim]
        top = [max(best, key=lambda u: best[u][k]) for k in range(len(v))]
        out[f"{twin}->{victim}"] = {"n": int(m.sum()), "scores": v.round(4).tolist(),
                                    "aceptado_como_victima": int((v >= c.thr).sum()),
                                    "identificado_como": [t if best[t][k] >= c.bb.ident_threshold else None
                                                          for k, t in enumerate(top)]}
    R["gemelos"] = out
    log("   ", out)
    return np.concatenate([np.array(v["scores"]) for v in out.values()])


# ============================================================================ 5. impostores open-set
def eval_open_set(c):
    log("[5] Impostores no inscritos (LFW test) contra la galeria de 51")
    ids, M = c.gallery()
    Y = c.proj(np.asarray(c.X[c.test_mask]))
    S = Y @ M.T
    best = c.best_per_id(ids, S)
    B = np.stack([best[u] for u in sorted(best)], 1)
    fpir = float((B.max(1) >= c.bb.ident_threshold).mean())
    fpir_verif_thr = float((B.max(1) >= c.thr).mean())
    v = best[VINI]
    order = np.argsort(-v)[:10]
    names = c.names[c.test_mask]
    R["open_set"] = {"n_probes": int(len(Y)), "fpir_identificacion": fpir,
                     "fpir_si_se_usara_umbral_verificacion": fpir_verif_thr,
                     "far_contra_vinicius": float((v >= c.thr).mean()),
                     "falsos_vinicius": int((v >= c.thr).sum()),
                     "top10_parecidos_vinicius": [(str(names[i]), round(float(v[i]), 4)) for i in order]}
    log("   ", {k: R["open_set"][k] for k in ("fpir_identificacion", "far_contra_vinicius", "falsos_vinicius")})
    log("    top parecidos:", R["open_set"]["top10_parecidos_vinicius"][:5])
    # montaje de los 5 mas parecidos (para el informe)
    tiles = [c.crops[np.where(c.test_mask)[0][i]] for i in order[:6]]
    cv2.imwrite(str(OUT / "parecidos_vinicius_lfw.png"), np.hstack(tiles))
    return v


# ============================================================================ 6. robustez / PAD
def _landmarks(bb, img):
    return bb.detector.detect(img, strict=False).landmarks


def _sunglasses(bb, img):
    img = img.copy()
    L = _landmarks(bb, img)
    d = np.linalg.norm(L[1] - L[0])
    c = ((L[0] + L[1]) / 2).astype(int)
    cv2.rectangle(img, tuple((c - [int(1.1 * d), int(.35 * d)]).tolist()),
                  tuple((c + [int(1.1 * d), int(.35 * d)]).tolist()), (10, 10, 10), -1)
    return img


def _mask(bb, img):
    img = img.copy()
    L = _landmarks(bb, img)
    d = np.linalg.norm(L[1] - L[0])
    top = int(L[2][1] + .1 * d)
    xc = int(L[2][0])
    cv2.rectangle(img, (xc - int(1.3 * d), top), (xc + int(1.3 * d), top + int(1.8 * d)), (230, 200, 160), -1)
    return img


def _screen(img):
    """Simulacion de foto de una pantalla: perspectiva + gamma + moire + desenfoque."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[.04 * w, .02 * h], [.97 * w, .05 * h], [.95 * w, .97 * h], [.02 * w, .94 * h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(20, 20, 20))
    yy, xx = np.mgrid[:h, :w]
    moire = 12 * np.sin(2 * np.pi * (xx * .9 + yy * .35) / 3.1)
    out = np.clip(255 * (out / 255.0) ** 0.8 + moire[..., None], 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (0, 0), 1.0)


def _print(img):
    """Simulacion de foto impresa: menos contraste/saturacion, tono calido, grano."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= .6
    out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = .75 * out + 40 + [0, 5, 15]
    out += np.random.default_rng(0).normal(0, 6, out.shape)
    return out.clip(0, 255).astype(np.uint8)


def transforms(bb):
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

    return {
        "original": lambda im: im,
        "espejo": lambda im: im[:, ::-1].copy(),
        "rotacion 15": rot(15),
        "desenfoque s=2": lambda im: cv2.GaussianBlur(im, (0, 0), 2),
        "desenfoque s=4": lambda im: cv2.GaussianBlur(im, (0, 0), 4),
        "jpeg q=10": jpeg(10),
        "baja resolucion x0.25": lowres(.25),
        "ruido s=20": lambda im: np.clip(im + np.random.default_rng(0).normal(0, 20, im.shape), 0, 255).astype(np.uint8),
        "oscura (gamma 2.2)": gamma(2.2),
        "sobreexpuesta (gamma 0.4)": gamma(.4),
        "escala de grises": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        "gafas de sol": lambda im: _sunglasses(bb, im),
        "mascarilla": lambda im: _mask(bb, im),
        "PAD: foto de pantalla": _screen,
        "PAD: foto impresa": _print,
    }


def eval_robustness(c):
    log("[6] Robustez / ataques de presentacion simulados (LOO Vinicius + genuinos LFW de inscritos)")
    ids, M = c.gallery()
    out = {}
    lfw_probes = []
    from bigbrother.dataset import enrollment_photos
    # probes extra: fotos del dataset del docente contra su propio template (misma foto -> techo)
    # y las fotos LFW de inscritos (se reconstruye la imagen original de LFW)
    lfw_dir = CACHE.parent / "lfw_home" / "lfw"
    for lfw_name, id_ in RESERVED_MAP.items():
        for p in sorted((lfw_dir / lfw_name).glob("*.jpg"))[:4]:
            lfw_probes.append((id_, p))
    del enrollment_photos
    for name, fn in transforms(c.bb).items():
        acc = fta = 0
        sc = []
        for i, p in enumerate(c.vini_paths):
            try:
                t = c.bb.extraer(fn(load_image(p)))
            except BigBrotherError:
                fta += 1
                continue
            s = np.delete(c.proj(t) @ np.stack(c.vini_refs).T, i).max()
            sc.append(s)
            acc += s >= c.thr
        acc2 = fta2 = 0
        for id_, p in lfw_probes:
            try:
                t = c.bb.extraer(fn(load_image(p)))
            except BigBrotherError:
                fta2 += 1
                continue
            acc2 += c.bb._score_id(id_, c.proj(t)) >= c.thr
        out[name] = {"vini_aceptadas": int(acc), "vini_fta": fta, "vini_n": len(c.vini_paths),
                     "vini_score_medio": float(np.mean(sc)) if sc else None,
                     "lfw_aceptadas": int(acc2), "lfw_fta": fta2, "lfw_n": len(lfw_probes)}
        log(f"   {name:28s} vini {acc}/{len(c.vini_paths)} (FTA {fta})  lfw-inscritos {acc2}/{len(lfw_probes)} (FTA {fta2})")
    R["robustez"] = out
    labels = list(out)
    a = [(out[k]["vini_aceptadas"] + out[k]["lfw_aceptadas"]) / (out[k]["vini_n"] + out[k]["lfw_n"]) for k in labels]
    f = [(out[k]["vini_fta"] + out[k]["lfw_fta"]) / (out[k]["vini_n"] + out[k]["lfw_n"]) for k in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(labels))
    ax.barh(yy, a, label="aceptadas (genuino)")
    ax.barh(yy, f, left=a, label="fallo de adquisicion (FTA)", color="lightgray")
    ax.set(yticks=yy, yticklabels=labels, xlim=(0, 1), xlabel="fraccion de intentos genuinos")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.set_title("Robustez del sistema ante transformaciones y PAD simulados")
    fig.tight_layout()
    fig.savefig(OUT / "robustez.png", dpi=120)
    plt.close(fig)


# ============================================================================ 7. morphing
def eval_morphing(c, lookalike_scores):
    log("[7] Morphing (50/50 en el recorte alineado) Vinicius + complice")
    test_idx = np.where(c.test_mask)[0]
    accomplices = test_idx[np.argsort(-lookalike_scores)[:5]]  # el atacante elige un parecido
    rnd = np.random.default_rng(0).choice(test_idx, 5, replace=False)
    res = {}
    for label, group in (("complice_parecido", accomplices), ("complice_aleatorio", rnd)):
        acc = 0
        n = 0
        for i, p in enumerate(c.vini_paths):
            vcrop = cv2.cvtColor(c.bb.recortar(load_image(p)), cv2.COLOR_BGR2GRAY)
            for j in group:
                for alpha in (0.5,):
                    m = cv2.addWeighted(vcrop, alpha, c.crops[j], 1 - alpha, 0)
                    s = np.delete(c.proj(template_from_crop(c.ex, m)) @ np.stack(c.vini_refs).T, i).max()
                    acc += s >= c.thr
                    n += 1
        res[label] = {"aceptados_como_vinicius": int(acc), "n": n}
    R["morphing"] = res
    log("   ", res)


# ============================================================================ 8. ataques _sd
def eval_sd_attacks(c):
    log("[8] Entradas basura por *_sd (salta el detector)")
    rng = np.random.default_rng(0)
    probes = {f"ruido_{k}": rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for k in range(20)}
    probes.update({f"gris_{v}": np.full((128, 128, 3), v, np.uint8) for v in (0, 127, 255)})
    yy, xx = np.mgrid[:128, :128]
    probes["degradado"] = np.repeat(((xx + yy) % 256).astype(np.uint8)[..., None], 3, 2)
    probes["ajedrez"] = np.repeat((((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)[..., None], 3, 2)
    img = load_image(c.vini_paths[0])
    probes["foto_sin_recortar"] = img
    probes["recorte_desalineado"] = cv2.resize(img, (128, 128))
    out = {}
    for k, im in probes.items():
        try:
            out[k] = c.bb.identificar_sd(im)
        except BigBrotherError as e:
            out[k] = f"rechazado ({type(e).__name__})"
    acc = {k: v for k, v in out.items() if v is not None and not v.startswith("rechazado")}
    # sin el filtro faceness (como estaba antes): cuantas basuras se identificaban
    class _NoFilter:  # filtro que lo acepta todo (como estaba el sistema antes)
        threshold = -np.inf

        @staticmethod
        def score(t):
            return 0.0
    saved, c.bb.faceness = c.bb.faceness, _NoFilter()
    before = {k: c.bb.identificar_sd(im) for k, im in probes.items()}
    c.bb.faceness = saved
    R["ataques_sd"] = {"n": len(out), "resultado": out, "identificados": acc,
                       "identificados_sin_filtro_faceness": {k: v for k, v in before.items() if v}}
    log("   identificados:", acc, "| sin filtro:", R["ataques_sd"]["identificados_sin_filtro_faceness"])


# ============================================================================ 9. ataques _t
def _counts_to_template(C, px=256):
    return (C / px).astype(np.float32).ravel()


def _quantize(H, px=256):
    """histogramas reales (n, bins) -> reticula de cuentas que suman px (resto mayor)."""
    C = np.floor(H * px)
    rem = (px - C.sum(1)).astype(int)
    frac = H * px - C
    for r in range(len(C)):
        if rem[r] > 0:
            C[r, np.argsort(-frac[r])[:rem[r]]] += 1
    return C


def eval_template_attacks(c):
    log("[9] Ataques por *_t")
    out = {}
    # 9a. templates invalidos
    t0 = c.bb.extraer(load_image(c.vini_paths[0]))
    bad = {"dim_incorrecta": t0[:-1], "nan": np.full_like(t0, np.nan), "negativo": -t0,
           "escalado": 2 * t0, "uniforme_no_reticula": np.full_like(t0, 1 / c.ex.bins),
           "perturbacion_1e-3": t0 + 1e-3}
    rej = {}
    for k, b in bad.items():
        try:
            c.bb.verificar_t(VINI, b)
            rej[k] = "ACEPTADO COMO ENTRADA"
        except BigBrotherError as e:
            rej[k] = type(e).__name__
    out["entradas_invalidas"] = rej
    # 9b. "wolf" templates validos: media de LFW cuantizada, y aleatorios
    Xtr = np.asarray(c.X[~c.test_mask][:3000])
    mean_t = _counts_to_template(_quantize(Xtr.mean(0).reshape(-1, c.ex.bins)))
    rng = np.random.default_rng(0)
    rand_ts = [_counts_to_template(rng.multinomial(256, Xtr[i].reshape(-1, c.ex.bins).mean(0), size=c.ex.n_hist))
               for i in range(20)]
    wolves = {"media_lfw": mean_t, **{f"aleatorio_{i}": t for i, t in enumerate(rand_ts)}}
    wr = {}
    for k, t in wolves.items():
        try:
            wr[k] = c.bb.identificar_t(t)
        except BigBrotherError as e:
            wr[k] = f"rechazado ({type(e).__name__})"
    out["wolf_templates"] = wr
    # 9c. hill-climbing si el sistema EXPUSIERA la puntuacion (caja blanca)
    hc = []
    test_idx = np.where(c.test_mask)[0]
    for run in range(3):
        r = np.random.default_rng(run)
        C = np.round(np.asarray(c.X[test_idx[r.integers(len(test_idx))]]).reshape(-1, c.ex.bins) * 256)
        best = c.bb._score_id(VINI, c.proj(_counts_to_template(C)))
        curve = [best]
        q = 0
        while best < c.thr and q < 30000:
            q += 1
            C2 = C.copy()
            for _ in range(4):
                h = r.integers(c.ex.n_hist)
                src = r.choice(np.nonzero(C2[h])[0])
                dst = r.integers(c.ex.bins)
                k = min(C2[h, src], r.integers(1, 9))
                C2[h, src] -= k
                C2[h, dst] += k
            s = c.bb._score_id(VINI, c.proj(_counts_to_template(C2)))
            if s > best:
                C, best = C2, s
            curve.append(best)
        accepted = c.bb.verificar_t(VINI, _counts_to_template(C))
        hc.append({"consultas": q, "score_final": float(best), "aceptado_por_verificar_t": bool(accepted),
                   "curve": curve})
    out["hill_climbing_con_scores"] = [{k: v for k, v in h.items() if k != "curve"} for h in hc]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, h in enumerate(hc):
        ax.plot(h["curve"], label=f"intento {i + 1}")
    ax.axhline(c.thr, c="k", ls="--", label="umbral")
    ax.set(xlabel="consultas al comparador", ylabel="similitud con vinicius_junior",
           title="Hill-climbing SI se expusiera la puntuacion (la API no lo hace)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hill_climbing.png", dpi=120)
    plt.close(fig)
    # 9d. inversion de un template robado (BD + clave): gradiente en GPU/CPU
    out["inversion_template_robado"] = invert_template(c)
    R["ataques_t"] = out
    log("   ", json.dumps({k: v for k, v in out.items()}, default=str)[:1500])


def invert_template(c):
    """Atacante con la BD descifrada (tiene la CLAVE): busca un histograma MLBP valido cuya
    proyeccion coincida con el template robado de Vinicius, y lo presenta por verificar_t."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.tensor(c.bb.projection.W, device=dev)
    mu = torch.tensor(c.bb.projection.mean, device=dev)
    target = torch.tensor(c.vini_refs[0], device=dev)
    fw = torch.tensor(c.bb.faceness.w, device=dev)
    fthr = c.bb.faceness.threshold + 2.0
    # el atacante (caja blanca) parte de la media de caras y penaliza no parecer una cara
    H0 = np.asarray(c.X[~c.test_mask][:2000]).mean(0).reshape(c.ex.n_hist, c.ex.bins)
    logits = torch.tensor(np.log(H0 + 1e-4), device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    t = time.time()
    for _ in range(1500):
        h = torch.softmax(logits, 1).reshape(-1)
        sq = torch.sqrt(h + 1e-12)
        z = (sq - mu) @ W
        face = sq @ fw + c.bb.faceness.b
        loss = -torch.dot(z / z.norm(), target) + 0.05 * torch.relu(fthr - face)
        opt.zero_grad()
        loss.backward()
        opt.step()
    H = torch.softmax(logits, 1).detach().cpu().numpy().astype(np.float64)
    tq = _counts_to_template(_quantize(H))
    others = list(range(1, len(c.vini_refs)))
    y = c.proj(tq)
    try:
        acc, ident = bool(c.bb.verificar_t(VINI, tq)), c.bb.identificar_t(tq)
    except BigBrotherError as e:
        acc, ident = f"rechazado ({type(e).__name__})", None
    return {"dispositivo": dev, "segundos": round(time.time() - t, 1),
            "sim_template_robado": float(y @ c.vini_refs[0]),
            "sim_otros_templates_vini": [round(float(y @ c.vini_refs[k]), 4) for k in others],
            "faceness": c.bb.faceness.score(tq), "faceness_umbral": c.bb.faceness.threshold,
            "aceptado_verificar_t": acc, "identificar_t": ident}


# ============================================================================ 10. BD / inscripcion
def eval_db_and_enrollment(c):
    log("[10] Ataques a la BD y a la inscripcion")
    import shutil
    from bigbrother.system import BigBrother
    out = {}
    db, keys = c.tmp / "db.csv", c.tmp / "keys"
    impostor = load_image(FOTOS_DIR / "dataset" / "idris_elba_2022.jpg")
    for name, call in {
        "inscribir(vinicius_junior, foto_atacante)": lambda: c.bb.inscribir(VINI, impostor),
        "inscribir(..., actualizar=True)": lambda: c.bb.inscribir(VINI, impostor, actualizar=True),
        "inscribir(clon, foto_de_vinicius)": lambda: c.bb.inscribir("clon_vini", load_image(c.vini_paths[1])),
        "inscribir('=HYPERLINK(...)', foto)": lambda: c.bb.inscribir('=HYPERLINK("x")', impostor),
    }.items():
        try:
            call()
            out[name] = "PERMITIDO"
        except BigBrotherError as e:
            out[name] = f"rechazado ({type(e).__name__})"
    backup = (db.read_bytes(), {p.name: p.read_bytes() for p in keys.iterdir()})

    def restore():
        db.write_bytes(backup[0])
        for n, b in backup[1].items():
            (keys / n).write_bytes(b)

    def tamper(kind):
        txt = db.read_text().splitlines()
        if kind == "copiar fila de idris como vinicius":
            row = next(r for r in txt if r.startswith("idris_elba,"))
            txt.append(row.replace("idris_elba", VINI, 1))
        elif kind == "borrar filas de vinicius":
            txt = [r for r in txt if not r.startswith(VINI)]
        elif kind == "modificar un byte del cifrado":
            r = txt[1]
            txt[1] = r[:-3] + ("A" if r[-3] != "A" else "B") + r[-2:]
        elif kind == "intercambiar ids de dos filas":
            a, b = txt[1].split(",", 1), txt[2].split(",", 1)
            txt[1], txt[2] = a[0] + "," + b[1], b[0] + "," + a[1]
        elif kind == "borrar la BD":
            db.unlink()
            return
        db.write_text("\n".join(txt) + "\n")

    for kind in ("copiar fila de idris como vinicius", "borrar filas de vinicius", "modificar un byte del cifrado",
                 "intercambiar ids de dos filas", "borrar la BD"):
        tamper(kind)
        try:
            BigBrother(db_path=db, keys_dir=keys)
            out[f"BD: {kind}"] = "NO DETECTADO"
        except BigBrotherError as e:
            out[f"BD: {kind}"] = f"detectado ({type(e).__name__})"
        restore()
    # BD + clave de otro despliegue
    other = c.tmp / "otra"
    shutil.copytree(keys, other)
    (other / "master.key").unlink()
    try:
        BigBrother(db_path=db, keys_dir=other)
        out["BD con otra clave"] = "NO DETECTADO"
    except BigBrotherError as e:
        out["BD con otra clave"] = f"detectado ({type(e).__name__})"
    # modelos / umbrales (copias en tmp para no tocar models/)
    mdir = c.tmp / "models_copia"
    mdir.mkdir()
    orig = config.MODEL_FILES
    config.MODEL_FILES = tuple(Path(shutil.copy(f, mdir / f.name)) for f in orig)
    try:
        for kind in ("bajar umbral en calibration.json", "borrar el filtro de caras", "cambiar la proyeccion LDA"):
            d2, k2 = c.tmp / f"m_{len(out)}" / "db.csv", c.tmp / f"m_{len(out)}" / "keys"
            BigBrother(db_path=d2, keys_dir=k2)._inscribir_admin([(VINI, c.bb.extraer(load_image(c.vini_paths[0])))])
            cal, face, proj = mdir / "calibration.json", mdir / "faceness.npz", mdir / "mlbp_projection.npz"
            backup = {f: f.read_bytes() for f in (cal, face, proj)}
            if kind.startswith("bajar"):
                cal.write_text(cal.read_text().replace('"threshold": 0.', '"threshold": -0.'))
            elif kind.startswith("borrar"):
                face.unlink()
            else:
                proj.write_bytes(proj.read_bytes()[:-10] + b"0123456789")
            try:
                BigBrother(db_path=d2, keys_dir=k2)
                out[f"Modelos: {kind}"] = "NO DETECTADO"
            except BigBrotherError as e:
                out[f"Modelos: {kind}"] = f"detectado ({type(e).__name__})"
            for f, b in backup.items():
                f.write_bytes(b)
    finally:
        config.MODEL_FILES = orig
    R["bd_e_inscripcion"] = out
    for k, v in out.items():
        log(f"   {k:45s} {v}")


# ============================================================================ 11. detector
def eval_detector(c):
    log("[11] Detector: varias caras / FTA")
    out = {}
    v = load_image(c.vini_paths[4])
    imp = load_image(FOTOS_DIR / "dataset" / "idris_elba_2022.jpg")
    imp = cv2.resize(imp, (int(imp.shape[1] * v.shape[0] / imp.shape[0]), v.shape[0]))
    both = np.hstack([v, imp])
    try:
        out["vinicius + atacante del mismo tamano"] = str(c.bb.identificar(both))
    except BigBrotherError as e:
        out["vinicius + atacante del mismo tamano"] = type(e).__name__
    small = cv2.resize(v, None, fx=.3, fy=.3)
    canvas = imp.copy()
    canvas[:small.shape[0], :small.shape[1]] = small
    try:
        out["atacante grande + vinicius pequeno"] = str(c.bb.identificar(canvas))
    except BigBrotherError as e:
        out["atacante grande + vinicius pequeno"] = type(e).__name__
    z = np.load(CACHE)
    out["lfw_fta_calidad"] = float(1 - z["quality_ok"].mean())
    out["lfw_no_detectadas"] = int(z["n_total"] - len(z["names"]))
    R["detector"] = out
    log("   ", out)


def plot_scores(c, gen, imp, loo, twins, reserved):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(imp, 120, density=True, alpha=.45, label="impostores LFW")
    ax.hist(gen, 120, density=True, alpha=.45, label="genuinos LFW")
    for vals, lab, mk in ((loo, "Vinicius LOO", "o"), (twins, "gemelo -> victima", "x"),
                          (reserved, "inscritos (foto LFW)", "|")):
        ax.plot(vals, np.full(len(vals), -0.1 - 0.15 * ("|ox".index(mk))), mk, ms=8, label=lab)
    ax.axvline(c.thr, c="k", ls="--", label=f"umbral = {c.thr:.3f}")
    ax.set(xlabel="similitud coseno", title="Puntuaciones: genuinos, impostores, gemelos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "puntuaciones.png", dpi=120)
    plt.close(fig)


def eval_timing(c):
    img = load_image(c.vini_paths[0])
    t = time.time()
    for _ in range(10):
        c.bb.verificar(VINI, img)
    R["tiempo_verificar_ms"] = round((time.time() - t) * 100, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    R["umbral"] = c.thr
    R["umbral_identificacion"] = c.bb.ident_threshold
    R["calibracion"] = json.loads(config.CALIBRATION_PATH.read_text())
    gen, imp = eval_lfw(c)
    _, loo = eval_vinicius(c)
    reserved = eval_reserved(c)
    twins = eval_twins(c)
    v = eval_open_set(c)
    plot_scores(c, gen, imp, loo, twins, reserved)
    eval_robustness(c)
    eval_morphing(c, v)
    eval_sd_attacks(c)
    eval_template_attacks(c)
    eval_detector(c)
    eval_timing(c)
    eval_db_and_enrollment(c)
    (OUT / "resultados.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    log("OK ->", OUT / "resultados.json")


if __name__ == "__main__":
    main()
