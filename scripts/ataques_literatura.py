"""Ataques empiricos de BigBrother, siguiendo el mapeo de docs/ATAQUES_LITERATURA.md (H1-H5).

    python scripts/ataques_literatura.py
    -> docs/resultados/ataques_literatura.json

Ataques (referencias en docs/ATAQUES_LITERATURA.md):
  A1 (H1, Ratha 2001 puntos 3/5/8)  Troyanos de SOFTWARE: la huella de keys/db.state cubre la
      BD y models/ pero NO el codigo. Tres variantes (anular decision, Troyano extractor,
      control con models/) + prototipo de mitigacion (digest del codigo en la huella).
  A2 (H3, Dong et al. CVPR 2019)     Hill-climbing de CAJA NEGRA con feedback binario (solo
      aceptar/rechazar de verificar_t) sobre la reticula de histogramas MLBP validos.
  A3 (H2, Ferrara et al. IJCB 2014)  Morphing por la via de la INSCRIPCION (magic passport)
      + cadena de poisoning (V16): morph en la galeria de la victima -> cara pura del atacante.
  A4 (H4, ISO/IEC 24745)            Rotacion de clave != revocacion: tras force-rebuild con
      clave nueva, el template robado/reconstruido sigue verificando.
  A5 (Yang et al. 2019)              Dodging del detector: parche fisico (cartulina) sobre los
      ojos -> NoFaceDetected (el genuino evade identificar()).

Todo el ataque se ejecuta en copias (sandbox /tmp/bb_sbx, arbol /tmp/bb_t1): la BD original
de la imagen no se modifica salvo para A4, que la reconstruye con --force en su sandbox.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from bigbrother import config  # noqa: E402
from bigbrother.errors import BigBrotherError, IntegrityError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402
from bigbrother.system import BigBrother  # noqa: E402

VICTIM = "vinicius_junior"
VINI_DIR = config.DATA_DIR / "fotos" / "vinicius_junior"
PAULA_01 = config.DATA_DIR / "ataques" / "gemela_paula" / "paula_01.jpg"   # cara del atacante
PAULA_02 = config.DATA_DIR / "ataques" / "gemela_paula" / "paula_02.jpg"  # 2a foto del atacante
OUT = config.DATA_DIR.parent / "docs" / "resultados" / "ataques_literatura.json"
SBX = Path("/tmp/bb_sbx")
T1 = Path("/tmp/bb_t1")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- sandbox
def fresh_db_copy(name):
    """Copia de la BD original (db.csv + keys) para experimentar sin tocarla."""
    d = SBX / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(config.DB_PATH, d / "db.csv")
    shutil.copytree(config.KEYS_DIR, d / "keys")
    return d / "db.csv", d / "keys"


def open_bb(db_path, keys_dir):
    return BigBrother(db_path=db_path, keys_dir=keys_dir)


# --------------------------------------------------------------------------- A2
class Oracle:
    """Unica informacion del sistema que ve el atacante de A2: un bool por consulta."""

    def __init__(self, bb, id_):
        self.bb, self.id, self.n = bb, id_, 0

    def __call__(self, t):
        self.n += 1
        try:
            return bool(self.bb.verificar_t(self.id, t))
        except BigBrotherError:
            return False  # para el atacante, error = no aceptado


def to_counts(t, ex):
    return np.round(np.asarray(t).reshape(ex.n_hist, ex.bins) * 256).astype(np.int64)


def from_counts(C):
    return (C.astype(np.float32) / 256.0).ravel()


def blend_counts(Cv, Ca, alphas):
    """Mezcla de conteos por region con correccion de resto mayor: el resultado es un
    histograma MLBP exactamente valido (cada region suma 256)."""
    a = np.broadcast_to(np.asarray(alphas, float), (Cv.shape[0],))
    W = (1 - a)[:, None] * Cv + a[:, None] * Ca
    F = np.floor(W).astype(np.int64)
    rem = 256 - F.sum(1)
    fr = W - F
    for i in np.nonzero(rem)[0]:
        F[i, np.argsort(-fr[i])[: rem[i]]] += 1
    return F


def a2_hillclimb_bb(bb, t_v, t_a):
    """Dong et al. (2019) adaptado: empieza con un template ACEPTADO (foto publica de la
    victima, como asume la literatura decision-based) y empuja region a region hacia la cara
    del atacante usando solo el bool de verificar_t."""
    ex = bb.extractor
    Cv, Ca = to_counts(t_v, ex), to_counts(t_a, ex)
    proj = bb.projection
    y_a, y_v = proj(t_a), proj(t_v)
    oracle = Oracle(bb, VICTIM)
    t0 = oracle.n
    assert oracle(from_counts(Cv)), "la foto publica de la victima debe verificar"
    assert not oracle(from_counts(Ca)), "la cara del atacante no debe verificar"

    # fase 1: frontera global (mezcla uniforme) por biseccion
    lo, hi = 0.0, None
    for a in np.arange(0.02, 1.0, 0.02):
        if oracle(from_counts(blend_counts(Cv, Ca, a))):
            lo = float(a)
        else:
            hi = float(a)
            break
    if hi is None:
        hi = 1.0
    n1 = oracle.n
    for _ in range(9):
        mid = (lo + hi) / 2
        if oracle(from_counts(blend_counts(Cv, Ca, mid))):
            lo = mid
        else:
            hi = mid
    alpha_star = lo
    log(f"  fase 1: frontera alpha*={alpha_star:.3f} tras {oracle.n - t0} consultas bool")

    # fase 2: hill-climb por regiones (cada region del template tiene su propio alpha)
    rng = np.random.default_rng(0)
    alphas = np.full(ex.n_hist, alpha_star)
    C = blend_counts(Cv, Ca, alpha_star)
    n2 = oracle.n
    accepted, streak = 0, 0
    while oracle.n - n2 < 4000 and streak < 1500:
        r = int(rng.integers(ex.n_hist))
        moved = False
        for d in (0.02, 0.08, 0.30):
            anew = min(1.0, alphas[r] + d)
            if anew <= alphas[r] + 1e-12:
                continue
            Cn = C.copy()
            Cn[r] = blend_counts(Cv[r][None], Ca[r][None], anew)[0]
            if oracle(from_counts(Cn)):
                alphas[r], C, accepted = anew, Cn, accepted + 1
                streak, moved = 0, True
                break
        if not moved:
            streak += 1
    t_star = from_counts(C)
    y_star = proj(t_star)
    res = {
        "consultas_totales": oracle.n,
        "fase0_sanity": 2, "fase1_biseccion": n1 - t0 - 2, "fase2_hillclimb": oracle.n - n2,
        "alpha_frontera_uniforme": round(alpha_star, 4),
        "pasos_aceptados_fase2": accepted,
        "alpha_medio_final": float(np.mean(alphas)),
        "fraccion_regiones_alpha_max": float(np.mean(alphas >= 0.999)),
        "sim_template_atacante": float(y_star @ y_a),   # cuanto se parece ya a la cara del atacante
        "sim_template_victima_foto": float(y_star @ y_v),
        "sim_base_victima_atacante": float(y_v @ y_a),
        "score_white_box_final": float(bb._score_id(VICTIM, y_star)),  # solo analisis
        "identificar_t": bb.identificar_t(t_star),
    }
    log(f"  fase 2: {accepted} pasos aceptados, {oracle.n - n2} consultas -> "
        f"sim(atacante)={res['sim_template_atacante']:.3f} "
        f"(base {res['sim_base_victima_atacante']:.3f})")
    return res, t_star


# --------------------------------------------------------------------------- A3
def a3_morphing(bb, t_v, t_a):
    """Magic passport: morph (1-alfa)*victima + alfa*atacante (alfa=0 victima pura). Para cada alfa:
    (1) inscripcion del morph como id nuevo (tropieza con la deduplicacion?),
    (2) verifican ambos padres contra el id del morph,
    (3) poisoning: actualizar=True mete el morph en la galeria de la VICTIMA (sin dedup)
        y despues la cara pura del atacante debe verificar como la victima."""
    cv = bb.recortar(str(VINI_DIR / "vini_01.jpg"))
    ca = bb.recortar(str(PAULA_01))
    y_a = bb.projection(t_a)
    y_vini06 = bb.projection(bb.extraer(str(VINI_DIR / "vini_06.jpg")))
    y_paula02 = bb.projection(bb.extraer(str(PAULA_02)))
    vini_probe = str(VINI_DIR / "vini_06.jpg")
    rows = []
    for a in np.arange(0.0, 1.01, 0.1):
        a = round(float(a), 2)
        m = cv2.addWeighted(cv, 1 - a, ca, a, 0.0)  # morph por mezcla de recortes alineados
        t_m = bb.extraer_sd(m)
        try:
            y_m = bb._proj(t_m)
            s_vict = float(bb._score_id(VICTIM, y_m))
            s_atac = float(y_m @ y_a)
        except BigBrotherError as e:  # p.ej. el morph no pasa el filtro de cara
            s_vict = s_atac = None
        entry = {
            "alpha": a,  # 0 = victima pura, 1 = atacante puro
            "score_morph_victima": s_vict,          # solo analisis (caja blanca)
            "score_morph_atacante": s_atac,
            "score_probe_victima_vini06": (float(y_m @ y_vini06) if s_vict is not None else None),
            "score_probe_atacante_paula02": (float(y_m @ y_paula02) if s_atac is not None else None),
        }
        db, keys = fresh_db_copy(f"m{a:.1f}")
        bbm = open_bb(db, keys)
        # (extra) el morph como FOTO presentada al sensor (pipeline completo) contra la galeria original
        try:
            entry["morph_foto_verifica_como_victima"] = bool(bb.verificar(VICTIM, m))
        except BigBrotherError as e:
            entry["morph_foto_verifica_como_victima"] = f"{type(e).__name__}"
        # (1) inscripcion como id nuevo
        try:
            bbm.inscribir_t("sospechoso", t_m)
            entry["inscrito_como_id_nuevo"] = True
            entry["victima_verifica_como_sospechoso"] = bbm.verificar("sospechoso", vini_probe)
            entry["atacante_verifica_como_sospechoso"] = bbm.verificar("sospechoso", str(PAULA_02))
        except BigBrotherError as e:
            entry["inscrito_como_id_nuevo"] = False
            entry["motivo_rechazo"] = f"{type(e).__name__}: {e}"
        # (3) poisoning: colar el morph en la galeria de la victima (sin dedup) y probar la cara pura
        db2, keys2 = fresh_db_copy(f"p{a:.1f}")
        bbp = open_bb(db2, keys2)
        try:
            bbp.inscribir_t(VICTIM, t_m, actualizar=True)
            entry["poisoning_aceptado"] = True
            entry["atacante_puro_verifica_como_victima"] = bbp.verificar(VICTIM, str(PAULA_02))
        except BigBrotherError as e:
            entry["poisoning_aceptado"] = False
            entry["motivo_poisoning"] = f"{type(e).__name__}"
        rows.append(entry)
        log(f"  alpha={a:.1f} score(v)={entry['score_morph_victima']:.3f} "
            f"nuevo_id={entry['inscrito_como_id_nuevo']} "
            f"poison={entry.get('poisoning_aceptado')} "
            f"atacante_puro={entry.get('atacante_puro_verifica_como_victima')}")
    return rows


# --------------------------------------------------------------------------- A4
def a4_revocacion(t_star):
    """La clave rota pero el template robado no: force-rebuild (clave nueva) y el template
    reconstruido en A2 sigue verificando (no hay frescura ni revocabilidad, ISO 24745)."""
    from build_db import build
    old_bb = open_bb(*fresh_db_copy("a4_old"))
    y_old = old_bb.db.get(VICTIM)
    raw_old = (config.DB_PATH).read_text(encoding="utf-8").splitlines()
    d4 = SBX / "a4_new"
    build(d4 / "db.csv", d4 / "keys", force=True)   # clave nueva, mismos ficheros de fotos
    new_bb = open_bb(d4 / "db.csv", d4 / "keys")
    y_new = new_bb.db.get(VICTIM)
    raw_new = (d4 / "db.csv").read_text(encoding="utf-8").splitlines()
    return {
        "template_robado_verifica_tras_rotacion": bool(new_bb.verificar_t(VICTIM, t_star)),
        "templates_identicos": bool(all(np.allclose(a, b) for a, b in zip(y_old, y_new))),
        "ciphertexts_distintos": raw_old[1][:40] != raw_new[1][:40],
        "n_templates_victima": len(y_old),
    }


# --------------------------------------------------------------------------- A5
def a5_dodging(bb):
    """Un parche fisico (cartulina) sobre los ojos desactiva al detector: el genuino evita
    identificar(). Control: el mismo parche en la frente no afecta."""
    img = load_image(str(VINI_DIR / "vini_06.jpg"))
    face = bb.detector.detect(img)
    re_, le = face.landmarks[0], face.landmarks[1]
    cx, cy = (re_[0] + le[0]) / 2, (re_[1] + le[1]) / 2
    w = abs(le[0] - re_[0])
    out = {"sin_parche": bool(bb.verificar(VICTIM, img))}
    for name, (x1, y1, x2, y2) in {
        "parche_ojos": (int(cx - w * 1.2), int(cy - w * 0.7), int(cx + w * 1.2), int(cy + w * 0.7)),
        "parche_frente_control": (int(cx - w), int(cy - w * 2.0), int(cx + w), int(cy - w * 1.1)),
    }.items():
        p = img.copy()
        cv2.rectangle(p, (x1, y1), (x2, y2), (0, 0, 0), -1)
        try:
            out[name] = f"verificado={bb.verificar(VICTIM, p)}"
        except BigBrotherError as e:
            out[name] = f"{type(e).__name__}"
    return out


# --------------------------------------------------------------------------- A1
def _patch(path, old, new):
    src = Path(path).read_text(encoding="utf-8")
    assert old in src, f"ancla no encontrada en {path}"
    Path(path).write_text(src.replace(old, new, 1), encoding="utf-8")


def _cli(tree, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-m", "bigbrother", *args], cwd=tree,
                       capture_output=True, text=True, env=env)
    out = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    try:
        out = json.loads(out)
    except ValueError:
        pass
    return {"cmd": " ".join(args), "rc": p.returncode, "salida": out}


def a1_troyanos(t_v):
    """Ratha puntos 3/5/8: manipular el CODIGO (no models/) no invalida la huella de la BD.
    Se ejecuta en una copia del arbol (/tmp/bb_t1) con su propia BD ya construida."""
    shutil.rmtree(T1, ignore_errors=True)
    shutil.copytree(ROOT, T1, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "docs"))
    P = T1 / "bigbrother"
    sysp = P / "system.py"
    sec = P / "security.py"
    cal = T1 / "models" / "calibration.json"
    cal_bytes = cal.read_bytes()
    sys_bytes = sysp.read_text(encoding="utf-8")
    np.save(T1 / "t_vini.npy", t_v)
    r = {}

    r["0_baseline"] = {
        "nota": "arbol copiado, BD intacta: el atacante (paula_01) NO verifica como la victima",
        "verificar": _cli(T1, "verificar", VICTIM, str(PAULA_01)),
        "arranque": _cli(T1, "listar"),
    }

    # punto 8 (Ratha): anular la decision tras la comparacion
    _patch(sysp, "        return self._score_id(id, y) >= self.threshold",
           "        return True  # TROYANO")
    r["1_anular_decision_R8"] = {
        "verificar": _cli(T1, "verificar", VICTIM, str(PAULA_01)),
        "arranque_sin_deteccion": _cli(T1, "listar"),
    }
    sysp.write_text(sys_bytes, encoding="utf-8")

    # punto 3 (Ratha): Troyano extractor que fabrica los rasgos de la victima
    _patch(sysp, "        return self.extractor.extract(normalize_face(as_bgr(cara)))",
           "        return np.load('/tmp/bb_t1/t_vini.npy')  # TROYANO")
    r["2_troyano_extractor_R3"] = {
        "verificar": _cli(T1, "verificar", VICTIM, str(PAULA_01)),
        "arranque_sin_deteccion": _cli(T1, "listar"),
    }
    sysp.write_text(sys_bytes, encoding="utf-8")

    # control: manipular models/ SI se detecta
    c = json.loads(cal_bytes)
    c["threshold"] = 0.02
    cal.write_text(json.dumps(c), encoding="utf-8")
    r["3_control_modelos"] = {
        "nota": "editar models/calibration.json (bajar el umbral) SI invalida la huella",
        "arranque": _cli(T1, "listar"),
    }
    cal.write_bytes(cal_bytes)

    # mitigacion (prototipo): digest del codigo en la huella
    _patch(sec, "    h = hashlib.sha256()\n    for f in files or config.MODEL_FILES:",
           "    files = list(files or config.MODEL_FILES) + sorted(config.ROOT.glob('bigbrother/*.py'))\n"
           "    h = hashlib.sha256()\n    for f in files:")
    r["4_mitigacion_digest_codigo"] = {
        "rebuild": _cli(T1, "construir_bd", "--force"),
        "arranque": _cli(T1, "listar"),
    }
    # y ahora el Troyano del punto 8 SI se detecta
    _patch(sysp, "        return self._score_id(id, y) >= self.threshold",
           "        return True  # TROYANO")
    r["5_troyano_con_mitigacion"] = {"arranque": _cli(T1, "listar")}
    sysp.write_text(sys_bytes, encoding="utf-8")
    return r


# --------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    SBX.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    db, keys = fresh_db_copy("main")
    bb = open_bb(db, keys)
    vini_fotos = sorted(str(p) for p in VINI_DIR.glob("*.jpg"))
    t_v = bb.extraer(vini_fotos[0])      # foto PUBLICA de la victima (arranque aceptado, Dong et al.)
    t_a = bb.extraer(str(PAULA_01))      # cara del atacante (no inscrita)

    log("A2: hill-climbing binario sobre la reticula (solo bool de verificar_t)")
    a2, t_star = a2_hillclimb_bb(bb, t_v, t_a)
    log("A3: morphing por inscripcion + poisoning (magic passport)")
    a3 = a3_morphing(bb, t_v, t_a)
    log("A4: rotacion de clave vs revocacion")
    a4 = a4_revocacion(t_star)
    log("A5: dodging del detector")
    a5 = a5_dodging(bb)
    log("A1: troyanos de software en copia del arbol")
    a1 = a1_troyanos(t_v)

    results = {
        "meta": {
            "victima": VICTIM, "atacante": "paula_01 (gemela de una inscrita, no inscrita)",
            "umbral_verificacion": config.THRESHOLD,
            "umbral_identificacion": config.IDENT_THRESHOLD,
            "umbral_dedup": config.DEDUP_THRESHOLD,
            "dim_template": bb.extractor.dim,
            "identidades": len(bb.identidades()),
        },
        "a1_troyanos_software": a1,
        "a2_hillclimb_binario": a2,
        "a3_morphing_inscripcion": a3,
        "a4_rotacion_sin_revocacion": a4,
        "a5_dodging_detector": a5,
        "segundos": round(time.time() - t_start, 1),
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"-> {OUT} ({results['segundos']} s)")


if __name__ == "__main__":
    main()
