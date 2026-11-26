"""A2 y A3 de scripts/ataques_literatura.py repetidos sin que la foto de partida de la victima
este inscrita (leave-one-out).

    python scripts/ataques_loo.py     # -> docs/resultados/ataques_loo.json

En ataques_literatura.py la "foto publica" de la victima es vini_01.jpg, que tambien es uno de
los 5 templates inscritos de vinicius_junior: el atacante parte de una copia exacta de la BD
(similitud 1,0). Aqui, para cada una de las 5 fotos de Vinicius, se construye la BD SIN esa
foto y se usa como foto publica, que es el caso realista. Se mide:
  - A2: frontera alpha* de la mezcla uniforme y similitud final del template con el atacante.
  - A3: para cada alpha, si la FOTO del morph verifica como la victima por el pipeline completo,
    si el poisoning por actualizar=True se acepta y si despues la cara pura del atacante
    (paula_02) verifica como la victima.
Trabaja sobre BDs temporales con clave propia (no toca data/db.csv).
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ataques_literatura as al  # noqa: E402
from bigbrother.dataset import enrollment_photos  # noqa: E402
from bigbrother.errors import BigBrotherError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402
from bigbrother.system import BigBrother  # noqa: E402

OUT = ROOT / "docs" / "resultados" / "ataques_loo.json"
ALPHAS = [round(a, 1) for a in np.arange(0.0, 1.01, 0.1)]


def build_without(photos, held, d):
    bb = BigBrother(db_path=d / "db.csv", keys_dir=d / "keys")
    bb._inscribir_admin([(i, bb.extraer(load_image(p))) for i, p in photos if p != held])
    return bb


def copy_db(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir)
    return BigBrother(db_path=dst_dir / "db.csv", keys_dir=dst_dir / "keys")


def main():
    photos = enrollment_photos()
    vini = [p for i, p in photos if i == al.VICTIM]
    base = Path(tempfile.mkdtemp(prefix="bb_loo_"))
    out = {"victima": al.VICTIM, "atacante": "paula_01 (morph) / paula_02 (cara pura)",
           "umbral_verificacion": None, "por_foto_publica": {}}
    for held in vini:
        d = base / held.stem / "base"
        d.mkdir(parents=True)
        bb = build_without(photos, held, d)
        out["umbral_verificacion"] = bb.threshold
        t_v = bb.extraer(str(held))
        t_a = bb.extraer(str(al.PAULA_01))
        r = {"score_foto_publica_vs_galeria": round(bb._score_id(al.VICTIM, bb._proj(t_v)), 4)}
        a2, _ = al.a2_hillclimb_bb(bb, t_v, t_a)
        r["A2"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a2.items()
                   if k in ("consultas_totales", "alpha_frontera_uniforme", "alpha_medio_final",
                            "sim_template_atacante", "score_white_box_final", "identificar_t")}
        cv, ca = bb.recortar(str(held)), bb.recortar(str(al.PAULA_01))
        a3 = []
        for a in ALPHAS:
            m = cv2.addWeighted(cv, 1 - a, ca, a, 0.0)
            e = {"alpha": a}
            try:
                e["morph_foto_verifica_como_victima"] = bool(bb.verificar(al.VICTIM, m))
            except BigBrotherError as ex:
                e["morph_foto_verifica_como_victima"] = type(ex).__name__
            bbp = copy_db(d, base / held.stem / f"p{a:.1f}")
            try:
                bbp.inscribir_t(al.VICTIM, bbp.extraer_sd(m), actualizar=True)
                e["poisoning_aceptado"] = True
                e["atacante_puro_verifica_como_victima"] = bool(bbp.verificar(al.VICTIM, str(al.PAULA_02)))
            except BigBrotherError as ex:
                e["poisoning_aceptado"] = False
                e["motivo"] = type(ex).__name__
            a3.append(e)
        r["A3"] = a3
        out["por_foto_publica"][held.name] = r
        print(held.name, json.dumps({k: v for k, v in r.items() if k != "A3"}), flush=True)
    shutil.rmtree(base, ignore_errors=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
