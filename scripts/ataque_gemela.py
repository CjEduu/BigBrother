"""Ataque con la gemela: fotos de Paula Etxeberria contra `aitana_etxebarria` (inscrita).

    python scripts/ataque_gemela.py     # -> docs/resultados/gemela_paula.json

Trabaja sobre una BD temporal con clave propia (no toca data/db.csv). Para cada foto informa
de lo que ve un atacante (verificar/identificar) y, para el informe, de las puntuaciones
internas y de cuantas identidades inscritas acepta esa foto en verificacion.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_db import build  # noqa: E402

from bigbrother.errors import BigBrotherError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402

FOTOS = ROOT / "data" / "ataques" / "gemela_paula"
VICTIMA = "aitana_etxebarria"
OUT = ROOT / "docs" / "resultados" / "gemela_paula.json"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="bb_gemela_"))
    bb = build(tmp / "db.csv", tmp / "keys", verbose=False)
    res = {"victima": VICTIMA, "umbral_verificacion": bb.threshold,
           "umbral_identificacion": bb.ident_threshold, "fotos": {}}
    for p in sorted(FOTOS.glob("*.jpg")):
        img = load_image(p)
        try:
            s = bb._scores(bb._proj(bb.extraer(img)))
        except BigBrotherError as e:
            res["fotos"][p.name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        orden = sorted(s, key=s.get, reverse=True)
        acepta = [i for i in orden if s[i] >= bb.threshold]
        res["fotos"][p.name] = {
            "verificar_victima": bb.verificar(VICTIMA, img),
            "identificar": bb.identificar(img),
            "score_victima": round(s[VICTIMA], 4),
            "rank_victima": orden.index(VICTIMA) + 1,
            "top5": [(i, round(s[i], 4)) for i in orden[:5]],
            "identidades_que_verifica": acepta,
        }
        print(p.name, json.dumps(res["fotos"][p.name], ensure_ascii=False))
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado", OUT)


if __name__ == "__main__":
    main()
