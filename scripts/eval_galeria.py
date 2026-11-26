"""FAR real entre las personas inscritas (galeria de fotos web de alta resolucion).

    python scripts/eval_galeria.py     # -> docs/resultados/galeria_impostores.json

El umbral de verificacion se calibro en LFW (FAR 0,1 %), pero las fotos de la galeria son de
otro dominio (alta resolucion, retratos posados/retocados). Aqui se mide la FAR que da ese
umbral comparando entre si los templates inscritos de personas DISTINTAS, y cuantas parejas
legitimas superarian el umbral de deduplicacion (no podrian inscribirse ambas por la API).
Trabaja sobre una BD temporal con clave propia (no toca data/db.csv).
"""
import json
import sys
import tempfile
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_db import build  # noqa: E402

OUT = ROOT / "docs" / "resultados" / "galeria_impostores.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="bb_galeria_"))
    bb = build(tmp / "db.csv", tmp / "keys", verbose=False)
    rows = bb.db.all()
    ids = [i for i, _ in rows]
    M = np.stack([r for _, r in rows])
    S = M @ M.T

    # pares de templates de personas distintas
    tpl = [(float(S[a, b]), ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))
           if ids[a] != ids[b]]
    s_tpl = np.array([x[0] for x in tpl])
    # pares de identidades (maximo sobre sus templates, como hace verificar)
    uniq = sorted(set(ids))
    idx = {u: [k for k, i in enumerate(ids) if i == u] for u in uniq}
    ide = [(float(S[np.ix_(idx[a], idx[b])].max()), a, b)
           for k, a in enumerate(uniq) for b in uniq[k + 1:]]
    s_ide = np.array([x[0] for x in ide])

    def resumen(s):
        n = len(s)
        out = {"n_pares": n, "media": round(float(s.mean()), 4)}
        for nombre, thr in (("verificacion", bb.threshold), ("identificacion", bb.ident_threshold),
                            ("deduplicacion", bb.dedup_threshold)):
            k = int((s >= thr).sum())
            out[nombre] = {"umbral": round(thr, 4), "pares_por_encima": k,
                           "tasa": round(k / n, 4), "ic95": wilson(k, n)}
        return out

    res = {
        "far_nominal_lfw": 0.001,
        "pares_de_templates": resumen(s_tpl),
        "pares_de_identidades": resumen(s_ide),
        "parejas_sobre_dedup": [(a, b, round(s, 4)) for s, a, b in sorted(ide, reverse=True)
                                if s >= bb.dedup_threshold],
        "top10_parejas": [(a, b, round(s, 4)) for s, a, b in sorted(ide, reverse=True)[:10]],
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
