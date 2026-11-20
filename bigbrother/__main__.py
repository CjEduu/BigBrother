"""CLI de BigBrother.

Ejemplos:
    python -m bigbrother listar
    python -m bigbrother verificar vinicius_junior foto.jpg
    python -m bigbrother identificar foto.jpg
    python -m bigbrother inscribir pepe foto.jpg
    python -m bigbrother recortar foto.jpg cara.png          # entrada para *_sd
    python -m bigbrother verificar_sd vinicius_junior cara.png
    python -m bigbrother extraer foto.jpg template.npy        # entrada para *_t
    python -m bigbrother verificar_t vinicius_junior template.npy

Salida: una linea JSON. Codigo de salida: 0 = aceptado / ok, 1 = rechazado / desconocido,
2 = error (sin cara, entrada invalida, BD manipulada, inscripcion rechazada...).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_template(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None, dtype=np.float64).ravel()


def _out(obj, code=0):
    print(json.dumps(obj, ensure_ascii=False))
    return code


def main(argv=None):
    p = argparse.ArgumentParser(prog="bigbrother", description="Reconocimiento facial MLBP")
    sub = p.add_subparsers(dest="cmd", required=True)
    for suf, arg, hlp in (("", "foto", "foto completa"), ("_sd", "foto", "recorte de cara (ver 'recortar')"),
                          ("_t", "template", "template .npy/.csv/.txt (ver 'extraer')")):
        s = sub.add_parser("inscribir" + suf)
        s.add_argument("id")
        s.add_argument(arg, help=hlp)
        s.add_argument("--actualizar", action="store_true",
                       help="anadir otra muestra a un id existente (debe verificar contra el)")
        s = sub.add_parser("verificar" + suf)
        s.add_argument("id")
        s.add_argument(arg, help=hlp)
        sub.add_parser("identificar" + suf).add_argument(arg, help=hlp)
    s = sub.add_parser("recortar", help="foto -> recorte alineado para *_sd")
    s.add_argument("foto")
    s.add_argument("salida")
    s = sub.add_parser("extraer", help="foto -> template MLBP (.npy o .csv)")
    s.add_argument("foto")
    s.add_argument("salida")
    s.add_argument("--sd", action="store_true", help="la foto ya es un recorte de cara")
    sub.add_parser("listar", help="ids inscritos")
    s = sub.add_parser("construir_bd", help="(re)construye la BD con data/fotos")
    s.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    from . import BigBrotherError, config
    from .imageio import load_image

    try:
        if a.cmd == "construir_bd":
            sys.path.insert(0, str(config.ROOT / "scripts"))
            from build_db import build
            bb = build(config.DB_PATH, config.KEYS_DIR, force=a.force, verbose=False)
            return _out({"ok": True, "templates": len(bb.db), "identidades": len(bb.identidades())})

        from .system import BigBrother
        bb = BigBrother()
        if a.cmd == "listar":
            return _out({"identidades": bb.identidades()})
        if a.cmd == "recortar":
            import cv2
            cv2.imwrite(a.salida, bb.recortar(load_image(a.foto)))
            return _out({"ok": True, "salida": a.salida})
        if a.cmd == "extraer":
            img = load_image(a.foto)
            t = bb.extraer_sd(img) if a.sd else bb.extraer(img)
            if a.salida.endswith(".npy"):
                np.save(a.salida, t)
            else:
                np.savetxt(a.salida, t[None], delimiter=",", fmt="%.8g")
            return _out({"ok": True, "dim": int(t.size), "salida": a.salida})

        base, _, suf = a.cmd.partition("_")
        x = _load_template(a.template) if suf == "t" else load_image(a.foto)
        fn = getattr(bb, a.cmd)
        if base == "inscribir":
            fn(a.id, x, actualizar=a.actualizar)
            return _out({"inscrito": a.id})
        if base == "verificar":
            ok = fn(a.id, x)
            return _out({"id": a.id, "verificado": ok}, 0 if ok else 1)
        r = fn(x)
        return _out({"identidad": r}, 0 if r else 1)
    except BigBrotherError as e:
        return _out({"error": type(e).__name__, "detalle": str(e)}, 2)
    except (FileNotFoundError, ValueError) as e:
        return _out({"error": type(e).__name__, "detalle": str(e)}, 2)


if __name__ == "__main__":
    sys.exit(main())
