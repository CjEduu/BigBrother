import argparse

import cv2

from .system import BigBrother


def main():
    p = argparse.ArgumentParser(prog="bigbrother")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inscribir", "verificar"):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.add_argument("foto")
    sub.add_parser("identificar").add_argument("foto")
    a = p.parse_args()

    bb = BigBrother()
    img = cv2.imread(a.foto)
    if img is None:
        raise SystemExit(f"No se pudo leer {a.foto}")
    if a.cmd == "inscribir":
        bb.inscribir(a.id, img)
        print("inscrito", a.id)
    elif a.cmd == "verificar":
        print(bb.verificar(a.id, img))
    else:
        print(bb.identificar(img))


if __name__ == "__main__":
    main()
