"""Construye la BD inicial: las 49 identidades del docente + vinicius_junior (50 en total).

    python scripts/build_db.py            # crea data/db.csv y keys/ (clave nueva)
    python scripts/build_db.py --force    # borra la BD y la clave existentes y la regenera
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigbrother import config  # noqa: E402
from bigbrother.dataset import enrollment_photos  # noqa: E402
from bigbrother.errors import BigBrotherError  # noqa: E402
from bigbrother.imageio import load_image  # noqa: E402


def build(db_path, keys_dir, fotos_dir=None, force=False, verbose=True):
    from bigbrother.system import BigBrother
    db_path, keys_dir = Path(db_path), Path(keys_dir)
    if db_path.exists() or (keys_dir / "db.state").exists():
        if not force:
            raise SystemExit(f"Ya existe una BD en {db_path} (usa --force para regenerarla)")
        db_path.unlink(missing_ok=True)
        shutil.rmtree(keys_dir, ignore_errors=True)
    bb = BigBrother(db_path=db_path, keys_dir=keys_dir)
    items, warnings = [], []
    photos = enrollment_photos(fotos_dir) if fotos_dir else enrollment_photos()
    for id_, path in photos:
        img = load_image(path)
        try:
            crop = bb.recortar(img)
        except BigBrotherError as e:
            # Inscripcion supervisada: el operador acepta la cara principal aunque no pase
            # el control estricto (se registra el aviso).
            crop = bb.recortar(img, strict=False)
            warnings.append(f"{path.name}: {e} -> usada la cara mas grande")
        items.append((id_, bb.extraer_sd(crop)))
    bb._inscribir_admin(items)
    if verbose:
        for w in warnings:
            print("AVISO", w)
        print(f"BD creada: {len(bb.db)} templates, {len(bb.identidades())} identidades -> {db_path}")
    return bb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--keys", default=config.KEYS_DIR)
    a = ap.parse_args()
    build(a.db, a.keys, force=a.force)
