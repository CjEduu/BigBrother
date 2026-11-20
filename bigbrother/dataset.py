"""Utilidades para las fotos de inscripcion (data/fotos)."""
import re
from pathlib import Path

from . import config

FOTOS_DIR = config.ROOT / "data" / "fotos"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}


def id_from_filename(name):
    """'aitana_etxebarria_(twin_melody)_2024.jpg' -> 'aitana_etxebarria'."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"_?\(.*?\)", "", stem)
    stem = re.sub(r"_\d{4}$", "", stem)
    return stem.strip("_")


def enrollment_photos(root=FOTOS_DIR):
    """[(id, path)] de las 50 fotos del docente (data/fotos/dataset/*) y de las
    carpetas por persona (data/fotos/<id>/*), p.ej. vinicius_junior."""
    root = Path(root)
    out = []
    for p in sorted((root / "dataset").iterdir()):
        if p.suffix.lower() in IMG_EXT:
            out.append((id_from_filename(p.name), p))
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name != "dataset":
            out += [(d.name, p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    return out


def dataset_ids(root=FOTOS_DIR):
    return sorted({i for i, _ in enrollment_photos(root)})
