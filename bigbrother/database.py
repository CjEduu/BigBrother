import csv
from pathlib import Path

import numpy as np


class Database:
    """BD en CSV: columnas id,template (floats separados por espacio). Varias filas por id."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []  # [(id, np.ndarray)]
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["id"], np.array(r["template"].split(), dtype=np.float32)))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "template"])
            for id_, t in self.rows:
                w.writerow([id_, " ".join(f"{x:.8g}" for x in t)])
        # TODO(seguridad): integridad (HMAC) y/o cifrado de templates

    def add(self, id_, template):
        self.rows.append((id_, np.asarray(template, dtype=np.float32)))
        self._save()

    def get(self, id_):
        return [t for i, t in self.rows if i == id_]

    def all(self):
        return self.rows
