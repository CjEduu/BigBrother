import csv
import io
import os
from pathlib import Path

import numpy as np

from .errors import IntegrityError
from .security import validate_id

HEADER = ["id", "nonce", "template_cifrado"]


class Database:
    """BD en CSV: id,nonce,template_cifrado. Varias filas (templates) por id.

    Los templates se guardan proyectados (espacio discriminante) y cifrados (AES-GCM).
    Cualquier modificacion externa del CSV hace que el sistema se niegue a arrancar
    (fail-closed): ver security.KeyStore.
    """

    def __init__(self, path, keystore):
        self.path = Path(path)
        self.ks = keystore
        self.rows = []  # [(id, np.ndarray)]
        self._load()

    def _load(self):
        raw = self.path.read_bytes() if self.path.exists() else None
        self.ks.check_db(raw)
        if raw is None:
            return
        reader = csv.reader(io.StringIO(raw.decode("utf-8")))
        if next(reader, None) != HEADER:
            raise IntegrityError("Cabecera de la BD invalida")
        for r in reader:
            if len(r) != 3:
                raise IntegrityError("Fila de la BD malformada")
            id_ = validate_id(r[0])
            self.rows.append((id_, self.ks.decrypt(id_, r[1], r[2])))

    def _save(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(HEADER)
        for id_, t in self.rows:
            w.writerow([id_, *self.ks.encrypt(id_, t)])
        data = buf.getvalue().encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self.path)
        self.ks.write_state(data, len(self.rows))

    def add(self, id_, template):
        self.add_many([(id_, template)])

    def add_many(self, items):
        for id_, t in items:
            self.rows.append((validate_id(id_), np.asarray(t, dtype=np.float32)))
        self._save()

    def get(self, id_):
        return [t for i, t in self.rows if i == id_]

    def ids(self):
        return sorted({i for i, _ in self.rows})

    def all(self):
        return self.rows

    def __len__(self):
        return len(self.rows)
