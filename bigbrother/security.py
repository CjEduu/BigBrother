"""Proteccion de la BD de templates.

- Cada template se cifra con AES-256-GCM. El id se usa como "associated data", de modo que
  copiar el cifrado de una fila a otra identidad (row swapping) se detecta.
- La huella (SHA-256) del CSV completo + numero de filas se guarda, autenticada con HMAC, en
  la zona protegida (KEYS_DIR). Detecta filas anadidas/borradas, reordenadas, BD restaurada
  a una version anterior (rollback) o BD borrada.
- La clave maestra vive en KEYS_DIR/master.key (o en la variable BIGBROTHER_KEY, base64).
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import config
from .errors import IntegrityError, InvalidInput

_ID_RE = re.compile(config.ID_REGEX)


def validate_id(id_):
    """Ids en minusculas, sin comas/saltos/comillas ni prefijos de formula (=,+,-,@)."""
    if not isinstance(id_, str) or not _ID_RE.match(id_):
        raise InvalidInput(f"Id no valido: {id_!r} (debe cumplir {config.ID_REGEX})")
    return id_


def _hkdf(master, info):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(master)


class KeyStore:
    def __init__(self, keys_dir=config.KEYS_DIR, create=True):
        self.keys_dir = Path(keys_dir)
        self.key_path = self.keys_dir / "master.key"
        self.state_path = self.keys_dir / "db.state"
        master = self._load_master(create)
        self._aead = AESGCM(_hkdf(master, b"bigbrother/template-enc/v1"))
        self._mac_key = _hkdf(master, b"bigbrother/db-state-mac/v1")

    def _load_master(self, create):
        env = os.environ.get("BIGBROTHER_KEY")
        if env:
            master = base64.b64decode(env)
        elif self.key_path.exists():
            master = base64.b64decode(self.key_path.read_bytes().strip())
        elif create:
            self.keys_dir.mkdir(parents=True, exist_ok=True)
            master = secrets.token_bytes(32)
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64encode(master))
        else:
            raise IntegrityError(f"No hay clave maestra en {self.key_path}")
        if len(master) != 32:
            raise IntegrityError("La clave maestra debe tener 32 bytes")
        return master

    # --- templates -----------------------------------------------------------
    @staticmethod
    def _aad(id_):
        return b"bigbrother/v1|" + id_.encode()

    def encrypt(self, id_, vec):
        nonce = secrets.token_bytes(12)
        ct = self._aead.encrypt(nonce, np.asarray(vec, "<f4").tobytes(), self._aad(id_))
        return base64.b64encode(nonce).decode(), base64.b64encode(ct).decode()

    def decrypt(self, id_, nonce_b64, ct_b64):
        try:
            pt = self._aead.decrypt(base64.b64decode(nonce_b64), base64.b64decode(ct_b64),
                                    self._aad(id_))
        except (InvalidTag, ValueError) as e:
            raise IntegrityError(f"Template de '{id_}' manipulado o clave incorrecta") from e
        return np.frombuffer(pt, "<f4").astype(np.float32)

    # --- huella de la BD ----------------------------------------------------
    def _mac(self, digest, rows):
        msg = f"{digest}|{rows}".encode()
        return hmac.new(self._mac_key, msg, hashlib.sha256).hexdigest()

    def write_state(self, db_bytes, rows):
        digest = hashlib.sha256(db_bytes).hexdigest()
        state = {"sha256": digest, "rows": rows, "mac": self._mac(digest, rows)}
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.state_path)

    def read_state(self):
        if not self.state_path.exists():
            return None
        try:
            s = json.loads(self.state_path.read_text())
            ok = hmac.compare_digest(s["mac"], self._mac(s["sha256"], int(s["rows"])))
        except (ValueError, KeyError, TypeError) as e:
            raise IntegrityError("Fichero de estado de la BD corrupto") from e
        if not ok:
            raise IntegrityError("Fichero de estado de la BD manipulado")
        return s

    def check_db(self, db_bytes):
        """Compara el CSV con la huella guardada. db_bytes=None si el CSV no existe."""
        s = self.read_state()
        if s is None:
            if db_bytes is not None:
                raise IntegrityError("La BD existe pero no hay huella: creada/copiada fuera del sistema")
            return
        if db_bytes is None:
            if s["rows"] > 0:
                raise IntegrityError("La BD ha sido borrada")
            return
        if not hmac.compare_digest(hashlib.sha256(db_bytes).hexdigest(), s["sha256"]):
            raise IntegrityError("La BD (CSV) ha sido modificada fuera del sistema")
