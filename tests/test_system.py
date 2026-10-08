import numpy as np

from bigbrother import BigBrother
from bigbrother.mlbp import MLBPExtractor


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, (128, 128), dtype=np.uint8)


def test_template_dim():
    ex = MLBPExtractor()
    t = ex.extract(_img(0))
    assert t.shape == (ex.dim,) and t.dtype == np.float32


def test_identificar_sd_roundtrip(tmp_path):
    bb = BigBrother(db_path=tmp_path / "db.csv", threshold=1e-3)
    bb.inscribir_sd("a", _img(1))
    bb.inscribir_sd("b", _img(2))
    assert bb.identificar_sd(_img(1))[0] == "a"
    assert bb.verificar_sd("b", _img(2))[0] is True
    assert bb.verificar_sd("b", _img(1))[0] is False


def test_db_persistence(tmp_path):
    p = tmp_path / "db.csv"
    BigBrother(db_path=p, threshold=1e-3).inscribir_sd("a", _img(1))
    assert BigBrother(db_path=p, threshold=1e-3).identificar_sd(_img(1))[0] == "a"
