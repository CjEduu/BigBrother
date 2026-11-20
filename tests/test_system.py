import csv
import shutil

import numpy as np
import pytest

from bigbrother import (BigBrother, EnrollmentRejected, IntegrityError, InvalidInput,
                        NoFaceDetected)
from bigbrother.dataset import FOTOS_DIR, id_from_filename
from bigbrother.imageio import load_image
from bigbrother.mlbp import MLBPExtractor

DS = FOTOS_DIR / "dataset"
KEANU = DS / "keanu_reeves_2021.png"
SALMA = DS / "salma_hayek_2012.jpg"
IDRIS = DS / "idris_elba_2022.jpg"


def _noise(seed, size=128):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.fixture
def bb(tmp_path):
    return BigBrother(db_path=tmp_path / "data" / "db.csv", keys_dir=tmp_path / "keys")


@pytest.fixture
def bb3(bb):
    """Sistema con 3 identidades inscritas."""
    for p in (KEANU, SALMA, IDRIS):
        bb.inscribir(id_from_filename(p.name), load_image(p))
    return bb


# ---------------------------------------------------------------- extractor / templates
def test_template_is_valid_lattice_histogram():
    ex = MLBPExtractor()
    t = ex.extract(_noise(0)[..., 0])
    assert t.shape == (ex.dim,) and t.dtype == np.float32
    H = ex.histograms(t)
    assert np.allclose(H.sum(1), 1) and np.allclose(H * 256, np.round(H * 256))


def test_id_from_filename():
    assert id_from_filename("aitana_etxebarria_(twin_melody)_2024.jpg") == "aitana_etxebarria"
    assert id_from_filename("james_phelps_(gemelo_weasley)_2011.jpg") == "james_phelps"
    assert id_from_filename("elisabeth_II_1945.jpg") == "elisabeth_ii"


def test_avif_with_jpg_extension_loads():
    img = load_image(DS / "aitana_etxebarria_(twin_melody)_2024.jpg")
    assert img.ndim == 3 and img.dtype == np.uint8


# ---------------------------------------------------------------- funcionalidad
def test_pipeline_basico(bb3):
    assert bb3.verificar("keanu_reeves", load_image(KEANU)) is True
    assert bb3.verificar("keanu_reeves", load_image(SALMA)) is False
    assert bb3.identificar(load_image(SALMA)) == "salma_hayek"
    assert bb3.verificar("no_existe", load_image(KEANU)) is False


def test_variantes_sd_y_t_coinciden(bb3):
    img = load_image(IDRIS)
    cara = bb3.recortar(img)
    t = bb3.extraer_sd(cara)
    assert np.array_equal(t, bb3.extraer(img))
    assert bb3.verificar_sd("idris_elba", cara) and bb3.verificar_t("idris_elba", t)
    assert bb3.identificar_sd(cara) == bb3.identificar_t(t) == "idris_elba"


def test_api_no_expone_puntuaciones(bb3):
    assert isinstance(bb3.verificar("keanu_reeves", load_image(KEANU)), bool)
    assert isinstance(bb3.identificar(load_image(KEANU)), str)


def test_sin_cara(bb):
    with pytest.raises(NoFaceDetected):
        bb.verificar("keanu_reeves", _noise(1, 400))


def test_basura_por_sd_rechazada(bb3):
    """Regresion: antes del filtro faceness, ruido aleatorio por _sd se identificaba como
    salma_hayek."""
    basura = [_noise(s) for s in range(10)] + [np.full((128, 128, 3), v, np.uint8) for v in (0, 127, 255)]
    for im in basura:
        with pytest.raises(NoFaceDetected):
            bb3.identificar_sd(im)
        with pytest.raises(NoFaceDetected):
            bb3.verificar_t("salma_hayek", bb3.extractor.extract(im[..., 0]))


# ---------------------------------------------------------------- politica de inscripcion
def test_no_se_puede_secuestrar_un_id(bb3):
    with pytest.raises(EnrollmentRejected):
        bb3.inscribir("keanu_reeves", load_image(SALMA))
    with pytest.raises(EnrollmentRejected):
        bb3.inscribir("keanu_reeves", load_image(SALMA), actualizar=True)
    assert bb3.verificar("keanu_reeves", load_image(SALMA)) is False


def test_deduplicacion(bb3):
    with pytest.raises(EnrollmentRejected):
        bb3.inscribir("otro_keanu", load_image(KEANU))


@pytest.mark.parametrize("bad", ["=cmd|' /C calc'!A0", "a,b", "Vinicius", "", "x\ny", "../etc", "-1+1"])
def test_ids_invalidos(bb, bad):
    with pytest.raises(InvalidInput):
        bb.inscribir_sd(bad, _noise(0))


def test_templates_invalidos(bb3):
    t = bb3.extraer(load_image(KEANU))
    bad = [t[:-1], np.full_like(t, np.nan), -t, t * 2, t + 1e-3,
           np.full_like(t, 1 / bb3.extractor.bins),            # uniforme (no reticula)
           (t + bb3.extraer(load_image(SALMA))) / 2 + 1e-5]    # "media" de dos templates
    for b in bad:
        with pytest.raises(InvalidInput):
            bb3.verificar_t("keanu_reeves", b)


# ---------------------------------------------------------------- integridad de la BD
def _rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _write(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def test_bd_cifrada(bb3):
    txt = bb3.db.path.read_text()
    assert "keanu_reeves" in txt  # el id es visible...
    rows = _rows(bb3.db.path)[1:]
    assert all(len(r) == 3 for r in rows)  # ...pero el template va cifrado


def test_persistencia(bb3, tmp_path):
    bb2 = BigBrother(db_path=bb3.db.path, keys_dir=tmp_path / "keys")
    assert bb2.identidades() == ["idris_elba", "keanu_reeves", "salma_hayek"]
    assert bb2.verificar("salma_hayek", load_image(SALMA))


@pytest.mark.parametrize("ataque", ["editar", "swap", "borrar_fila", "anadir_fila", "borrar_bd",
                                    "rollback", "sin_huella", "otra_clave"])
def test_manipulacion_bd_detectada(bb3, tmp_path, ataque):
    db, keys = bb3.db.path, tmp_path / "keys"
    rows = _rows(db)
    if ataque == "editar":
        r = rows[1]
        r[2] = r[2][:-6] + ("AAAAAA" if not r[2].endswith("AAAAAA") else "BBBBBB")
        _write(db, rows)
    elif ataque == "swap":  # el template de salma pasa a llamarse keanu
        i = next(k for k, r in enumerate(rows) if r[0] == "keanu_reeves")
        j = next(k for k, r in enumerate(rows) if r[0] == "salma_hayek")
        rows[i][1:], rows[j][1:] = rows[j][1:], rows[i][1:]
        _write(db, rows)
    elif ataque == "borrar_fila":
        _write(db, rows[:-1])
    elif ataque == "anadir_fila":
        _write(db, rows + [["vinicius_junior", rows[1][1], rows[1][2]]])
    elif ataque == "borrar_bd":
        db.unlink()
    elif ataque == "rollback":
        old = db.read_bytes()
        bb3.inscribir("vinicius_junior", load_image(FOTOS_DIR / "vinicius_junior" / "vini_06.jpg"))
        db.write_bytes(old)
    elif ataque == "sin_huella":
        (keys / "db.state").unlink()
    elif ataque == "otra_clave":
        shutil.rmtree(keys)
    with pytest.raises(IntegrityError):
        b = BigBrother(db_path=db, keys_dir=keys)
        if ataque == "otra_clave":  # clave nueva generada -> no puede leer la BD
            b.identidades()


# ---------------------------------------------------------------- integridad de modelos/umbrales
@pytest.mark.parametrize("ataque", ["bajar_umbral", "borrar_filtro_caras", "cambiar_proyeccion"])
def test_manipulacion_modelos_detectada(tmp_path, monkeypatch, ataque):
    from bigbrother import config
    copies = []
    for f in config.MODEL_FILES:
        dst = tmp_path / "models" / f.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(f, dst)
        copies.append(dst)
    monkeypatch.setattr(config, "MODEL_FILES", tuple(copies))
    db, keys = tmp_path / "db.csv", tmp_path / "keys"
    BigBrother(db_path=db, keys_dir=keys).inscribir("keanu_reeves", load_image(KEANU))
    cal, face, proj = (tmp_path / "models" / n for n in
                       ("calibration.json", "faceness.npz", "mlbp_projection.npz"))
    if ataque == "bajar_umbral":
        cal.write_text(cal.read_text().replace('"threshold": 0.', '"threshold": -0.'))
    elif ataque == "borrar_filtro_caras":
        face.unlink()
    else:
        proj.write_bytes(proj.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(IntegrityError):
        BigBrother(db_path=db, keys_dir=keys)


def test_umbral_no_se_cambia_por_entorno(monkeypatch):
    import importlib

    from bigbrother import config
    monkeypatch.setenv("BIGBROTHER_THRESHOLD", "-1")
    monkeypatch.setenv("BIGBROTHER_IDENT_THRESHOLD", "-1")
    assert importlib.reload(config).THRESHOLD > 0.1 and config.IDENT_THRESHOLD > 0.1


def test_sin_clave_no_genera_otra_si_hay_bd(bb3, tmp_path):
    (tmp_path / "keys" / "master.key").unlink()
    with pytest.raises(IntegrityError):
        BigBrother(db_path=bb3.db.path, keys_dir=tmp_path / "keys")
    assert not (tmp_path / "keys" / "master.key").exists()


def test_actualizar_con_muestra_que_verifica(bb3):
    """Anadir otra muestra solo es posible con una foto que ya verifica (no es un bypass,
    pero si un atacante consigue verificar UNA vez, puede persistir: ver INFORME V-actualizar)."""
    n = len(bb3.db.get("keanu_reeves"))
    bb3.inscribir("keanu_reeves", load_image(KEANU)[:, ::-1].copy(), actualizar=True)
    assert len(bb3.db.get("keanu_reeves")) == n + 1
