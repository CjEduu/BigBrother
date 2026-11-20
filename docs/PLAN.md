# Plan de trabajo - BigBrother (MLBP)

Pipeline: foto -> YuNet (detector + calidad) -> alineado 5 landmarks -> ecualizacion -> MLBP (template)
-> filtro "es una cara" -> sqrt/PCA-blanqueado/LDA (LFW) -> coseno -> umbral

## Estado
- [x] Dataset loader + inscripcion de las 50 identidades (49 docente + vinicius_junior) -> `scripts/build_db.py`
- [x] Entrenamiento proyeccion + calibracion de umbrales (FAR 0.1% verif., FPIR 1% ident.) -> `scripts/train_projection.py`, `calibrate_identification.py`
- [x] Evaluacion propia: ROC, LOO Vinicius, open-set, gemelos LFW, robustez, PAD simulado, morphing -> `scripts/evaluate.py`
- [x] Seguridad: bypass del detector (_sd/_t), lectura de BD (cifrado), escritura de BD (integridad + fail-closed)
- [x] Control de inscripcion (no sobrescribir ids, actualizar solo si verifica, deduplicacion)
- [x] Docker + README + CI (GitHub Actions construye la imagen y pasa los tests)
- [x] Informe de vulnerabilidades/fortalezas -> `docs/INFORME.md`
- [ ] Evaluacion del sistema rival (viernes 27)
- [ ] Presentacion

## Ideas pendientes (no implementadas, ver INFORME)
- Deteccion de vida / PAD (V2), limite de intentos (V8), clave en KMS (V1), mas fotos por inscrito (V6).
