# Plan de trabajo - BigBrother (MLBP)

Pipeline: foto -> YuNet (detector) -> alineado 5 landmarks -> MLBP -> template -> chi2 -> decision

## Decisiones
- Detector: YuNet (OpenCV, ONNX). Da bbox + 5 landmarks para alinear, solo depende de OpenCV.
- Extractor: LBP uniforme (P=8), radios 1,2,3, rejilla 8x8, histogramas normalizados -> 1920 dim.
- Comparador: chi-cuadrado, con pesos por region (a calibrar).
- Persona 51: vinicius_junior.

## Issues
1. Dataset loader + inscripcion masiva de las 51 personas
2. Calibrar umbral (FAR/FRR/EER) y pesos por region
3. Evaluacion propia: ROC/DET, Rank-1
4. Seguridad: bypass del detector (_sd/_t), lectura de BD, escritura de BD
5. Mitigaciones: HMAC/cifrado de templates, control de inscripcion
6. Ataques de presentacion (foto en pantalla/impresa)
7. Docker + README de uso para el grupo rival
8. Evaluacion del rival
9. Presentacion
