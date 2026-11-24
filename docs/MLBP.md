# Cómo funciona MLBP (generación del template en BigBrother)

> **Versión interactiva:** abre [`mlbp_interactivo.html`](mlbp_interactivo.html) directamente en cualquier navegador (no requiere servidor ni conexión a internet) para explorar paso a paso los simuladores de píxeles, el mapa de etiquetas, el conteo por casillas, el selector multiescala `R = 1..6` y la proyección PCA+LDA.

---

## La idea en 10 segundos

Un ordenador no ve «ojos» ni «cejas»: solo ve una tabla de `128 × 128 = 16.384` números de brillo (`0` = negro, `255` = blanco). Comparar píxeles crudos falla en cuanto cambia un poco la iluminación o la cara se desplaza un par de píxeles.

**MLBP** (*Multiscale Local Binary Patterns*, Chan et al., 2007) resuelve esto convirtiendo la foto en un **resumen estadístico de microtexturas por zona y a varias escalas**:

```
Foto completa
  └─► YuNet (detección + 5 landmarks + control de calidad)
        └─► Alineado geométrico 128×128 + gris + equalizeHist    [bigbrother/preprocess.py]
              └─► MLBP: 6 radios × rejilla 8×8 × 59 etiquetas    [bigbrother/mlbp.py]
                    = TEMPLATE (vector float32 de 22.656 decimales)
                          └─► Validación + Faceness + √x + PCA(1000) + LDA(300) + L2
                                = Vector unitario 300D guardado cifrado en BD  [bigbrother/matcher.py]
```

El **template** no es una imagen ni un hash criptográfico: es una sola lista plana de **22.656 porcentajes** que responde a *"en tal casilla de la cara, con tal tamaño de lupa, ¿qué proporción de bordes, esquinas, zonas lisas y ruido apareció?"*.

---

## Paso 1 — De brillo (0..255) a «etiqueta de forma» (0..58)

Para cada píxel central $(x_c, y_c)$ con intensidad $g_c$, se muestrean **$P = 8$ vecinos** repartidos en círculo a distancia $R$ (cada $45^\circ$, usando interpolación bilineal cuando no caen en coordenadas enteras) y se hace una comparación binaria:

$$\text{LBP}_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c)\, 2^p, \qquad s(z) = \begin{cases} 1 & \text{si } z \ge 0 \\ 0 & \text{si } z < 0 \end{cases}$$

- Como solo importa el **signo** ($g_p \ge g_c$), un cambio global de brillo o contraste no altera los 8 bits.
- Al aplicar esto a los `128 × 128` píxeles, el brillo absoluto se descarta y se obtiene un **mapa LBP del mismo tamaño `128 × 128`**, donde cada píxel lleva una etiqueta de textura.

### ¿De dónde salen las 59 etiquetas (`nri_uniform`)?

Con 8 bits hay $2^8 = 256$ combinaciones posibles, pero la mayoría son ruido caótico. Un patrón es **uniforme** si al recorrer sus 8 bits en círculo hay **como máximo 2 transiciones** `0 ↔ 1`:

| Grupo | Ejemplo de 8 bits | Qué forma representa | Bins |
| :--- | :--- | :--- | :---: |
| **0 transiciones ($U=0$)** | `00000000`, `11111111` | Zona plana o pico/valle aislado | **2** |
| **2 transiciones ($U=2$)** | `11110000`, `00111000`… | Borde o esquina limpio: **7** longitudes de arco de `1`s × **8** orientaciones | **56** |
| **Más de 2 ($U>2$)** | `01010110`, `10101010`… | Los **198** patrones irregulares/ruidosos se agrupan en un solo cajón (`#58`) | **1** |
| **Total** | | `points * (points - 1) + 3 = 8 * 7 + 3` (`n_bins()` en `bigbrother/mlbp.py`) | **59** (`0..58`) |

Usamos `nri_uniform` (*non-rotation-invariant uniform*) en lugar de colapsar rotaciones porque **la cara ya llega alineada**: distinguir un borde horizontal (ceja, labio) de uno vertical (tabique nasal) aporta información discriminativa.

---

## Paso 2 — Rejilla espacial `8×8` e histogramas normalizados

Si contáramos las etiquetas sobre toda la cara a la vez, perderíamos la posición de los rasgos (mezclaríamos ojos con mejillas). Por eso:

1. Se divide el mapa LBP de `128 × 128` en una **rejilla `8 × 8` = 64 casillas**, cada una de `16 × 16 = 256` píxeles.
2. Dentro de cada casilla se **cuenta** cuántos píxeles cayeron en la etiqueta `0`, cuántos en la `1`, …, cuántos en la `58`.
3. Se divide cada conteo entre `256` para convertirlo en una distribución de probabilidad (**histograma normalizado** que suma `1,00`).

> **Propiedad usada en seguridad:** como cada celda tiene exactamente 256 píxeles, cada valor de un template legítimo es un múltiplo exacto de `1/256` y cada bloque de 59 valores suma `1`. `BigBrother.validar_template()` comprueba ambas propiedades en los endpoints `*_t` para rechazar vectores sintéticos arbitrarios.

---

## Paso 3 — Multiescala (`R = 1..6`) y concatenación del template

Un solo radio pequeño (`R = 1`, diámetro 3 px) solo capta microtextura fina (poros, piel), mientras que radios mayores (`R = 5, 6`, hasta 13 px de diámetro) captan contornos de ojos, arcos de cejas o el puente nasal.

En [`bigbrother/config.py`](../bigbrother/config.py) se fijan:

```python
FACE_SIZE = 128
LBP_POINTS = 8
LBP_RADII = (1, 2, 3, 4, 5, 6)
LBP_METHOD = "nri_uniform"   # 59 bins
GRID = (8, 8)                # 64 casillas de 16x16 px
PHOTOMETRIC = "equalize"     # cv2.equalizeHist (menor EER en LFW que Tan-Triggs: 15,5% vs 19,7%)
```

Todo el extractor ([`MLBPExtractor.extract`](../bigbrother/mlbp.py)) son 7 líneas:

```python
out = np.empty((len(self.radii), self.n_cells, self.bins), np.float32)  # (6, 64, 59)
for k, r in enumerate(self.radii):
    codes = local_binary_pattern(gray, self.points, r, method=self.method).astype(np.intp)
    idx = self._cell * self.bins + codes
    h = np.bincount(idx.ravel(), minlength=self.n_cells * self.bins)
    h = h.reshape(self.n_cells, self.bins).astype(np.float32)
    out[k] = h / h.sum(axis=1, keepdims=True)
return out.ravel()  # 6 * 64 * 59 = 22.656 floats
```

El bloque `idx = self._cell * 59 + codes` + `np.bincount` es simplemente una forma vectorizada de contar el histograma de 59 bins para las 64 casillas a la vez sin un doble bucle en Python.

Al aplastar `(6, 64, 59)` con `.ravel()` obtenemos el **template MLBP**:

$$6\text{ radios} \times 64\text{ casillas} \times 59\text{ bins} = 384\text{ histogramas} \times 59 = \mathbf{22.656\text{ floats}}$$

---

## Paso 4 — Qué ocurre después del template (Faceness + PCA/LDA)

El template crudo trata por igual todas las casillas (frente, fondo, ojos) y todas las escalas. Antes de comparar o almacenar en la BD cifrada, [`BigBrother._proj`](../bigbrother/system.py) aplica:

1. **Validación estructural** (`validar_template`): forma `22.656`, finito, `>= 0`, suma `1` por casilla y retícula `k/256`.
2. **Filtro `Faceness`** (`bigbrother/matcher.py`): clasificador lineal sobre $\sqrt{x}$ que rechaza ruido o recortes no faciales en los métodos `*_sd` y `*_t`.
3. **Proyección discriminante** (`scripts/train_projection.py`, entrenada en LFW con 4.017 identidades):
   - **Raíz cuadrada ($\sqrt{x}$):** equivale a usar distancia de Hellinger en lugar de euclídea, suavizando bins dominantes.
   - **Centrado y PCA blanqueado (`1.000` dims):** elimina redundancia y normaliza varianzas.
   - **LDA (`300` dims):** maximiza la separación entre personas distintas y comprime variaciones de la misma persona (pose, luz, expresión).
   - **Normalización $L_2$:** deja un vector unitario $\hat{y} \in \mathbb{R}^{300}$ cuyo producto escalar es directamente la **similitud coseno**.

Impacto medido en el split de test de LFW (`models/calibration.json`, 4.300 pares genuinos y 980.849 impostores):

| Comparador | Dimensión | EER (menor es mejor) | TAR @ FAR 1 % | TAR @ FAR 0,1 % |
| :--- | :---: | :---: | :---: | :---: |
| MLBP crudo + distancia $\chi^2$ | 22.656 | 33,60 % | 11,53 % | 4,21 % |
| **MLBP + $\sqrt{x}$ + PCA(1000) + LDA(300) + coseno** | **300** | **15,49 %** | **54,12 %** | **34,77 %** |

---

## Preguntas frecuentes

### 1. ¿Por qué hay 59 valores (`0..58`) si una imagen en grises tiene 256 niveles de brillo?
Porque los valores `0..58` del mapa LBP **no son brillos**, sino **59 categorías de forma/textura**. Los 256 brillos originales se usan solo para decidir si cada uno de los 8 vecinos es `≥` o `<` que el píxel central; a partir de ahí se guarda únicamente el índice del patrón (`2` con 0 transiciones + `56` bordes/esquinas orientados + `1` cajón para los 198 patrones no uniformes).

### 2. Cuando sube el radio `R`, ¿por qué no aumentan las combinaciones posibles?
Porque el número de puntos muestreados se mantiene **fijo en `P = 8`** para todos los radios (solo se abre el círculo, muestreando 8 puntos equidistantes cada $45^\circ$ con interpolación bilineal). Si subiéramos `P` con el radio, el número de bins crecería demasiado para casillas de solo `256` píxeles:

| Vecinos `P` | Combinaciones ($2^P$) | Bins `nri_uniform` | Media de píxeles por bin (en 256 px) |
| :---: | :---: | :---: | :---: |
| **8** (usado) | 256 | **59** | ~4,3 px/bin (histograma denso) |
| 16 | 65.536 | 243 | ~1,0 px/bin (casi vacío) |
| 24 | 16,7 M | 555 | ~0,46 px/bin (más cajas que píxeles) |

### 3. Si tuviéramos una imagen más grande, ¿crecería el template? ¿Y qué pasa con los radios?
- **Foto de entrada más grande (p. ej., 4K):** no cambia nada; el preprocesado siempre alinea y reescala la cara a `128×128`.
- **Cambiar `FACE_SIZE` a `256×256` en diseño:**
  - El tamaño del template depende de `GRID`, **no** de `FACE_SIZE`. Con `GRID = (8, 8)` seguiría midiendo `22.656` (cada casilla tendría `32×32 = 1.024` píxeles en vez de `256`). Solo crecería si además ampliaras la rejilla (p. ej., `16×16` → `90.624` floats).
  - **Sobre los radios `R`:** el círculo de radio `R` se centra en **cada píxel** de la imagen (no una sola vez en el centro de la casilla intentando alcanzar sus esquinas). Sin embargo, en `256×256` cada ojo o ceja ocupa el doble de píxeles que en `128×128`, así que dejar `R = 1..6` se quedaría corto en escala física y habría que escalar también los radios (p. ej., `2, 4, 6, 8, 10, 12`) para seguir captando los mismos rasgos faciales.
