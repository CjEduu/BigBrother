# Ataques al reconocimiento facial: la literatura y su mapeo en BigBrother

Grupo EVE · Sistemas Biométricos

Este documento es el complemento **teórico** de [`INFORME.md`](INFORME.md): allí se
documentan los ataques que **probamos** contra el sistema (V1–V19, con cifras de
`scripts/evaluate.py`); aquí se recogen los ataques que la **literatura científica**
describe para este tipo de sistemas, se explica **cómo funcionan** según sus autores
originales y se mapea cada uno sobre los componentes concretos de BigBrother
(sensor → `_sd` → `_t` → BD → decisión). El objetivo es doble: justificar por qué
defendemos lo que defendemos (con la referencia que lo motiva) y descubrir huecos
que la evaluación empírica por sí sola no revela (sección 5).

---

## 1. Marcos de referencia de la literatura

### 1.1 Los ocho puntos de ataque de Ratha

El marco clásico para razonar sobre ataques a un sistema biométrico es el de
**Ratha, Connell y Bolle (2001)** [*Enhancing security and privacy in biometrics-based
authentication systems*, IBM Systems Journal 40(3):614–634], que enumera ocho puntos
vulnerables de la arquitectura genérica captura → extracción → comparador → BD → decisión:

| # | Punto de ataque (Ratha) | Equivalente en BigBrother |
|---|---|---|
| 1 | Biometric falsa en el sensor (dedo de goma, careta, foto) | Foto/pantalla/máscara presentada a `verificar` (pasa por YuNet) |
| 2 | Repetición ("replay") de datos digitales capturados | Los métodos `_sd` y `_t` aceptan **literalmente** datos intermedios ya procesados: un recorte o un template capturado se puede reenviar sin vida ni frescura |
| 3 | Sustitución del extractor de rasgos por un caballo de Troya que produce rasgos elegidos por el atacante | Editar `mlbp.py` / `matcher.py` (ver hueco H1: el código NO está cubierto por la huella de integridad) |
| 4 | Manipulación del vector de rasgos en el canal extractor→comparador | El canal extractor→comparador es **la API pública** `verificar_t`: el atacante entrega el vector que quiera (lo limita `validar_template`, sección 4.7) |
| 5 | Sustitución del comparador (Troyano que devuelve la puntuación elegida) | Editar `system.py` / `matcher.py` (H1) |
| 6 | Manipulación de la BD de templates | Cualquier escritura en `data/db.csv`: fila copiada, borrada, reordenada, rollback (defendido por `security.py`) |
| 7 | Interposición en el canal BD↔comparador | No existe como canal: todo ocurre en un proceso; se reduce a H1 (control del proceso) |
| 8 | Anulación de la decisión final después de la comparación | Editar el `return` de `verificar` (H1) o capturar el contenedor como root (V18) |

Dos observaciones estructurales importantes para el mapeo:

- **BigBrother expone los puntos 2 y 4 como API legítima** (`_sd`, `_t`). Es una
  decisión de diseño del enunciado (los 9 métodos), no un fallo; pero convierte esos
  canales, que en Ratha son superficies internas, en superficies **públicas**, y obliga
  a las contramedidas que damos (filtro de cara + validación de retícula).
- Al ser un proceso único sin red, los puntos 3–8 colapsan en una sola condición:
  *quien controla el código o el entorno, controla todo* (H1, V17, V18). La integridad
  que sí verificamos (BD, modelos, umbrales) cubre el punto 6 y una parte de los
  puntos 3/5 (los **modelos**, pero no el **código**; ver sección 5.1).

#El pipeline completo, con los tres canales de la API y el punto exacto donde golpea
cada ataque medido, está ilustrado en
[`ATAQUES_EMPIRICOS.md`](ATAQUES_EMPIRICOS.md) → [`resultados/pipeline_ataques.png`](resultados/pipeline_ataques.png).

## 1.2 Normas de referencia

- **ISO/IEC 30107** (*Presentation attack detection*): define PA (ataque de
  presentación), PAI (instrumento), y las métricas APCER/BPCER con las que se evalúa
  un detector de ataques. BigBrother **no tiene ningún módulo PAD**: este es el marco
  en el que se sitúa la vulnerabilidad V2.
- **ISO/IEC 24745** (*Biometric information protection*): exige a los templates
  protegidos **irreversibilidad, revocabilidad y desvinculabilidad**. La sección 4.6
  usa estas tres propiedades como vara de medir la BD de BigBrother.
- **NIST SP 800-63B** (autenticación digital): para uso biométrico sin supervisión
  exige límites de intentos y puntos de operación conservadores; conecta con V8
  (intentos ilimitados).

---

## 2. Presentación: foto, pantalla, máscara, deepfake (punto 1 de Ratha)

**Literatura.** Es la familia más estudiada del reconocimiento facial. Las revisiones
de Ramachandra y Busch (ACM Computing Surveys 50(1), 2017) y de Matineh et al. (ACM
Computing Surveys, 2024) ordenan los instrumentos de ataque (PAI) por sofisticación:

1. **Foto impresa** (print attack).
2. **Replay en pantalla**: vídeo o foto mostrado en un teléfono/monitor (introduce
   moiré, desenfoque, gamma y a veces perspectiva).
3. **Máscara 3D** (silicona, resina).
4. **Deepfakes / síntesis**: rostro generado o intercambiado presentado como vídeo.

El estado del arte en PAD (liveness: parpadeo, textura de superficie, profundidad,
reflectancia) asume que **sin PAD cualquier FRS comercial acepta fotos impresas y
pantallas**; la pregunta en la literatura no es *si* funciona, sino *a qué coste se
detecta* (APCER vs BPCER).

**Mapeo en BigBrother.** Punto 1 de Ratha; entra por `verificar(id, foto)` y solo
encuentra YuNet + control de calidad (tamaño, yaw, ambigüedad), que **no** distinguen
bona fide de presentación. La evaluación lo confirma (INFORME V2): simulaciones de
foto en pantalla e impresa aceptadas 5/5; cualquier foto pública de la persona basta
(LFW: los 5 templates de Vinícius se atacan con fotos *held-out* del mismo). El
filtro "¿es una cara?" (`faceness.npz`) no ayuda ni debe ayudar: una foto impresa de
una cara **es** una cara.

**Lectura de la literatura aplicable:** la única mitigación conocida es un módulo PAD
(ISO/IEC 30107), deliberadamente fuera del alcance de la práctica. La recomendación
operativa de la literatura (NIST SP 800-63B) para sistemas sin PAD es limitar
intentos y supervisar; BigBrother no limita intentos (V8).

---

## 3. Morphing (punto 1 aplicado a la inscripción)

**Literatura.** Ferrara, Franco y Maltoni (*The magic passport*, IJCB 2014) introducen
el ataque: se mezclan dos caras (actor malicioso + cómplice) en una sola imagen que
**verifica contra ambas**; se inscribe como foto de documento y ambas personas pasan
el control. Scherhag et al. (BIOSIG 2017) estandarizan cómo medir la vulnerabilidad
de un FRS al morphing (MMPMR: *morphing match probability multiple presented
morphs*), y las revisiones de Venkatesh et al. (2020) y Ferrara et al. (*Morphing
attack potential*, 2022 — marco de potencial de ataque estilo Common Criteria)
recorren generación, detección (MAD) y evaluación. Ramachandra et al. (2023) y
O'Haire et al. (IJCB 2022) muestran que los morphing funcionan **mejor** con
parecidos y, en el límite, con gemelos idénticos.

El mecanismo es el mismo de siempre: el morphing no engaña al detector sino al
**comparador**; el morphed queda dentro de la nube de aceptación de ambos padres
porque la extracción de rasgos promedia razonablemente la mezcla.

**Mapeo en BigBrother.** Dos vías:

- **Vía verificación (medida, INFORME V3):** foto morphed presentada a
  `verificar("vinicius_junior", ...)` — aceptada 24/25 con cómplice parecido, 7/25
  con cómplice aleatorio. Solo mide el lado de la víctima (el morph como foto
  degradada de Vinícius), no que el cómplice también verifique.
- **Vía inscripción (hueco H2, medida después en A3 de
  [`ATAQUES_EMPIRICOS.md`](ATAQUES_EMPIRICOS.md)):** el morphing canónico de la literatura
  ataca a la **inscripción**, y en BigBrother esa vía es más peligrosa por el
  encadenamiento V16: si un morph de (atacante, víctima) se inscribe como id nuevo,
  ambos verifican contra él; y si el atacante ya verificó **una vez** por cualquier
  vía (morph, parecido, foto robada), `inscribir(..., actualizar=True)` le permite
  añadir su propia cara al id de la víctima (*template poisoning*), que es la
  versión persistente del morphing: a partir de ese momento él es genuino.
  Mitigación parcial que sí tenemos: la **deduplicación** rechaza inscribir una cara
  que ya coincida con otra identidad (un morph 50/50 de un inscrito + uno nuevo
  probablemente puntuaría ≥ `dedup_threshold` contra el inscrito y se rechazaría), y
  `actualizar=True` exige que la muestra nueva verifique antes de añadirla. Ambas
  políticas son exactamente los controles de *identity proofing* que la literatura de
  morphing recomienda para el lado de la inscripción.

**Nota:** al igual que en el punto 2, sin MAD (detección de morphing) la única
defensa real es que la inscripción sea un acto autenticado/supervisado, no una API
abierta (V15).

---

## 4. Ataques indirectos al comparador: hill-climbing, fuerza bruta, adversariales

Esta familia no toca el sensor: trabaja sobre el **canal de datos del comparador**
(punto 4 de Ratha — en BigBrother, la API) usando su respuesta como oráculo.

### 4.1 Hill-climbing con feedback de puntuación

**Literatura.** Adler (*Sample images can be independently restored from face
recognition templates*, CCECE 2003) demostró el mecanismo canónico sobre un FRS de
eigenfaces: parte de una cara arbitraria y la modifica iterativamente (en su caso,
proyectando perturbaciones sobre eigenfaces) aceptando cada cambio que sube la
puntuación; en unas 3000 consultas alcanza una confianza del 99,8 % — y de paso
reconstruye una imagen aproximada de la víctima (esto también es el primer ataque de
inversión de templates, sección 4.6). Galbally, McCool, Fierrez y Marcel (*On the
vulnerability of face verification systems to hill-climbing attacks*, Pattern
Recognition 43(3), 2010) refinan el ataque por cuadrantes y demuestran que funciona
aunque el sistema solo devuelva **puntuaciones cuantizadas** (la contramedida que
recomienda BioAPI). Gomez-Barrero et al. (ICB 2012) logran lo mismo con el algoritmo
del simplex ascendente; Martinez-Diaz et al. (*Hill-climbing and brute-force attacks
on biometric systems*, ICCST 2006) añaden la fuerza bruta como caso límite y muestran
que el número de consultas es lo que separa ambos. La leña común: **cualquier sistema
que filtre su puntuación es hill-climbable**; la cuantización solo encarece el ataque.

**Mapeo en BigBrother.** Es la V9 del INFORME. La decisión de diseño que lo mitiga es
deliberada y es exactamente la contramedida de la literatura llevada al extremo: la
API pública **no devuelve puntuaciones, solo `bool`/id/None** (test
`test_api_no_expone_puntuaciones`). En caja blanca (código accesible, `bb._score_id`)
la evaluación mide ~900–1.600 consultas hasta el umbral (`resultados/hill_climbing.png`).

### 4.2 Hill-climbing con feedback binario (decisión sí/no)

**Literatura.** Contra lo que la cuantización sugiere, el feedback de 1 bit también
sirve. Dong et al. (*Efficient decision-based black-box adversarial attacks on face
recognition*, CVPR 2019) optimizan perturbaciones para FRS profundos usando **solo la
decisión** de aceptar/rechazar (sin puntuación ni gradiente), con presupuestos de
miles de consultas. La condición para que el ataque sea práctico es un **oráculo con
intentos ilimitados**, exactamente lo que NIST SP 800-63B prohíbe para biometría sin
supervisión.

**Mapeo en BigBrother.** Hueco H3 (medido después en A2 de
[`ATAQUES_EMPIRICOS.md`](ATAQUES_EMPIRICOS.md): converge con ~4.000 consultas). El análisis
previo ya decía que era **plausible**: el objetivo es lineal (coseno sobre proyección LDA), la superficie es
suave, y no hay límite de intentos (V8). Cada consulta por `verificar_t` está además
restringida a la retícula de histogramas válidos (la ventaja: el espacio de búsqueda
es discreto y más pequeño; la desventaja: los pasos son también más toscos). La
mitigación real no es el feedback binario sino **limitar la tasa de consultas**, que
no existe. En caja negra con V8 abierto, un hill-climb binario lento pero viable está
dentro de lo que la literatura predice.

### 4.3 Ejemplos adversariales digitales y físicos (dodging e impersonación)

**Literatura.** Sharif et al. (*Accessorize to a crime*, CCS 2016) marcan el
estándar de los ataques **físicamente realizables**: gafas impresas con un patrón
optimizado que logran tanto *impersonation* (verificar como otro) como *dodging* (no
ser reconocido) contra FRS profundos. Dong et al. (CVPR 2019, ya citado) cubren la
variante de caja negra; Deb et al. (*AdvFaces*, 2019) optimizan perturbaciones con
calidad perceptual. Para la **detección**, Yang et al. (*Design and interpretation of
universal adversarial patches in face detection*, 2019) muestran parches que ciegan
detectores (dodging por el punto 1). Singh et al. (2023) atacan detector y extractor
**a la vez**.

**Mapeo en BigBrother.** Puntos 1 y 4 de Ratha a la vez. Matiz importante: casi toda
esta literatura ataca **redes profundas con gradientes diferenciables**; nuestro
comparador es MLBP+LDA y el flujo previo contiene pasos no diferenciables y
destructivos de señal (ecualización, cuantización LBP a 59 bins, rejilla 8×8). Un
parche adversarial debe **sobrevivir a todos ellos**, lo que encarece el ataque
digital y el físico (la impresión añade ruido). PERO, a cambio, nuestro pipeline es
público y simple: un atacante de caja blanca lo re-implementa y tiene gradiente
analítico exacto hasta la proyección (el LBP discretiza, pero el gradiente a través
de la comparación post-extracción es exacto sobre el histograma, y la búsqueda sobre
la retícula de histogramas es lo que de hecho hace `invert_template` en
`evaluate.py`, sección 4.6). Vectores:

- *Impersonación* vía `verificar_t`: buscable directamente en el espacio de templates
  (retícula), sin pasar por imagen — la contramedida es `validar_template` + faceness
  (4.7).
- *Dodging* del detector (parches tipo Yang et al.) o **degradación deliberada**
  (gafas de sol, mascarilla: INFORME V7): contra un sistema de verificación, dodging
  es un ataque de **negación** contra el genuino, no una entrada para el impostor;
  nuestra evaluación lo trata correctamente como FRR, no como FAR.

**Lectura aplicable:** para pipelines de rasgos diseñados a mano, la literatura
adversarial de "pequeñas perturbaciones imperceptibles" es menos directamente
aplicable; la amenaza equivalente (y medida por nosotros) es la búsqueda directa en
el **espacio de templates**, que está mejor defendida aquí que en un FRS profundo
gracias a la validación de retícula.

### 4.4 El filtro "¿es una cara?" como clasificador atacable

**Literatura.** Cualquier clasificador lineal delante de un pipeline es un objetivo
adversarial clásico: el hiperplano se cruza con perturbaciones pequeñas
(mismo principio de Sharif et al., aplicado a un modelo mucho más débil). Además
hereda los **sesgos demográficos de su entrenamiento**, con la consecuencia
documentada por Buolamwini y Gebru (*Gender Shades*, FAT*/AIES 2018) y por NIST
(FRVT Part 3, *Demographic Effects*, NISTIR 8280, 2019): la precisión se degrada de
forma desigual por fenotipo, y los umbrales no se pueden elegir sin trade-offs
diferenciales.

**Mapeo en BigBrother.** Es la V14 del INFORME: `faceness` es una regresión
logística lineal sobre √template; el 0,5 % de las no-caras de test lo pasan por
diseño (umbral deliberadamente permisivo para no rechazar caras reales). Y es un
ejemplo **vivido** del sesgo de entrenamiento que la literatura predice: la primera
versión (entrenada con LFW solo) rechazaba las caras reales de Adam Pearson, Forest
Whitaker y Samuel L. Jackson; se reentrenó con aumentos fotométricos.
Tras el filtro queda siempre el umbral de similitud, así que evadir faceness **no
suficiente**: solo ahorra un paso del hill-climbing de 4.2.

### 4.5 Menagerie: lobos, corderos y usuarios desiguales

**Literatura.** Doddington et al. (*Sheep, goats, lambs and wolves*, ICSLP 1998)
formalizaron que los usuarios de un sistema biométrico **no son intercambiables**:
hay *wolves* (impostores que casan con cualquiera), *lambs* (víctimas fáciles contra
las que cualquiera casa), *goats* (genuinos que nunca verifican). Yager y Dunstone
(*The biometric menagerie*, IEEE TPAMI 2009) lo extienden y muestran que la
correlación entre puntuaciones genuinas e impostoras de un usuario predice su
vulnerabilidad; los efectos **persisten entre algoritmos y datos**, o sea, no son
ruido sino estructura.

**Mapeo en BigBrother.** Este marco explica las V4 y V19 sin necesidad de buscar
fallos: los 4 desconocidos de LFW más parecidos a Vinícius puntúan al nivel de sus
genuinos (0,32–0,35 vs 0,31–0,36) — Vinícius es, en el sentido de Doddington, un
*cordero* frente a hombres de su mismo fenotipo; y el retrato retocado de Paula
verifica contra 10 inscritas — un *wolf* aparente con foto adecuada. La consecuencia
de diseño que la literatura impone: **el FAR nominal en LFW no es el FAR operativo
contra un atacante que se elige a sí mismo** (el atacante elige ser el lobo). Las
mitigaciones que tenemos (umbral estricto, ident_threshold propio) reducen la
frecuencia pero no la estructura.

### 4.6 Gemelos y parecidos

**Literatura.** La evidencia más dura fuera de papers: el propio NIST evaluó
comerciales en **FRVT Part 9a** (*Face Recognition Verification Accuracy on
Distinguishing Twins*, NISTIR 8439, 2022) y encontró que, con puntuación de
similitud como único criterio, muchos FRS comerciales **no distinguen gemelos
idénticos**. Phillips et al. (*Distinguishing identical twins by face recognition*,
FG 2011, dataset MBE) llegaron a lo mismo. La literatura de morphing (Ramachandra
et al. 2023) confirma que gemelos/parecidos son el insumo ideal para morphs.

**Mapeo en BigBrother.** Es la V5, y el INFORME ya lo dice con todas las letras:
Paula Etxeberria (gemela) verifica como `aitana_etxebarria` 2/2; MLBP **no puede**
separar gemelos con un rasgo de textura global — solo depende del parecido foto-foto.
Estamos en el caso NIST: la conclusión no es "mal calibrado" sino "rasgo
insuficiente"; las defensas de la literatura son multibiométrico (huella, iris) o
PAD, ambas fuera de alcance.

### 4.7 Validación de templates: defensa contra la fabricación de rasgos (punto 4)

No es un ataque sino la contramedida a mapear: contra la fabricación de vectores de
rasgos (punto 4 de Ratha, canal extractor→comparador), BigBrother exige que un
template sea **exactamente lo que el extractor produce**: dimensión fija, no
negativo, cada histograma regional sumando 1, y valores en múltiplos de 1/256 (la
retícula de cuentas de píxeles de cada región de la rejilla). Esto desactiva los
vectores optimizados arbitrarios del hill-climbing clásico (Adler optimizaba en un
espacio continuo de eigenface; aquí el espacio de búsqueda es la retícula entera de
histogramas MLBP válidos) — no lo desactiva del todo, porque la retícula sigue siendo
buscable (4.2), pero acota. El segundo nivel es `faceness` (4.4). Es el equivalente,
en el mundo de rasgos diseñados a mano, de las "restricciones de espacio latente"
que la literatura adversarial usa para hacer perturbaciones perceptualmente válidas.

---

## 5. Ataques a la BD de templates (punto 6) y a los módulos (puntos 3/5/8)

### 5.1 Compromiso de BD: lectura, escritura, rollback, row-swapping

**Literatura.** Ross, Shah y Jain (*Biometric template security: Challenges and
solutions*, EUSIPCO 2005) sientan la taxonomía: compromiso de BD, sustitución de
templates, reutilización cruzada entre aplicaciones. ISO/IEC 24745 define las tres
propiedades deseables: **irreversibilidad** (no reconstruir la muestra), 
**revocabilidad** (anular/renovar tras compromiso) y **desvinculabilidad** (no
enlazar templates entre aplicaciones). Patel, Ratha y Chellappa (*Cancelable
biometrics: A review*, IEEE Signal Processing Magazine, 2015) recorren las
transformaciones que las consiguen.

**Mapeo en BigBrother.** Contra la **escritura** (tamper, rollback, borrado,
row-swapping) el sistema implementa una política de integridad completa y medida
(tests de manipulación de BD): huella SHA-256 + nº de filas + SHA de los modelos
firmada con HMAC en `keys/`; y el row-swapping se detecta cifrando con **AES-256-GCM
usando el id como AAD** (una fila movida a otro id no descifra). Contra la
**lectura** sin clave: templates cifrados **y ya proyectados** a 300-D (sección
5.2). Evaluado en el INFORME: todas las variantes detectadas, fail-closed.

Frente a ISO 24745 quedan dos propiedades fuera:
- **Revocabilidad parcial:** `construir_bd --force` regenera BD y clave, pero los
  templates en sí no son "renovables" por identidad (un template comprometido y
  reconstruido sigue casando con la cara del usuario; solo el cambio de clave evita
  que el *ciphertext* sirva).
- **Desvinculabilidad/irreversibilidad:** inexistente por diseño (V1). La literatura
  dice que esto es la norma, no la excepción: incluso los FRS profundos se invierten
  (siguiente sección). El consejo operativo de 24745 para este caso: clave fuera del
  alcance del atacante (KMS/HSM), que es exactamente lo que V1/V18 recomiendan.

### 5.2 Inversión de templates (reconstrucción de la muestra)

**Literatura.** Adler (CCECE 2003, y *Vulnerabilities in biometric encryption
systems*, 2005) reconstruye imágenes desde templates PCA/LDA — **nuestro caso
exacto**: el template almacenado es una proyección **lineal** (PCA+LDA) de un
histograma MLBP. Zhmoginov y Sandler (*Inverting face embeddings*, 2016) y Mai et al.
(*On the reconstruction of face images from deep face templates*, IEEE TPAMI
41(5), 2019) demuestran que ni los embeddings profundos de alta dimensión se salvan:
NbNet reconstruye caras identificables desde templates VGG-Face. La revisión de
Gomez-Barrero y Galbally (*Reversing the irreversible: A survey on inverse
biometrics*, Computers & Security 90, 2020) consolida el campo: los templates sin
protección **son** datos personales, y la reconstrucción es el trampolín para otros
ataques (una muestra reconstruida se reinyecta por el punto 2/4).

**Mapeo en BigBrother.** Es la V1, y es la vulnerabilidad más seria del sistema
(reconocida en el INFORME). La evaluación lo reproduce fielmente a la literatura: con
clave+BD (`docker exec`), `invert_template` optimiza por gradiente un histograma MLBP
**válido en la retícula** cuyo coseno con el template proyectado de Vinícius supera el
umbral en ~3 s de GPU; `verificar_t` lo acepta e `identificar_t` devuelve
`vinicius_junior`. Dos detalles que la literatura ayuda a entender:

- Almacenar el template **después** de la proyección LDA (en vez del histograma crudo
  de 22.656 valores) no protege: la proyección es lineal y de rango 300, pero la
  comparación es en ese mismo espacio, así que el atacante no necesita recuperar el
  histograma original — le basta cualquier solución del sistema lineal, proyectada y
  normalizada (y aún más: cualquier histograma de la retícula que proyecte al
  coseno alto).
- `validar_template` **no** detiene esto: el ataque produce deliberadamente un
  histograma de retícula válido. La validación detiene vectores arbitrarios, no
  inversiones bien hechas — coincide con la conclusión de la revisión de
  Gomez-Barrero y Galbally: contra inversión solo valen irreversibilidad real
  (hashing salado, cancelables) o no llegar a la BD (clave en HSM).

Mitigación práctica ya anotada (V1): separar clave de BD (env `BIGBROTHER_KEYS` /
`BIGBROTHER_KEY`, soportadas), KMS en despliegue real.

### 5.3 Troyanos de módulos y anulación de la decisión (puntos 3, 5, 8): hueco H1

**Literatura.** Ratha (2001) prevé explícitamente la sustitución del extractor y del
comparador por caballos de Troya y la anulación de la decisión tras la comparación.
Es el punto menos automatizable de la taxonomía: exige acceso de escritura al
software, y la literatura lo trata como problema de **integridad de software**
(firmas, despliegue de solo lectura), no de algoritmos biométricos.

**Mapeo en BigBrother.** Aquí el mapeo revela un **hueco real** que la evaluación
empírica no cubre (los tests atacan `data/` y `models/`, no el código): la huella
firmada en `keys/db.state` cubre la BD **y los modelos** (`yunet.onnx`,
`mlbp_projection.npz`, `faceness.npz`, `calibration.json`) — pero **no los ficheros
Python**. Un atacante con escritura en `bigbrother/` (sin tocar `models/` ni `data/`)
puede, p. ej.:

- cambiar `THRESHOLD` o el `>=` de `verificar_t` (punto 8, anulación de decisión);
- hacer que `Projection.__call__` devuelva un vector fijo (punto 5, Troyano);
- saltarse `validar_template` o `faceness` (punto 4).

...y el sistema arranca y verifica la integridad **sin que nada falle**, porque la
huella sigue siendo correcta. Con `docker exec` como root (V18) ya se tenía esta
capacidad de facto (V1 subsume H1 en la demo), pero el mapeo de Ratha lo hace
explícito: la integridad que damos es de *datos y modelos*, no de *software*. La
contramedida estándar de la literatura de supply-chain: firmar/verificar el paquete
(imagen Docker inmutable con digest pinneado ya lo aproxima; añadir el SHA del
código a `db.state` sería el equivalente local de lo que ya hacemos con los modelos).

### 5.4 Replay de datos intermedios (punto 2) e inscripción abierta

**Literatura.** El replay canónico de Ratha ("resubmit recorded data") asume un canal
interno interceptado. La literatura moderna de templates le da la vuelta: la muestra
**reconstruida** en 5.2 es un replay perfecto (Galbally et al. llaman a esto la
sinergia inversión→replay).

**Mapeo en BigBrother.** Los métodos `_sd`/`_t` son, estructuralmente, canales de
replay **oficiales**: aceptan el recorte o el template que en un sistema cerrado
nunca saldría del proceso. Sin frescura ni vida, cualquier template capturado (o
reconstruido) se reutiliza para siempre. Las mitigaciones alineadas con la
literatura: faceness + validación de retícula elevan el coste de fabricar el
template, y el decision-only (4.1) impide el hill-climbing cómodo; pero un template
capturado una vez (5.2) vale para siempre — la renovación de credenciales no existe
(ver revocabilidad en 5.1). En el lado de la **inscripción**: Ratha no la enumera
porque asume enrolamiento supervisado; la literatura de morphing (sección 3) sí la
trata como superficie de ataque. BigBrother deja la inscripción abierta (V15) y el
poisoning por `actualizar=True` (V16) es su cadena de persistencia; la deduplicación
y el "actualizar solo si verifica" son los controles que la cortan parcialmente.

### 5.5 Denegación de servicio

**Literatura.** La disponibilidad es la Cenicienta de la literatura biométrica (las
revisiones de seguridad genérica — p. ej. Fleming, *Biometric security: concepts,
issues and flaws*, 2004 — la mencionan pero apenas hay trabajo específico). Para FRS
se discute sobrecarga computacional (imágenes gigantes) y, más recientemente, el
**trade-off fail-closed**: toda política de integridad estricta convierte un bypass
en una caída.

**Mapeo en BigBrother.** Ambos están presentes y bien tratados: anti-bomba de
descompresión y límites de tamaño (`imageio.py`, medido: imágenes de 6000 px
reescaladas, ficheros > 30 MB rechazados); y el fail-closed asumido como decisión
(V10): quien pueda escribir en `db.csv` no puede entrar, pero puede **tirar** el
sistema. Con el marco de la literatura lo enmarcamos correctamente: V10 no es un
defecto de diseño sino la elección deliberada de confidencialidad/integridad sobre
disponibilidad cuando el atacante ya tiene escritura en la BD.

---

## 6. Matriz de mapeo literatura ↔ BigBrother

| Familia (literatura) | Referencias clave | Punto Ratha | Vía en BigBrother | ¿Aplica? | Contramedida existente | En INFORME |
|---|---|---|---|---|---|---|
| Presentación (foto/pantalla/máscara/deepfake) | ISO 30107; Ramachandra & Busch 2017; Matineh et al. 2024 | 1 | `verificar` con foto física | **Sí, total** | ninguna (sin PAD) — por alcance | V2 |
| Morphing | Ferrara et al. 2014; Scherhag et al. 2017; Venkatesh et al. 2020; Ramachandra et al. 2023 | 1 (en inscripción) | `verificar` (medido); inscripción encadenada con `actualizar` (A3) | Parcial: la foto morphed verifica; *magic passport* no logrado; *poisoning* sí | dedup + "actualizar solo si verifica" | V3, V16 |
| Hill-climbing con puntuaciones | Adler 2003; Galbally et al. 2010; Gomez-Barrero et al. 2012; Martinez-Diaz et al. 2006 | 4 | `verificar` vía código (`_score_id`) | Caja blanca: sí | API sin puntuaciones (BioAPI al extremo) | V9 |
| Hill-climbing con decisión binaria | Dong et al. CVPR 2019; NIST SP 800-63B (límite) | 4 | `verificar_t` repetido | **Sí, medido (A2)** | retícula + faceness elevan el coste; **sin límite de intentos** | (H3, V8) |
| Adversariales digitales/físicos | Sharif et al. CCS 2016; Deb et al. 2019; Yang et al. 2019 (detector) | 1+4 | perturbaciones en imagen; parches vs YuNet | Parcial (pipeline no diferenciable + caja blanca trivial) | igual que hill-climbing | V7 (dodging), V9 |
| Menagerie (lobos/corderos) | Doddington et al. 1998; Yager & Dunstone TPAMI 2009 | — (estadístico) | elección de atacante/parecido | **Sí, estructural** | umbral estricto reduce frecuencia, no estructura | V4, V19 |
| Sesgo demográfico | Buolamwini & Gebru 2018; NISTIR 8280 (FRVT-3) 2019 | — | FAR/FRR desiguales; faceness sesgable | **Sí** (medido en faceness) | reentrenamiento con aumentos; umbral permisivo | V4, V14, §4.4 |
| Gemelos | NISTIR 8439 (FRVT 9a) 2022; Phillips et al. FG 2011 | 1 | `verificar` de la gemela | **Sí, intrínseco** | ninguna posible con MLBP solo | V5 |
| Fabricación de templates (punto 4) | Ratha 2001; espacio de soluciones de Adler | 4 | `verificar_t` con vector sintético | Sí (limitado) | **validación de retícula** + faceness | fortalezas del INFORME |
| Inversión de templates | Adler 2003/05; Zhmoginov & Sandler 2016; Mai et al. TPAMI 2019; Gomez-Barrero & Galbally 2020 | 6 (con clave) | clave+BD → template válido por gradiente | **Sí, medido** | cifrado (vale hasta robo de clave); falta irreversibilidad | V1, V18 |
| Compromiso de BD | Ross et al. 2005; ISO 24745; Patel et al. 2015 | 6 | escritura/rollback/swap en `db.csv` | **No (detectado)** | HMAC + AAD por id + fail-closed | fortalezas |
| Troyano extractor/comparador/decisión | Ratha 2001 (puntos 3, 5, 8) | 3, 5, 8 | escribir en `bigbrother/*.py` | **Sí — hueco H1** | la huella cubre modelos, **no el código** | (nuevo) |
| Replay de intermedios | Ratha 2001 (punto 2); Gomez-Barrero & Galbally 2020 | 2 | `_sd`/`_t` reutilizable para siempre | Sí (diseño del enunciado) | faceness + retícula; sin frescura | fortalezas (parcial) |
| Inscripción abierta / poisoning | Ferrara et al. 2014 (lado inscripción); NIST 800-63B | — | `inscribir` libre + `actualizar=True` | Sí | id no sobrescribible, dedup, verificar-para-actualizar | V15, V16 |
| DoS / disponibilidad | Fleming 2004; (trade-off fail-closed) | todos | bomba de imágenes; escribir `db.csv`; H1 también | Sí (asumido) | límites de imagen; fail-closed deliberado | V10, V8 |

---

## 7. Huecos que el mapeo revela (no cubiertos por INFORME.md)

Resumen de lo nuevo que aporta cruzar con la literatura, por prioridad.
**Todos ellos se han medido ya de forma empírica**: proceso, cifras y veredictos en
[`ATAQUES_EMPIRICOS.md`](ATAQUES_EMPIRICOS.md) y
[`resultados/ataques_literatura.json`](resultados/ataques_literatura.json)
(`python scripts/ataques_literatura.py`).

1. **H1 — Integridad de software:** la huella firmada cubre BD+modelos pero no el
   código Python (`security.models_digest` → `config.MODEL_FILES`). Un atacante con
   escritura en el paquete controla extractor, comparador y decisión sin ser
   detectado. Corrección barata: incluir un digest del paquete `bigbrother/` en
   `db.state` (o firmar la imagen, que el despliegue Docker ya permite con digest
   pinneado).
   → **medido (A1): éxito total del troyano, arranque indetectado; la corrección del
   digest del código lo detecta (ATAQUES_EMPIRICOS §A1).**
2. **H2 — Morphing por la vía de inscripción:** solo medimos la vía verificación;
   la canónica de la literatura (*magic passport*) ataca la inscripción. El
   encadenamiento `inscribir(morph)` + `actualizar` merece un experimento propio
   (con la deduplicación activada, medir cuántos morphs de víctima+desconocido
   sobreviven a `dedup_threshold`).
   → **medido (A3): el dedup bloquea el id nuevo; el camino `actualizar=True` no aplica
   dedup y el morph se cuela (α ≤ 0,5; 0,2–0,5 si la foto de la víctima no está inscrita); el magic passport con fotos hold-out no se logra
   con el generador débil (blending lineal, límite inferior).**
3. **H3 — Hill-climbing binario de caja negra:** el INFORME tilda V9 de "solo caja
   blanca"; la literatura (Dong et al.) dice que el feedback de decisión basta con
   presupuesto de consultas, y no limitamos la tasa (V8). Experimento: hill-climb
   sobre la retícula de histogramas usando solo el `bool` de `verificar_t`, contando
   consultas. Predicción: converge más lento que el de caja blanca (~900–1.600) pero
   converge.
   → **medido (A2): converge — 4.058 consultas bool y un template 0,95 similar al atacante (0,61–0,91 si la foto de partida no está inscrita)
   que verifica como la víctima (predicción cumplida, mismo orden de magnitud que la caja
   blanca).**
4. **H4 — Revocabilidad (ISO 24745):** tras un compromiso (V1) la renovación es
   global (`--force`, regenera clave), no per-identidad; y un template filtrado es
   válido para siempre al no existir ni frescura ni cancelación. Como mínimo,
   documentar que el modelo de amenaza asume que la filtración de un template
   equivale a la filtración de la credencial permanente de ese usuario.
   → **medido (A4): confirmado — el template robado verifica tras la rotación de clave
   (templates bit-idénticos, ciphertexts distintos).**
5. **H5 — Faceness como superficie demográfica:** la regresión logística ya mostró
   sesgo de entrenamiento (§4.4); por la literatura de sesgo (NISTIR 8280),
   cualquier umbral único de faceness reparte el error desigualmente por fenotipo.
   Ya mitigado con umbral permisivo + red de seguridad (umbral de similitud), pero es
   estructura permanente del sistema.

---

## 8. Conclusiones del mapeo

- El sistema se defiende bien exactamente donde la literatura dice que se puede
  defenderse **sin** cambiar el rasgo biométrico: integridad de BD (punto 6), bloqueo
  del canal de puntuaciones (contra Adler/Galbally), retícula contra la fabricación
  de rasgos (punto 4), políticas de inscripción contra el morphing y el poisoning.
- Quedan abiertas exactamente las tres familias para las que **no hay defensa
  algorítmica** con un rasgo MLBP solo, y la literatura lo confirma con la misma
  claridad que nuestra evaluación: presentación sin PAD (ISO 30107 fuera de alcance),
  gemelos (NIST FRVT 9a: ni los comerciales), y compromiso con clave (inversión:
  Gomez-Barrero & Galbally 2020; mitigación operativa: KMS/HSM).
- El cruce con la literatura aporta valor nuevo en H1–H5: sobre todo H1 (integridad
  de software, la única superficie donde el sistema da un falso sentido de
  seguridad: "verifico integridad" — pero solo de datos y modelos) y H3 (el
  decision-only no es inmunidad, solo encarecimiento: Dong et al.).
- El marco de Ratha encaja notablemente bien en una arquitectura con tres niveles de
  API: convierte la enumeración abstracta de 8 puntos en una lista concreta de
  componentes, y deja claro que los métodos `_sd`/`_t` son la materialización
  deliberada de los puntos 2 y 4 como superficie pública.

---

## Referencias

- A. Adler, *Sample images can be independently restored from face recognition templates*, IEEE CCECE, 2003.
- A. Adler, *Vulnerabilities in biometric encryption systems*, Proc. Audio- and Video-Based Biometric Person Authentication (AVBPA), 2005.
- J. Buolamwini, T. Gebru, *Gender Shades: Intersectional accuracy disparities in commercial gender classification*, PMLR 81 (FAT*/AIES), 2018.
- Y. Dong, H. Su, B. Wu, Z. Li, W. Liu, T. Zhang, J. Zhu, *Efficient decision-based black-box adversarial attacks on face recognition*, IEEE CVPR, 2019.
- D. Deb, J. Zhang, A. K. Jain, *AdvFaces: Adversarial face synthesis*, arXiv:1908.05008, 2019.
- G. Doddington, W. Liggett, A. Martin, M. Przybocki, D. Reynolds, *Sheep, goats, lambs and wolves: A statistical analysis of speaker performance in the NIST 1998 speaker recognition evaluation*, ICSLP, 1998.
- A. Ferrara, A. Franco, D. Maltoni, *The magic passport*, IEEE IJCB, 2014.
- M. Ferrara, A. Franco, D. Maltoni, *Morphing attack potential*, arXiv:2204.13374, 2022.
- S. Fleming, *Biometric security: concepts, issues and flaws*, OUCS-2004-21, University of Otago, 2004.
- J. Galbally, C. McCool, J. Fierrez, S. Marcel, *On the vulnerability of face verification systems to hill-climbing attacks*, Pattern Recognition 43(3), 2010.
- M. Gomez-Barrero, J. Galbally, J. Fierrez, J. Ortega-Garcia, *Face verification put to test: A hill-climbing attack based on the uphill-simplex algorithm*, IEEE ICB, 2012.
- M. Gomez-Barrero, J. Galbally, *Reversing the irreversible: A survey on inverse biometrics*, Computers & Security 90, 2020 (arXiv:2401.02861).
- K. Hanaoka et al., *Ongoing Face Recognition Vendor Test (FRVT) Part 9a: Face recognition verification accuracy on distinguishing twins*, NISTIR 8439, NIST, 2022.
- P. Grother, E. Ngan, K. Hanaoka, *Face Recognition Vendor Test (FRVT) Part 3: Demographic effects*, NISTIR 8280, NIST, 2019.
- G. Mai, K. Cao, P. C. Yuen, *On the reconstruction of face images from deep face templates*, IEEE TPAMI 41(5), 2019 (arXiv:1703.00832).
- M. Martinez-Diaz, J. Fierrez, F. Alonso-Fernandez, J. Ortega-Garcia, *Hill-climbing and brute-force attacks on biometric systems: a case study in match-on-card fingerprint verification*, IEEE ICCST, 2006.
- E. Maiorana, G. E. Hine, P. Campisi, *Hill-climbing attacks on multibiometrics recognition systems*, IEEE Trans. Information Forensics and Security 10(5), 2015.
- P. Matineh, B. Amin, Y. Qi et al., *Presentation attack detection: a systematic literature review*, ACM Computing Surveys, 2024.
- K. O'Haire, S. Soleymani, B. Chaudhary et al., *Identical twins face morph database generation*, IEEE IJCB, 2022.
- V. M. Patel, N. Ratha, R. Chellappa, *Cancelable biometrics: a review*, IEEE Signal Processing Magazine 32(5), 2015.
- P. J. Phillips et al., *Distinguishing identical twins by face recognition*, IEEE FG, 2011.
- R. Ramachandra, C. Busch, *Presentation attack detection methods for face recognition systems: a comprehensive survey*, ACM Computing Surveys 50(1), 2017.
- R. Ramachandra, S. Venkatesh, G. Jaswal et al., *Vulnerability of face morphing attacks: a case study on lookalike and identical twins*, arXiv:2303.14004, 2023.
- N. K. Ratha, J. H. Connell, R. M. Bolle, *Enhancing security and privacy in biometrics-based authentication systems*, IBM Systems Journal 40(3), 2001.
- A. Ross, J. Shah, A. K. Jain, *Biometric template security: challenges and solutions*, EUSIPCO, 2005.
- C. Rathgeb, A. Uhl, *Attacking iris recognition: an efficient hill-climbing technique*, IEEE ICPR, 2010.
- K. Scherhag et al., *Biometric systems under morphing attacks: assessment of morphing techniques and vulnerability reporting*, BIOSIG, 2017.
- M. Sharif, S. Bhagavatula, L. Bauer, M. K. Reiter, *Accessorize to a crime: real and stealthy attacks on state-of-the-art face recognition*, ACM CCS, 2016.
- I. Singh, K. Kakizaki, T. Araki, *Simultaneous adversarial attacks on multiple face recognition system components*, arXiv:2304.05048, 2023.
- S. K. Venkatesh, R. Ramachandra, K. B. Raja et al., *Face morphing attack generation & detection: a comprehensive survey*, arXiv:2011.02045, 2020.
- X. Yang, F. Wei, H. Zhang, J. Yang, *Design and interpretation of universal adversarial patches in face detection*, arXiv:1912.05021, 2019.
- N. Yager, T. Dunstone, *The biometric menagerie*, IEEE TPAMI 31(2), 2009.
- A. Zhmoginov, M. Sandler, *Inverting face embeddings with convolutional neural networks*, arXiv:1606.04189, 2016.
- ISO/IEC 30107-1/-3, *Information technology — Biometric presented attack detection*.
- ISO/IEC 24745, *Information technology — Security techniques — Biometric information protection*.
- NIST SP 800-63B, *Digital Identity Guidelines: Authentication and lifecycle management*.
