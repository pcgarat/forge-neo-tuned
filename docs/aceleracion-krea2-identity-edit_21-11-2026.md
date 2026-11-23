# Última modificación: 2026-11-21

# Acelerar Krea 2 + Identity Edit (RTX 4060 8 GB)

Investigación y mediciones para reducir el tiempo de **Krea 2 con Krea2 Identity Edit** en este stack
(Forge Neo en Docker, RTX 4060 8 GB). El objetivo de este documento es fijar el **índice de puntos a
analizar uno a uno**; las mediciones de hoy sirven de línea base y de evidencia inicial, no de
conclusión cerrada.

**Ámbito:** `img2img`/`txt2img` con el accordion *Krea2 Identity Edit* (patch de backend +
`sd-forge-krea2-edit`), checkpoint Krea 2 Turbo, TE Qwen3-VL 4B con visión y LoRA `krea2_identity_edit`.

---

## 0. Resumen ejecutivo

**Receta final medida** (`make bench-edit`, 640×640, Euler/Simple, CFG 1, misma fuente y semilla):

| Configuración | Pasos | Mediana | VRAM |
|---|---|---|---|
| Perfil de partida: `--lowvram` + TE en CPU + v1.2 | 8 | **145,6 s** | 5,1 GB |
| + TE en GPU | 8 | 35,2 s | 4,9 GB |
| + V2 r64 + `ref_boost` 2 | 8 | 31,0 s | 4,3 GB |
| + distill 4-step (receta base) | 4 | 19,9 s | 5,4 GB |
| **+ SPEED a 832 px** | **4** | **~13 s** (1,67× sobre 21,0 s a 832) | — |

**≈11× más rápido** que el punto de partida a 832 px, con calidad igual o mejor que el baseline
v1.2 (más detalle y nitidez, brillo estable, sin artefactos). 0 OOM en todas las configuraciones.

Las cuatro palancas, por orden de impacto:

1. **Dónde corre el text encoder: 4,1×.** Con `--lowvram`, `text_encoder_device()` devuelve
   **CPU** y el Qwen3-VL (5,3 GB fp8) hace el prefill de la imagen de referencia en CPU.
2. **SPEED (primeros pasos a media resolución): 1,67× a 832 px, 1,91× a 1024 px.** Exige
   configuración manual; por defecto no hace nada (ver P12).
3. **Destilación a 4 pasos: 1,70×.** Se compone con V2 sin interferencias.
4. **V2 del LoRA + boost 2: 1,15×** y, sobre todo, **exposición estable** (v1.2 se destrozaba con
   boost 2; V2 no).

Lo que **no** funciona, medido y descartado: W4A8 (−28 % de velocidad, misma calidad), la
variante "segura" 0,75+boost 2 con v1.2 (revienta la exposición), `Euler a` (AssertionError con
Krea 2), el scheduler `Turbo` (brillo 112) y desactivar el vision nativo de Qwen3-VL.

Gratis y recomendado: **scheduler `beta`** (indistinguible de `simple` pero es la recomendación
de la comunidad) o **`Karras`** si se busca +35 % de detalle.

Corolario crítico: **antes de optimizar nada hay que medir con el TE en GPU y con el LoRA
activo.** Sin lo primero, un factor 4 domina todo; sin lo segundo, se mide un camino que no es
edición (y sale ~3× más rápido).

---

## 1. Hardware y stack

| Dato | Valor |
|---|---|
| GPU | RTX 4060, 8 GB, sm89 (Ada) |
| CPU / RAM | 16 hilos, 32 GB |
| Forge Neo | pin `5998fb3` (`FORGE_NEO_REF`), Neo 2.33. Las cifras de §2–§4 se midieron con el pin anterior (`41359cd`, 2.29.1) |
| torch | 2.13.0+cu130 |
| Atención | Comfy-Kitchen INT8 activo (`--use-ck-attention`), xformers para VAE |
| Checkpoint | `krea2_krea2Int4Convrot_v10Turbo` (6,4 GB) |
| TE | `Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled` (4,9 GB) |
| VAE | `krea2RealVae_v10` |
| LoRA | `krea2_identity_edit_v1_2_r64` (0,46 GB) |
| Flags actuales | `--cuda-malloc --lowvram --fp8_e4m3fn-unet --reserve-vram 2 --cuda-stream --pin-shared-memory --mmap-torch-files --use-ck-attention` |

---

## 2. Línea base medida (2026-11-21)

Instrumento: `make bench-edit` → [`docs/bench-edit_21-11-2026.md`](bench-edit_21-11-2026.md)
(640×640, 8 pasos, CFG 1, Euler/Simple, `ref_boost` 4, AR `fit`, semilla 777, LoRA
`krea2_identity_edit_v1_2_r64` **verificado en el `info` de cada respuesta**).

> **Aviso de método — corregido.** Las cifras de la primera pasada de esta sección estaban
> tomadas **sin el tag `<lora:...>` en el prompt**, así que el modelo editaba sin LoRA y los
> tiempos eran ~3× más bajos. Lo detectó el guardián del benchmark (el `info` no contenía
> `Lora hashes`). Las cifras de abajo son las buenas; las antiguas se conservan en §2.5 por
> trazabilidad, marcadas como inválidas.

### 2.1 Identity Edit real: TE en CPU vs GPU × con/sin LoRA

| Configuración | Sin LoRA | Con LoRA (edit real) |
|---|---|---|
| TE en CPU (`--lowvram`, sin flag) | 131,1 s | **145,6 s** |
| TE en GPU (`--text-enc-device cuda`) | 12,0 s | **35,2 s** |
| Ganancia del TE en GPU | 10,9× | **4,1×** |

Dos hechos que ordenan todo lo demás:

1. **El TE en GPU sigue siendo la palanca dominante**, pero en un edit real la ganancia es
   **4,1×**, no 11×: el coste del LoRA y del DiT tapa parte del ahorro.
2. **El LoRA de edición cuesta ~3×** (12,0 s → 35,2 s a igual resolución y pasos) porque añade
   los tokens de la fuente como contexto al DiT. Es el coste intrínseco de "editar" frente a
   "referenciar", no un problema de configuración.

Pico de VRAM en ambas ramas del edit real: ~5,0 GB de 8 GB. **0 OOM**.

### 2.2 Escalado con pasos (edit real, TE en GPU, 640×640, grounding 768)

| Pasos | Tiempo |
|---|---|
| 4 | 6,6 s (sin LoRA) |
| 8 | 35,2 s (con LoRA) |
| 12 | 17,2 s (sin LoRA) |

Solo válido el punto medido con LoRA; el resto sigue pendiente con el LoRA activo (es el
primer trabajo que el nuevo benchmark hace barato).

### 2.3 `grounding_px` × ubicación del TE (camino referencia, sin LoRA — histórico)

| `grounding_px` | TE en CPU | TE en GPU |
|---|---|---|
| 768 | 135,4 s | 12,0 s |
| 384 | 60,9 s | 11,0 s |
| 256 | 39,4 s | 10,9 s |

Sigue siendo válido para el camino de **referencia** (sin LoRA). Con el TE en GPU la curva se
aplana: `grounding_px` es un dial de **calidad**, no de velocidad. Pendiente repetirlo con el
LoRA activo, donde el peso de los tokens de visión sobre el total es menor.

### 2.4 `ref_boost` (camino referencia)

| `ref_boost` | TE en CPU | TE en GPU |
|---|---|---|
| 4,0 | 135,4 s | 12,0 s |
| 1,0 | 121,4 s | 10,9 s |

~10 % de coste, por el bias denso `[1,1,L,L]` que fuerza la ruta *masked* del kernel. En el
edit real ese porcentaje se diluye (el LoRA pesa mucho más). Microbenchmark aislado (bf16,
24 heads, d=128):

| Longitud | CK int8 sin máscara | CK int8 con bias | SDPA sin máscara | SDPA con bias |
|---|---|---|---|---|
| 3.600 | 4,8 ms | ~4,0 ms | 7,2 ms | 12,2 ms |
| 5.600 | 5,2 ms | 25,9 ms | 12,5 ms | 24,9 ms |
| 8.800 | 12,0 ms | 61,3 ms | — | — |

CK gana a SDPA en todos los casos, también con máscara. **No hay nada que cambiar aquí.**

### 2.5 Cifras retiradas (inválidas: sin LoRA en el prompt)

Se conservan solo para no repetir el error. **No usar como referencia.**

| Caso | Tiempo | Por qué no vale |
|---|---|---|
| "Edit 1024, g768, TE GPU" | 37,3 s | sin LoRA → camino referencia |
| "Edit 1280, g768, TE GPU" | 75,3 s | sin LoRA; además inestable (outlier 106,7 s) |
| "Edit 1024, g768, sin `--lowvram`" | 40,9 s | sin LoRA |
| "Edit 1280, g768, sin `--lowvram`" | 78,7 s (outlier 378,3 s) | sin LoRA; 1280 px no es estable en 8 GB |

Lección: **1280 px sigue siendo el techo práctico** (presión de RAM↔VRAM con outliers de 3–5×),
pero la comparación de perfiles a esa resolución hay que rehacerla con el LoRA activo.

### 2.6 Ruido de medición

Corridas anómalas (129 s donde la mediana era 13 s) coincidiendo con el **CivitAI Browser
descargando en background** (aria2). Cualquier benchmark debe hacerse con descargas cerradas.
Además hay un problema conocido de memoria entre renders en Forge Neo
([issue #1275](https://github.com/Haoming02/sd-webui-forge-classic/issues/1275), cerrado en
Neo 2.29.2). El nuevo `make bench-edit` calienta y luego mide la mediana, lo que reduce el
efecto del primer arranque.

---

## 3. Índice de puntos a analizar (uno a uno)

Ordenado por impacto esperado. "Estado" = qué sabemos ya; "Decisión" = qué hay que cerrar.

| # | Punto | Impacto esperado | Estado |
|---|---|---|---|
| P1 | TE en GPU (`--text-enc-device cuda`) | **4,1×** en edit real (10,9× sin LoRA) | **Hecho**: eje `TE=`, preset `make krea2`, `up-interactive` |
| P2 | Perfil de VRAM (`lowvram` vs `normalvram`) | **0 % (indistinguible)** | **Medido**: 1280 px OK, 1536 px OOM |
| P3 | `grounding_px` (calidad ↔ velocidad) | ~1 s con TE en GPU | Medido sin LoRA; repetir con LoRA |
| P4 | Checkpoint: INT4 / INT8 / fp8 / W4A8 | W4A8: **−28 % (peor)** | **Medido y descartado** (P4) |
| P5 | TE: fp8 / int8_convrot / bf16 / abliterated | velocidad igual; el oficial es **inestable** | **Medido** (P5): abliterado por estabilidad |
| P6 | VAE: oficial vs RealVAE vs HD | **0 % de tiempo; cambia el acabado** | **Medido** (P6) |
| P7 | LoRA: r64 / full / V2 + placement | **V2 1,15× más rápido y más estable** | **Medido** (P7); placement pendiente || P8 | Distill 4-step (y 3-step) | **1,70×**; pierde 1–10 % de detalle | **Medido** (P8) |
| P9 | `ref_boost` y el bias denso L×L | **0 % de tiempo**; 351 MB a 1280 px | **Medido** (P9): se queda |
| P9b | Bias de atención disperso | **no viable sin tocar el kernel** | **Analizado**: 3 vías descartadas |
| P10 | Backend de atención con máscara (CK vs SDPA) | 1,5–2× | Medido en microbench: CK gana |
| P11 | Caché de pasos (TeaCache / FBC / Spectrum) | sin soporte Krea 2 en Forge | **Descartado** (P11) |
| P12 | SPEED (primeros pasos a media resolución) | **1,67–1,91×** | **Medido** (P12): segunda palanca |
| P13 | Pre-encode del source antes del sampling | evita thrashing | Verificado en el patch |
| P14 | Auto face-ref prep (2 pasadas anidadas) | ×3 coste | Medido conceptualmente |
| P15 | Caché de condicionamiento entre generaciones | variable | Pendiente |
| P16 | Varianza / thrashing (descargas, memoria) | hasta 10× | Detectado, sin mitigar |
| P17 | Benchmark reproducible de Identity Edit | — | **Hecho**: `make bench-edit` (+ `-sweep`) |
| P18 | Auto-attach de `sd-forge-krea2` (rompe si el VAE no dice qwen/wan) | bloquea el arranque | **Detectado** (P18) |

---

## 4. Detalle de los puntos

### P1 — TE en GPU (hecho)

**Por qué:** `--lowvram` deja `vram_state = LOW_VRAM`, y `text_encoder_device()` solo devuelve GPU en
`NORMAL_VRAM`/`HIGH_VRAM`. El TE se ejecuta en CPU con un modelo de 5,3 GB y una torre visual de
24 bloques.

**Evidencia:** 135 s → 12 s. Medido también que el UNet se descarga antes de cargar el TE, así que
en la práctica no explota: pico observado ~5,2 GB de 8 GB.

**Matiz que cambia el diseño:** con **texto solo** el TE en GPU es *ligeramente peor* (7,9 s vs 6,1 s
a 640 px) porque compite con el UNet y fuerza recargas. La ganancia aparece solo cuando hay **tokens
de visión**: el prefill de la imagen en GPU es ~100× más rápido que en CPU, y ahí el DiT ni siquiera
está cargado (Forge descarga el UNet antes del encode de texto). Por eso el eje **no** es "siempre
GPU": es un eje por modelo.

**Implementación (en repo):**

- `Makefile`: eje `TE=auto|gpu|cpu` → `TE_FLAGS` (`--text-enc-device cuda|cpu`), añadido en
  `_launch-model` junto a `ATTN_FLAGS`. `make krea2` fija `TE=gpu`; `klein`/`wan`/`run` quedan en
  `auto`. Override con `make run TE=...`.
- `scripts/choose_profile.py`: `Recommendation.te`; recomienda `TE=gpu` solo para `krea2` con
  `long_sequence` (Moodboard/Identity Edit), que es la pregunta que ya existía en el asistente.
- `.env.example`: variante documentada para `make up` (perfil por `.env`).
- Tests: `make test-choose-profile` (27 OK), incluyendo que otros modelos nunca activan el eje.

**Riesgo abierto:** OOM puntual con Moodboard + Edit simultáneos o ≥1,5 MP. Hay que forzar ese caso
en la validación. En este repo quedó arrancado con el perfil 8 GB + `TE=gpu`; 12 s estables en 2
corridas consecutivas.

### P2 — Perfil de VRAM y techo de resolución (medido)

**Perfil de VRAM: irrelevante con la receta final.** `--lowvram` y `normalvram` dan lo mismo:

| Perfil | 832×832 | 1024×1024 |
|---|---|---|
| `--lowvram` (perfil 8 GB) | 29,58 s | 43,80 s |
| `normalvram` (sin `--lowvram`) | 29,61 s | 43,84 s |

Diferencia dentro del ruido (±0,1 s). Tiene sentido: con TE en GPU y 4 pasos el modelo cabe y el
offload deja de ser el cuello. **No hay que cambiar el perfil**; `--lowvram` se queda porque es
el que protege frente a picos (Moodboard + Edit, hires).

**Escalado con la resolución** (receta final, 4 pasos, `ref_boost` 2):

| Resolución | Mediana | vs 640 | Área vs 640 | VRAM pico |
|---|---|---|---|---|
| 640×640 | 19,9 s | 1,00× | 1,00× | 5,4 GB |
| 832×832 | 29,6 s | 1,49× | 1,69× | 5,4 GB |
| 1024×1024 | 43,8 s | 2,20× | 2,56× | 5,5 GB |
| 1280×1280 | 73,2 s | 3,68× | 4,00× | 6,3 GB |

El tiempo escala casi linealmente con el área (el coste de atención manda), y la VRAM sube poco.
**0 OOM hasta 1280 px.**

**1536×1536 (2,36 MP): OOM.** El límite está entre 1280 y 1536 px:

```
OutOfMemoryError: Allocation on device 0 would exceed allowed memory
  Currently allocated : 5.85 GiB   Requested: 385.57 MiB   Device limit: 7.60 GiB
  Free (according to CUDA): 106.75 MiB
```

El fallo es **limpio y rápido** (2 s, excepción de la API, el servidor sigue vivo): no cuelga ni
deja el contenedor inservible. Para 1536 px haría falta `--reserve-vram 0`, VAE en tiles y
probablemente tumbar `--pin-shared-memory`. **Techo práctico en 8 GB: entre 1,6 y 2,3 MP**, con
1280 px como valor alto cómodo y 1024 px como punto dulce de uso diario.

**1280 px es estable con la receta final.** Tres corridas seguidas: 73,26 / 73,16 / 73,11 s
(0,15 s de dispersión). Los outliers de 100–380 s que se vieron en las primeras mediciones **no
eran un techo de la resolución**: eran consecuencia de medir sin el LoRA (camino distinto). Con el
benchmark actual, 1280 px entra sin OOM en todas las corridas.

### P3 — `grounding_px`

Es un dial de **calidad**, no de rendimiento, cuando el TE está en GPU: 768 → 256 ahorra ~1 s. El
LoRA se entrenó con jitter 384–768 ([krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)),
así que 256 está fuera de distribución: en la comparativa visual de hoy la identidad se mantuvo
razonablemente, pero es un riesgo que hay que fijar por escrito. Propuesta: **384 como valor por
defecto** del preset (dentro de distribución, ~9 % más rápido que 768 a igual TE), y 768 solo cuando
se busque máxima semejanza.

### P4 — Checkpoint: INT4 vs W4A8 (medido)

Hipótesis de la comunidad: W4A8 (pesos INT4, activaciones INT8) corrige el aspecto "crujiente"
del INT4 a coste cero de memoria, porque reutiliza el mismo tamaño de fichero.

**Resultado en esta GPU: no compensa.** [Krea-2-Turbo-w4a8](https://huggingface.co/realrebelai/Rebels_w4a8s)
(6,8 GB) frente al INT4 ConvRot actual (6,4 GB):

| Checkpoint | Mediana (edit, 640 px, 8 pasos) | Tamaño | VRAM pico |
|---|---|---|---|
| INT4 ConvRot (`krea2Int4Convrot_v10Turbo`) | **35,0 s** | 6,4 GB | 3,0 GB |
| W4A8 (`krea2_turbo_w4a8`) | **44,8 s** | 6,8 GB | 5,4 GB |

**W4A8 es un 28 % más lento y un 6 % más grande.** Y la calidad es indistinguible:

| Checkpoint | Detalle (HF) | Nitidez | Brillo |
|---|---|---|---|
| INT4 | 2977 | 6,24 | 67,7 |
| W4A8 | 3014 | 6,23 | 67,8 |

Diferencia L1 entre ambos: **1,41** (escala 0–255), es decir, el mismo ruido de cuantización. A
ojo no se distingue.

**Por qué:** el soporte nativo de `asym_w4a8_int8` existe en `comfy_kitchen` (`AsymW4A8Int8Layout`
registrado) y Forge tiene la ruta en `operations_mixed_precision.py`, pero el kernel fusionado
exige grupos ConvRot de 256 y en este build/GPU no entra en la ruta rápida: el fichero es más
grande (mezcla capas en alta precisión) y el camino de dequant pesa más. El INT4 ConvRot de
Comfy-Org sí usa el kernel nativo por la ruta `convrot_w4a4`.

**Veredicto: descartado.** El INT4 ConvRot actual ya es la mejor opción en 8 GB. W4A8 tendría
sentido si el kernel nativo entrara en juego (otra GPU o una versión más nueva con grupos de 128).

> **Nota de método:** el INT4 se midió con `override_settings` en el payload de generación, que
> **sí funciona** (el que devuelve HTTP 500 es el `POST /sdapi/v1/options`). Para fijar el modelo
> en la UI sin reiniciar: `override_settings_restore_afterwards: false`.

### P5 — Text encoder: oficial vs abliterado (medido)

Comparación limpia: `qwen3vl_4b.safetensors` (oficial de Comfy-Org) y
`Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled` tienen **formato byte-idéntico** (mismos
dtypes: 252 F8_E4M3 + 461 BF16 + 252 U8, ambos 4,88 GiB). Solo cambian los pesos, así que cualquier
diferencia es del modelo, no de la cuantización.

**Velocidad: idéntica.** 19,90 s (oficial) vs 19,35 s (abliterado) con la receta final a 640 px.
La diferencia está dentro del ruido; el TE no es el cuello en un edit de 4 pasos con visión.

**Calidad: difieren de forma sistemática.** Misma fuente, misma semilla:

| Semilla | TE | Nitidez | Brillo medio | Detalle (HF) |
|---|---|---|---|---|
| 777 | oficial | 6,83 | 75,4 | 3084 |
| 777 | abliterado | 6,34 | 69,1 | 3085 |
| 4242 | oficial | **7,11** | **89,7** | — |
| 4242 | abliterado | 5,96 | **68,5** | — |

El oficial sale **más nítido y más claro** en ambas semillas; el abliterado, más suave y más oscuro.
El detalle de alta frecuencia es idéntico (3084 vs 3085), así que no es un problema de información:
es cómo el encoder interpreta la instrucción y la imagen.

**El dato que decide: estabilidad.** El brillo del oficial se mueve **75,4 → 89,7** entre semillas
(±14); el del abliterado se queda en **69,1 → 68,5** (±0,3). Para edición con identidad, donde el
resultado se compara contra una referencia, **la exposición predecible importa más que un punto
extra de nitidez**: con el oficial hay que corregir la exposición semilla a semilla.

**Recomendación: mantener el abliterado** para Identity Edit. No por el filtrado (irrelevante aquí:
la instrucción es descriptiva), sino porque produce resultados más consistentes. El oficial sería
preferible si se busca el máximo microcontraste y se acepta ajustar exposición a mano.

#### Qué opciones hay y qué se descartó

Opciones oficiales: `bf16` (8,9 GB), `fp8_scaled` (5,2 GB), `int8_convrot` (4,9 GB). La comunidad
reporta **poca diferencia de velocidad** entre cuantizaciones; cambia la comprensión del prompt. En
8 GB, `bf16` queda descartado si el TE va a GPU.

**Descartado como palanca de velocidad:** `krea2_moodboard_native_vision = false` (splice-only, sin
DeepStack ni mrope de 3 ejes). Medido con el TE en CPU, misma semilla y ajustes: **66,8 s vs 60,9 s**
a `grounding_px` 384, e indistinguible (136,4 vs 135,4 s) a 768. El path nativo de visión **no** es un
cuello de botella evitable: es un eje de calidad, no de rendimiento; se deja en `true`. (Medición
previa a P1, así que los absolutos están en la banda de 60–135 s; la comparación es válida por ser el
mismo perfil en ambas ramas.)

Detalle de configuración: los logs muestran un auto-attach (`qwen3-vl-4b-heretic_int8`,
`qwen_image_vae`) que después se sobrescribe con lo del perfil. Conviene dejar explícito solo un
origen de verdad — y con esta comparación, el elegido importa (ver P18).

> Pendiente: `qwen3vl_4b_int8_convrot` (mismo formato INT8, 4,5 GiB). No se probó porque el tiempo
> es idéntico y la comparación interesante era de pesos, no de cuantización. El riesgo real de
> ese fichero es otro: el auto-attach de `sd-forge-krea2` lo puede elegir por orden alfabético
> (ver P18).

Opciones oficiales: `bf16` (8,9 GB), `fp8_scaled` (5,2 GB), `int8_convrot` (4,9 GB). La comunidad
reporta **poca diferencia de velocidad** entre ellas; lo que cambia es comprensión del prompt.
En 8 GB, `bf16` queda descartado si el TE va a GPU.

Crítico aparte: el TE actual es una **abliteración** (Heretic/Huihui). Para el path de visión eso
puede degradar la lectura de la imagen de referencia. Comparar contra el oficial `fp8_scaled` con el
mismo prompt y semilla es un análisis barato y potencialmente revelador.

Detalle de configuración: los logs muestran un auto-attach (`qwen3-vl-4b-heretic_int8`,
`qwen_image_vae`) que después se sobrescribe con lo del perfil. Conviene dejar explícito solo un
origen de verdad.

**Descartado como palanca de velocidad:** `krea2_moodboard_native_vision = false` (splice-only, sin
DeepStack ni mrope de 3 ejes). Medido con el TE en CPU, misma semilla y ajustes: **66,8 s vs 60,9 s**
a `grounding_px` 384, e indistinguible (136,4 vs 135,4 s) a 768. El path nativo de visión **no** es un
cuello de botella evitable: es un eje de calidad, no de rendimiento; se deja en `true`. (Medición
previa a P1, así que los absolutos están en la banda de 60–135 s; la comparación es válida por ser el
mismo perfil en ambas ramas, pero conviene re-verificarla con `TE=gpu` si se quiere cerrar del todo.)

### P6 — VAE: rendimiento y acabado (medido)

**Velocidad: idéntica.** Los tres VAEs de 485 MB (y el de 243 MB) van igual con la receta final:
24,0 s / 24,5 s / 24,8 s a 640 px. La primera medición dio 33,7 s para `krea2RealVae`, pero era
**efecto de orden** (primer kernel JIT tras cambiar de VAE): repetida en caliente, 24,0 s. Con 4
pasos el VAE no es palanca de rendimiento.

**Acabado: diferencias medibles** (832×832, misma semilla y prompt):

| VAE | Detalle (HF) | Nitidez | Brillo | Sombras <25 | L1 vs Real |
|---|---|---|---|---|---|
| **krea2RealVae_v10** | **3086** | 6,33 | 68,8 | 30,3 % | — |
| qwen_image_vae | 2570 | 6,28 | 70,3 | 29,3 % | 2,39 |
| **Krea2-HD-vae** | 2854 | **6,68** | 64,3 | **36,5 %** | 6,28 |
| Wan2.1_VAE | 2695 | 6,40 | 70,1 | 29,2 % | 2,35 |

Se confirma el consenso de la comunidad, ahora con números:

- **`krea2RealVae_v10` conserva más detalle fino** (3086, un 20 % más que `qwen_image_vae`) y
  mantiene las sombras abiertas. Es el mejor punto de partida para edición de identidad, donde
  piel y textura importan.
- **`Krea2-HD-vae` da más nitidez percibida (6,68)** y más contraste, pero **aplasta sombras**: un
  36,5 % de píxeles por debajo de 25 frente al 30,3 % del Real. Recuperar ese detalle en post es
  posible pero ya se perdió.
- **`qwen_image_vae` y `Wan2.1_VAE` son casi idénticos entre sí** (L1 2,39 vs 2,35 contra Real) y
  los más planos. El Wan es el más pequeño (243 MB) y una alternativa razonable si el espacio
  aprieta, sin ganancia visible.

**Veredicto: es una decisión estética, no técnica.** Se mantiene `krea2RealVae_v10` como defecto
y `Krea2-HD-vae` queda como opción para piezas donde se busque punch y no importen las sombras.

#### Bug de imagen encontrado al medirlo (arreglado)

La primera vez que se intentó cambiar de VAE, toda generación con VAE tiled fallaba con:

```
subprocess.CalledProcessError: ... gcc ... driver.c ...
/usr/local/lib/python3.13/dist-packages/triton/backends/nvidia/driver.c:9:10:
    fatal error: Python.h: No such file or directory
```

La etapa `runtime` del `Dockerfile` instalaba **`gcc` y `g++`** con el comentario "Triton (flash/sage
JIT) necesita un C compiler en runtime", pero **no `python3.13-dev`**. Sin las cabeceras, Triton no
puede compilar su shim `driver.c` y el primer kernel JIT revienta: el compilador estaba, pero era
inútil. Afecta a cualquier ruta que compile en runtime (VAE tiled, Sage/Triton, compile).

**Arreglado en `Dockerfile` y `Dockerfile.cuda12`** (etapa `runtime`, todos los perfiles incluido
slim, que comparte Dockerfile). En el contenedor vivo se instaló a mano para poder terminar las
mediciones; el fix real entra con el próximo `make build`.

### P7 — LoRA de edición: v1.2 vs V2 (medido)

**V2 funciona con nuestro patch, sin la extensión.** El guardián del benchmark confirma que se
aplica (`Lora hashes: krea2_identity_edit_v2_r64`) y el edit se arma. No hace falta migrar a Neo
2.31 ni al path nativo `krea2_do_reference`.

**Velocidad** (640×640, 8 pasos, grounding 768):

| LoRA | `ref_boost` | Mediana | Pico VRAM |
|---|---|---|---|
| v1.2 r64 | 4 | 35,0 s | 3,0 GB |
| **V2 r64** | 4 | **33,1 s** | 3,2 GB |
| V2 r64 | 3 | 30,7 s | 4,3 GB |
| V2 r64 | 2 | 30,5 s | 4,3 GB |

**Calidad** (832×832, misma fuente y semilla):

| Semilla | LoRA | Detalle (HF) | Nitidez | Brillo | L1 entre ambos |
|---|---|---|---|---|---|
| 777 | v1.2 | 3636 | 7,29 | 69,9 | — |
| 777 | **V2** | 2995 | 6,23 | 67,1 | 4,9 |
| 1234 | v1.2 | 3470 | 6,42 | 64,0 | — |
| 1234 | **V2** | 2962 | 5,94 | 65,5 | 5,9 |

**Barrido de `ref_boost` con V2** (semilla 777, misma fuente):

| boost | Detalle (HF) | Nitidez | Brillo |
|---|---|---|---|
| 1 | 2955 | 6,32 | 69,7 |
| 2 | 2967 | 6,23 | 67,6 |
| 3 | 2977 | 6,25 | 67,1 |
| 4 | 3009 | 6,26 | 67,1 |

Lecturas:

- **V2 es más limpia y estable**: menos "crujido" de alta frecuencia (−14…−17 %) y sobre todo
  **exposición estable en todo el rango de boost**. Donde v1.2 se destrozaba con boost 2 (~130 de
  brillo), V2 se queda en 67–70. Confirma la hipótesis de P8: la receta de boost 2 **pertenece a
  V2**; con v1.2 no aplica.
- **El aspecto es más "fotográfico" y menos "render"**: piel y tela más suaves, menos microcontraste
  artificial. A cambio, quien busque ese grano duro de v1.2 lo va a notar como pérdida.
- **`ref_boost` importa poco en V2** (HF 2955→3009, brillo 69,7→67,1): identidad y detalle se
  mantienen. El boost es un dial suave, no un salvavidas. **2 es el valor del autor y es estable**;
  4 no aporta y cuesta ~2,5 s.
- **Coste:** V2 con boost 2 es **30,5 s vs 35,0 s de v1.2** → **1,15×** más rápido, con VRAM
  similar (4,3 vs 3,0 GB; ambos sobrados en 8 GB).

**Recomendación:** migrar a **V2 r64 + `ref_boost` 2**. Es más rápido, más estable y su receta de
entrenamiento coincide con los controles que ya tenemos.

### Anexo de la receta final (verificado)

```
Modelo:      Krea/krea2_krea2Int4Convrot_v10Turbo  (INT4 ConvRot, 6,4 GB)
Text encoder: Huihui-Qwen3-VL-4B-abliterated-fp8_scaled  (TE en GPU)
VAE:         krea2RealVae_v10
Prompt:      <lora:krea2_identity_edit_v2_r64:1> <lora:krea2_turbo_4step_rank_64_lora:1> <instrucción>
Pasos / CFG: 4 / 1.0      Sampler: Euler / Simple
grounding_px: 768         AR: fit source to output (v1.2)
ref_boost:   2            Resolución: ≤1 MP (1280 px es inestable en 8 GB)
```

Resultado: **19,9 s**, 5,4 GB de pico, 0 OOM, calidad igual o mejor que el baseline v1.2 de 8 pasos.

**Y se compone con la destilación 4-step: esa es la receta final.** Medido (832×832, semilla 777):

| Configuración | Pasos | Detalle (HF) | Nitidez | Brillo | L1 vs 8 pasos |
|---|---|---|---|---|---|
| V2 solo | 8 | 2977 | 6,24 | 67,7 | — |
| **V2 + distill 4-step** | **4** | **3085** | **6,34** | 69,1 | **2,01** |

Con V2 la destilación **no pierde detalle: gana** (+3,6 % de alta frecuencia, +1,6 % de nitidez), y
el brillo se queda estable. Es lo contrario de lo que pasaba con v1.2, donde el par (0,75 + boost 2)
reventaba la exposición. Los dos LoRA conviven sin interferencias.

Tiempo: **19,9 s frente a 35,0 s** del punto de partida con v1.2 (1,76×).

> **Trampa si se instala la extensión [krea2-edit-forge-neo](https://github.com/eduardoabreu81/krea2-edit-forge-neo):**
> su `compat.native_support()` busca el literal `ref_boost` en el forward del DiT. Nuestro patch
> añade `ref_boosts`, así que **detecta un falso positivo** y la extensión se desactiva sola con
> "your Forge Neo seems to have the V2 controls by itself". No los tiene: no hay `ref_position`
> (`aligned`/`beside`). Para usar placement habría que adaptar el detector, o migrar a Neo ≥2.31
> y abandonar el patch del toolkit.

> **Pendiente:** `krea2_general_edit_v2` (edición por instrucción: quitar, recolorear, restyle) y
> el placement `beside`, que es donde V2 gana más según el autor y que nuestro patch no expone.

### P8 — Destilación a 4 pasos (medido)

[Krea2-Turbo-Distill-4step-LoRA](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-4step-LoRA)
(`krea2_turbo_4step_rank_64_lora`, 418 MB), cargado junto al de edición en el prompt:

```
<lora:krea2_identity_edit_v1_2_r64:1> <lora:krea2_turbo_4step_rank_64_lora:1> ...
```

**Velocidad** (`make bench-edit`, 640×640, grounding 768, ref_boost 4, TE en GPU):

| Configuración | Pasos | Mediana | vs base |
|---|---|---|---|
| Solo edit LoRA | 8 | 35,2 s | — |
| Edit + distill 4-step | 4 | **20,7 s** | **1,70×** |

**Calidad** (832×832, misma semilla, contra 8 pasos con el mismo LoRA de edición; energía de
alta frecuencia y luminancia medias):

| Semilla | HF (detalle fino) | Brillo medio | L1 vs 8 pasos |
|---|---|---|---|
| 777 | −10,5 % | +23 % | 17,6 |
| 1234 | −3,9 % | +8 % | 6,7 |
| 4242 | −1,1 % | +7 % | 5,8 |

Lecturas:

- **El ahorro es real y es el mayor disponible tras P1**: 1,70×, sin tocar VRAM (pico 5,4 GB,
  0 OOM). A 640 px son ~14 s menos por edición.
- **Pierde detalle fino**, poco y de forma variable: entre −1 % y −10 % según semilla. El caso
  peor (777) se nota a simple vista; los otros dos son indistinguibles sin medir.
- **Sube el brillo de forma consistente (+7…+23 %)**: es el efecto más sistemático y el que
  conviene compensar (prompt negativo o ajuste posterior), no ignorar.
- Confirma lo que reporta la comunidad: recupera la mayor parte de la pérdida del 4 pasos, no
  toda. La comunidad también avisa de que **no está entrenado para Identity Edit**, solo que
  funciona junto a él.

**Variante "segura" del autor (peso 0,75 + `ref_boost` 2): descartada.** Medida a 832×832,
semilla 777, contra el mismo 8 pasos:

| Variante | Detalle (HF) | Nitidez | Brillo medio | L1 vs 8 pasos |
|---|---|---|---|---|
| 8 pasos (base) | 2570 | 5,76 | 68,0 | — |
| 4p, peso 1,0, `ref_boost` 4 | 2302 | 5,69 | 83,8 | 17,6 |
| 4p, peso 1,0, **`ref_boost` 2** | 2404 | 5,50 | **128,5** | 68,6 |
| 4p, peso 0,75, **`ref_boost` 2** | 2303 | 5,13 | **131,7** | 72,4 |

**El culpable no es la fuerza 0,75: es `ref_boost` 2.** Con boost 2 la imagen sale destrozada
(brillo ~130 contra 68, L1 68–72) **aunque la fuerza del LoRA sea 1,0**, y a ojo aparece
sobreexpuesta y lavada. Con boost 4 y peso 1,0 el resultado es el mejor de los cuatro.

Interpretación: `ref_boost` está pensado para el LoRA **V2** (donde el autor recomienda 2). Con
el **v1.2** el valor entrenado es ~4; bajarlo a 2 no "relaja" la semejanza, desequilibra el
condicionamiento y el modelo compensa subiendo la exposición. **No mezclar recetas de V2 con
pesos v1.2.**

**Receta adoptada mientras se use v1.2:** 4 pasos, CFG 1, `ref_boost` 4, distill a peso 1,0,
`grounding_px` 768. Es la única combinación medida que es a la vez rápida (1,70×) y correcta.

> Pendiente: probar el [3-step](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-3step-LoRA)
> (más rápido, menos fiel) y repetir el barrido de `ref_boost` (2/3/4/5) con **V2**, donde la
> recomendación del autor sí aplica.

### P9 / P9b — `ref_boost` y el bias denso: analizado, se queda como está

**Coste en tiempo: cero con la receta actual.** Mismos 4 pasos y misma semilla:

| `ref_boost` | Mediana |
|---|---|
| 2 | 23,88 s |
| 4 | 23,70 s |

Indistinguible. Los ~10 % que se midieron al principio eran de otra configuración (8 pasos, v1.2,
otra longitud de secuencia). Con 4 pasos el bias no es el cuello.

**Coste en memoria: real y creciente.** El bias se crea **una vez** (`_ref_attn_bias`, línea 50) y
se comparte en las 28 capas, pero es **denso `[1,1,L,L]`** aunque solo el 16–22 % sean valores
distintos de cero:

| Resolución | L | Bias denso | % no-cero |
|---|---|---|---|
| 640 | 3.968 | 30 MB | 16,3 % |
| 832 | 6.176 | 73 MB | 19,2 % |
| 1024 | 8.960 | 153 MB | 20,9 % |
| 1280 | 13.568 | **351 MB** | 22,2 % |
| 1536 | 19.200 | **703 MB** | — |

**Se evaluaron tres alternativas y ninguna es viable:**

1. **Sesgo por columna `[1,1,1,L]`** — el kernel lo acepta y es 5,3× más rápido (12,0 ms vs 64,1 ms
   a L=8.960, ver microbenchmark abajo), **pero es incorrecto**: el bias es un **bloque**
   (filas target × columnas ref), no una columna. Aplicarlo por columna lo extiende a las filas de
   la referencia, que no deben tocarse. Verificado numéricamente: max|diff| 0,063, y el efecto
   del bias es del mismo orden (0,004 de media), así que no es ruido.
2. **`expand` con stride 0** — el kernel lo materializa igual (156 MB) y produce el mismo error que
   la opción 1.
3. **Mover el sesgo al score y prescindir de la máscara** — el bias entra en un **softmax global**,
   así que no es separable por bloques; hacerlo exigiría materializar los scores
   `[1,heads,L,L]`, que a L=8.960 son **3,7 GB**. Peor que el bias por 24×.

**Veredicto: el bias denso es la opción eficiente, no un descuido.** Optimizarlo exigiría un kernel
de atención que acepte un sesgo por bloque sin materializarlo (tipo `einsum` con paddings o el
`FlexAttention` de PyTorch con `BlockMask`), que es trabajo de backend de Comfy-Kitchen, no de este
patch. **No se toca.**

> Microbenchmark que motivó el análisis (bf16, 24 heads, d=128, L=8.960, CK INT8):
> sin máscara 18,7 ms / 341 MiB · `[1,1,1,L]` 12,0 ms / 342 MiB · densa `[1,1,L,L]` 64,1 ms /
> 496 MiB. La forma de la máscara **sí** importa al kernel; lo que no hay es una forma correcta
> más barata que la densa para este sesgo.

#### Nota sobre el techo de resolución

Los 351 MB a 1280 px y 703 MB a 1536 px explican por qué el OOM aparece antes de lo que sugeriría
el tamaño del modelo: el bias crece con L². Es un candidato claro a mejora futura **si** se toca el
backend, no a un parche local.

### P10 — Backend de atención

CK INT8 gana a SDPA en todas las longitudes medidas, también con máscara (4,0 vs 12,2 ms a L=3.600).
**No hay que cambiar nada aquí.** El microbenchmark de P9b confirma además que CK respeta la
forma de la máscara: con `[1,1,1,L]` baja a 12,0 ms y con la densa sube a 64,1 ms a L=8.960, así
que el coste viene del **tamaño** de la máscara y no del kernel.

Sage está descartado para esta GPU (su path `auto` elige FP8, inestable en sm89; ver
`docs/attention-backends_09-10-2026.md`).

### P11 / P12 — Caché de pasos y SPEED (medido: SPEED gana)

**SPEED (spectral progressive diffusion) funciona con Krea 2 y es la segunda palanca más grande.**
Port de [aoleg/ComfyUI-SPEED](https://github.com/aoleg/ComfyUI-SPEED) (rama `main`, ya con Krea 2):
corre los primeros pasos a media resolución latente y expande espectralmente al final.

| Caso | Sin SPEED | Con SPEED | Ganancia |
|---|---|---|---|
| 832×832, 4 pasos, semilla 777 | 35,2 s | **21,0 s** | **1,67×** |
| 1024×1024, 4 pasos, semilla 5150 | 51,9 s | **27,1 s** | **1,91×** |

Calidad: prácticamente idéntica. A igual semilla, L1 de **2,1** (832 px) y **1,7** (1024 px),
detalle −5 % y brillo −1,4. La ganancia **crece con la resolución**, como era de esperar: el
75 % de los pasos corre a la mitad de lado (4× menos píxeles).

**Cómo hay que configurarlo (importante, no es plug-and-play):**

```python
# siempreon_scripts -> "speed (spectral progressive diffusion)"
[True, "dct", "manual", "krea-2", "0.5,1.0", True, 0.05, 1.0,
 "0.55",          # manual_sigmas: umbral de transición
 203.615097, 1.915461, -1, False,
 False, 4.5, "turbo", True, "seed"]
```

Dos trampas encontradas por el camino, ambas silenciosas:

1. **`mode="neo_shift"` no vale**: el código solo acepta `delta_optimal` o `manual`; con
   `neo_shift` cae en `ValueError` o en un plan sin transiciones.
2. **El umbral hay que calibrarlo, o no hace nada.** Con `620` (mi primer intento) el log dice
   `schedule starts at sigma 1.0000, at or below the transition's sigma 620 → dropping it` y
   `transitions: none`: la imagen sale **bit-idéntica** (L1 = 0,0) y **más lenta**, sin ningún
   error visible. El umbral tiene que caer **dentro** del schedule (sigma 1,0 → 0,40 con 4 pasos);
   con `0.55` se queda en `3/4 steps coarse (75 %), 1 at full resolution`.

**Caché de pasos (TeaCache / FirstBlockCache / Spectrum):** siguen **sin soporte declarado de
Krea 2**. `sd_forge_spectrum` está horneado y expuesto por API, pero su lista de modelos es
SD1.5/2/XL/Flux/SD3/Chroma. En ComfyUI existe `ComfyUI-Krea2-Accel` (1,3–2×) que no tiene
equivalente en Forge. **No se prueba más**: SPEED ya da más ganancia (1,7–1,9×) con una base
espectral y sin caché aproximada.

> **Riesgo a valorar:** SPEED **cambia la composición** (así lo reporta la comunidad y así lo
> confirma que a distinta semilla la imagen difiere algo más). Para edición con identidad, donde
> la instrucción manda, es aceptable; para re-generar una imagen concreta, no.

#### Scheduler: `beta` y `Karras` (gratis)

Con la receta final, mismos 4 pasos y misma semilla:

| Scheduler | Tiempo | Detalle (HF) | Brillo | L1 vs Simple |
|---|---|---|---|---|
| Simple | 36,1 s | 3086 | 68,8 | — |
| **beta** | 35,2 s | 3070 | 67,9 | 1,49 |
| **Karras** | 35,2 s | **4167** | 79,8 | 21,06 |
| Turbo | 35,2 s | 2935 | **112,3** | 83,12 |

- **`beta` ≈ `simple`** (L1 1,49): indistinguible, pero es la recomendación de quien más ha
  probado Krea 2 en Forge. Gratis adoptarlo.
- **`Karras` sube el detalle un 35 %** y visualmente sale más nítido sin artefactos, a cambio de
  +11 de brillo. Es la opción si se busca textura; hay que vigilar la exposición.
- **`Turbo` revienta la imagen** (brillo 112, L1 83): descartado.
- **`Euler a` no funciona con Krea 2**: `AssertionError` en el denoiser CFG de Forge. Descartado.

`TeaCache`/`FirstBlockCache` (`sd-forge-blockcache`, `sd_forge_spectrum`) **no declaran soporte de
Krea 2**: están listados para SD1.5/2/XL/Flux/SD3/Chroma. En ComfyUI existe un port específico
(`ComfyUI-Krea2-Accel`) con 1,3–2× que no tiene equivalente en Forge. Dado que SPEED ya da
1,7–1,9× con base espectral (no caché aproximada), **no se persigue**.

### P13 — Pre-encode del source

Upstream documenta que si el VAE-encode del source ocurre en el primer paso de sampling, el gestor
de memoria puede expulsar parte del UNet y **nunca recargarlo** → cada paso pasa a streamear pesos
([pixel path y VRAM](https://github.com/lbouaraba/comfyui-krea2edit)). En nuestro patch, `arm_edit`
hace el `vae.encode` en `process()` (antes de sampling), así que **ya estamos en el camino bueno**.
Hay que verificar que sigue siendo así cuando se active el pixel-path "fit" con hires.

### P14 — Auto face-ref prep

Genera **dos pasadas anidadas completas** (~×3 el coste de un edit) antes de la pasada final. Tiene
caché en disco por hash de la imagen, así que el coste es único por referencia. Correcto, pero hay
que evitar activarlo en pruebas de rendimiento.

### P15 — Caché de condicionamiento

El script hace `bust_cond_caches(p)` **siempre** que el accordion está activo, para forzar recálculo
con las fuentes armadas. Es correcto, pero implica que prompt matrix / xyz / reprocesos con la misma
referencia pagan el TE completo cada vez. Analizar si se puede cachear por `(hash imagen, grounding,
prompt)`.

### P16 — Varianza y thrashing

Corridas 10–30× más lentas de lo normal sin causa aparente. Hipótesis: descargas en background,
paginación de RAM (32 GB con 19 GB en caché) y el comportamiento del gestor entre renders. Antes de
sacar conclusiones de cualquier benchmark hay que: cerrar CivitAI Browser/aria2, fijar semilla y
resolución, y descartar los primeros N lanzamientos.

### P17 — Benchmark reproducible (hecho)

`scripts/bench_edit.py` + `make bench-edit` / `make bench-edit-sweep` / `make test-bench-edit`
(39 tests), en la línea de `bench_attn`/`bench_offload`.

**El problema que evita:** en el primer intento median las corridas **sin el tag `<lora:...>`** en
el prompt. Forge no falla, la extensión arma la referencia igual… y el modelo edita **sin LoRA**,
que sale ~3× más rápido. Mis primeras cifras de P1 eran de ese camino.

**Cómo lo evita:** tras cada generación lee el `info` de la respuesta y exige dos pruebas:
`Krea2 Edit: "N source(s)"` (la referencia llegó al DiT) y `Lora hashes: "<nombre>: <sha>"`
(el LoRA se aplicó). Si falta cualquiera, **aborta** en vez de apuntar un número falso.

Descartado el log del contenedor como fuente: Forge solo emite `[LORA] Loaded` la primera vez
que aplica un LoRA, así que con el modelo ya residente no aparece y da falsos negativos. Se
mantiene como aviso informativo, nunca como verificación.

**Segundo fallo que evita (encontrado al usarlo):** el benchmark medía el checkpoint que hubiera
cargado en la UI. Con Flux Klein activo, los LoRA de Krea 2 dan `[LORA] Mismatch`, el edit no se
arma y se perdían minutos en un error engañoso. Ahora comprueba `sd_model_checkpoint` en
`/sdapi/v1/options` **antes de medir** y falla en ~0,2 s diciendo qué cargar. No se puede fijar
por petición: `override_settings.sd_model_checkpoint` devuelve **HTTP 500** en Neo 2.29.

También documenta el eje que más importa: la columna **TE** sale de `/cmd-flags`
(`--text-enc-device`, o `cpu` implícito por `--lowvram`), para que ninguna fila se pueda leer sin
saber dónde corrió el encoder.

### P18 — Auto-attach de `sd-forge-krea2` vs configuración manual (bug real)

Encontrado al validar el perfil: tras recrear el contenedor, la generación fallaba con

```
ModuleNotFoundError: Failed to recognize model...
  at /data/extensions/sd-forge-krea2/scripts/krea2_register.py:515  _auto_pieces_forge_loader
```

**Causa:** el checkpoint es un *bare DiT* (`txtfusion.projector` presente, sin `text_encoders.`),
así que la extensión intenta adjuntar TE+VAE sola, escaneando por nombre de fichero:

| Pieza | Filtro del scan | ¿Encaja la config actual? |
|---|---|---|
| TE | nombre con `qwen3vl` / `qwen3_vl` / `qwen3-vl` | El abliterado `Huihui-Qwen3-VL-...` **sí** encaja |
| VAE | nombre con `qwen` **o** `wan` | `krea2RealVae_v10` **NO** encaja |

El VAE custom (`Krea2-HD-vae`, `krea2RealVae_v10`) no contiene ninguna de las dos palabras, así
que el scan devuelve `None` y el loader lanza `Failed to recognize model` aunque la UI ya tuviera
la combinación correcta (checkpoint + TE + VAE) seleccionada.

**Riesgo adicional:** aunque el VAE fallara solo, el TE preferido del scan es `bf16` primero y,
si no hay, el **primero por orden alfabético** de los que encajan. En este árbol eso puede ser
`qwen3vl_4b_int8_convrot` o `qwen3-vl-4b-instruct-int4` (menor calidad) en vez del abliterado
elegido. Es decir: al recargar el modelo, **puede cambiar el TE sin avisar**.

**Mitigación:** evitar disparar el auto-attach. Funciona si los módulos van explícitos en la
selección (`forge_additional_modules`) y no se fuerza un reload con el modelo "bare". Al recrear
el contenedor la config guardada puede quedar apuntando a otro preset (`forge_preset: flux`), y
ahí es donde salta.

**Fix propuesto (no aplicado):** parchear `_find_krea2_modules()` del repo
[Stable-yogi/sd-forge-krea2](https://github.com/Stable-yogi/sd-forge-krea2) para que:
1. acepte también VAE con `krea2`/`krea` en el nombre, y
2. respete la selección explícita de `forge_additional_modules` antes de decidir por nombre.

Sería un parche pequeño y en una extensión de terceros, no en el core. Anotado como candidato
para `builtin-extensions/` si el síntoma se repite.

`--fp8_e4m3fn-unet` no aporta nada con un checkpoint ya cuantizado en INT4; solo afecta a la carga de
pesos bf16/fp16. Mantenerlo oscurece qué está haciendo el perfil. Se puede retirar y medir.

---

## 5. Qué dice la comunidad (resumen)

| Tema | Consenso |
|---|---|
| Identity Edit lento | Casi siempre es presión de VRAM/offload, no el sampler. 8–12 pasos, CFG 1, ≤1,5 MP, `grounding_px` 384 si va justo |
| Checkpoint en 8 GB | INT4 ConvRot; INT8 no cabe; W4A8 recupera calidad a coste cero de VRAM |
| TE en GPU | Diferencia abismal en setups con `--lowvram`; en Comfy el TE corre en GPU por defecto, por eso no ven el problema |
| Prefill de visión | El coste de Qwen3-VL escala con (píxeles/32)² de tokens; de ahí el dial `grounding_px` |
| Modelo | Turbo para inferir, Raw para entrenar LoRAs |
| VAE | Diferencias sutiles; no merece la pena perseguir |
| Caché de pasos | Muy prometedor justo en GPUs con poca VRAM; soporte en Forge Neo sin confirmar |

---

## 6. Plan de análisis propuesto

1. **P1** — TE en GPU como configuración canónica (eje `TE=`). *Hecho.*
2. **P17** — Benchmark reproducible de Identity Edit. *Hecho* (`make bench-edit`).
3. **P8** — Distill 4 pasos con edit: **1,70×** medido. Adoptar con peso 1,0 y `ref_boost` 4;
   la variante "segura" (0,75 + boost 2) se descartó por destrozar la exposición.
4. **P3** — Fijar `grounding_px` por defecto: comparativa a 384 vs 768 **con LoRA activo** y
   revisión visual de identidad.
5. **P4** — **Descartado**: W4A8 es 28 % más lento y calidad idéntica. INT4 ConvRot se queda.
6. **P2** — **Medido**: el perfil de VRAM no cambia nada (0 %). Techo entre 1,6 y 2,3 MP
   (1280 px estable, 1536 px OOM limpio en 2 s).
7. **P7** — V2 medido; queda el `general_edit_v2` y el placement `beside` (requiere adaptar el
   detector de la extensión o migrar a Neo ≥2.31).
8. **P9b** — **Analizado y descartado**: el bias denso es la opción eficiente. Las tres alternativas
   fallan (columna = incorrecta, expand = materializa igual, score = 3,7 GB). Solo mejorable en el
   kernel de Comfy-Kitchen.
9. **P12** — **Medido**: SPEED da **1,67× a 832 px y 1,91× a 1024 px** con calidad casi idéntica.
   Requiere configuración manual (`mode=manual` + umbral calibrado) o no hace nada.
10. **P11** — **Descartado**: la caché de pasos no soporta Krea 2 en Forge; SPEED ya cubre más.
10. **P18** — Auto-attach de `sd-forge-krea2`: parchear `_find_krea2_modules()` para aceptar VAE
    `krea2`/`krea` y respetar la selección explícita. **Prioridad alta**: bloquea el arranque.
11. **P5** — **Medido**: el TE oficial es más nítido pero inestable en exposición (±14 de brillo entre
    semillas vs ±0,3 del abliterado). Se mantiene el abliterado para edición.
12. **P6** — **Medido**: los VAE no cambian el tiempo (24,0 s todos). Real conserva más detalle,
    HD aplasta sombras. Decisión estética; se queda Real.

---

## Anexo: reproducción

```sh
# Krea 2 con Identity Edit / Moodboard: TE en GPU sobre el perfil 8 GB
make krea2            # ATTN=ck + TE=gpu

# Benchmark del edit (verifica que el LoRA se aplicó en el info de la respuesta)
make bench-edit
make bench-edit BENCH_EDIT_ARGS="--grounding-px 384 --runs 3"
make bench-edit-sweep # TE en CPU vs GPU sobre el mismo perfil

# Comprobación de dónde corre el TE
docker exec forge-neo bash -c 'tr "\0" " " < /proc/1/cmdline' | grep -o "text-enc-device [a-z]*"
```

Los tiempos de este documento salen de `make bench-edit` (2 corridas tras calentamiento,
semilla 777, misma imagen fuente). El anexado va a
[`docs/bench-edit_21-11-2026.md`](bench-edit_21-11-2026.md).
