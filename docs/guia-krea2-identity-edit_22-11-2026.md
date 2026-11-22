# Última modificación: 2026-11-22

# Guía: Krea 2 + Identity Edit en 8 GB

Guía práctica para editar una foto manteniendo la identidad de la persona, con la configuración
óptima medida en una RTX 4060 de 8 GB. Para el razonamiento y las mediciones completas, ver
[`aceleracion-krea2-identity-edit_21-11-2026.md`](aceleracion-krea2-identity-edit_21-11-2026.md).

**Resumen en una línea:** `make krea2`, cargar el checkpoint Krea 2 INT4, el LoRA de edición V2 y
el de 4 pasos, poner **4 pasos / CFG 1 / Euler / beta / ref_boost 2**, y editar a **1 MP o menos**.

---

## 1. Antes de arrancar: qué tiene que existir

Comprueba que están en `DATA_PATH/Models`:

| Fichero | Carpeta | Para qué |
|---|---|---|
| `krea2_krea2Int4Convrot_v10Turbo.safetensors` | `StableDiffusion/Krea/` | El modelo (6,4 GB). INT4 ConvRot es el único que cabe entero en 8 GB |
| `Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors` | `TextEncoders/` | El encoder de texto **con visión** (4,9 GB). Sin visión no hay edición |
| `krea2RealVae_v10.safetensors` | `VAE/` | El VAE. Conserva más detalle que el resto |
| `krea2_identity_edit_v2_r64.safetensors` | `Lora/Krea/` | El LoRA que hace la edición (0,46 GB) |
| `krea2_turbo_4step_rank_64_lora.safetensors` | `Lora/Krea/` | El LoRA que baja de 8 a 4 pasos (0,42 GB) |

Si falta alguno, `make krea2-recipe` te dice qué tienes cargado ahora mismo.

---

## 2. Arrancar

```sh
make krea2
```

Eso pone los tres ajustes de arranque que importan:

```
Atención: ck --use-ck-attention
TE:       gpu (--text-enc-device cuda)
```

**El del medio es el importante.** Con el perfil de 8 GB (`--lowvram`) Forge pone el text encoder
en CPU por defecto, y con una imagen de referencia eso multiplica el tiempo por cuatro. `make krea2`
lo fuerza a GPU.

---

## 3. Configurar la interfaz

### 3.1 Modelo y módulos

En la parte superior, **UI Preset → `krea`**. Después:

| Campo | Valor |
|---|---|
| Checkpoint | `Krea/krea2_krea2Int4Convrot_v10Turbo.safetensors` |
| Text Encoder | `Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors` |
| VAE | `krea2RealVae_v10.safetensors` |

> **Ojo:** si el Text Encoder y el VAE aparecen vacíos o con otro modelo, cárgalos a mano. No dejes
> que se auto-elija: la extensión de Krea 2 reparte módulos por nombre de fichero y puede poner un
> encoder de menor calidad sin avisar.

### 3.2 LoRA en el prompt

Los dos LoRA van **en el prompt**, no solo en el desplegable:

```
<lora:krea2_identity_edit_v2_r64:1> <lora:krea2_turbo_4step_rank_64_lora:1> <tu instrucción>
```

Es lo más fácil de olvidar y lo más caro: **sin el tag del LoRA de edición, Forge no falla**, arma
la referencia igual y el modelo edita sin LoRA. Sale ~3× más rápido y con mucha peor identidad.
Si dudas, comprueba en el infotext de la imagen generada que aparece `Lora hashes:` con los dos
nombres y `Krea2 Edit: "1 source(s)"`.

### 3.3 Parámetros de muestreo

| Ajuste | Valor | Por qué |
|---|---|---|
| Pasos | **4** | El LoRA de destilación está entrenado para 4 |
| CFG | **1.0** | El Turbo no usa guía; con CFG 1 el negativo se ignora |
| Sampler | **Euler** | `Euler a` da error con Krea 2 |
| Scheduler | **beta** | Equivalente a `simple` en lo medido, pero es la recomendación de la comunidad |
| Tamaño | **hasta ~1 MP** | 832×832 va holgado; 1280 px entra; 1536 px se queda sin VRAM |

> Si quieres **más textura**, cambia el scheduler a **Karras**: sube el detalle fino un 35 %. A
> cambio, la imagen sale ~11 puntos más clara, así que vigila la exposición. No uses `Turbo`: sale
> quemada.

### 3.4 Accordion *Krea2 Identity Edit*

| Campo | Valor | Nota |
|---|---|---|
| Enable | ✅ | |
| Source image | tu foto | Una sola imagen para editar a una persona |
| 2nd reference | vacío | Solo para composiciones persona + escena |
| `grounding_px` | **768** | Cuánto detalle ve el encoder. Menos = obedece más la instrucción; más = más parecido. 768 es el entrenado |
| Aspect ratio | **fit source to output (v1.2)** | Redimensiona en píxeles: no emborrona y respeta tu encuadre |
| `ref_boost` | **2** | Cuánto tira hacia la referencia. 2 es lo entrenado para V2. **No bajes de 2 ni subas a 4 sin probar**: 2 es el punto estable |
| `ref_boost (scene)` | 1.0 | Solo aplica con dos imágenes |
| Auto face-ref prep | ❌ **desactivado** | Lanza dos generaciones extra antes de la tuya: triplica el tiempo. Úsalo solo si la foto de origen tiene sombreros, gafas o manos tapando la cara |

---

## 4. Qué esperar

Con la receta de arriba, en una RTX 4060 de 8 GB:

| Resolución | Tiempo | VRAM |
|---|---|---|
| 640×640 | ~20 s | 5,4 GB |
| 832×832 | ~30 s | 5,5 GB |
| 1024×1024 | ~44 s | 5,6 GB |
| 1280×1280 | ~73 s | 6,3 GB |
| 1536×1536 | **OOM** | — |

Si añades **SPEED** (sección 6), esos tiempos bajan a **~13 s / ~21 s / ~27 s** respectivamente en
los tres primeros.

Punto de partida, para comparar: la configuración por defecto (encoder en CPU, LoRA v1.2, 8 pasos)
tardaba **145 s** en la misma edición a 640 px.

---

## 5. Escribir la instrucción

La instrucción es tu prompt. Consejos que salen de la documentación del LoRA:

- **Di cuánto se ve de la persona**: "full body", "sitting cross-legged", "de rodillas". Si no lo
  dices, tiende a copiar el encuadre de la foto original.
- **Describe la ropa nueva y lo que desaparece**: "sin abrigo", "lleva una camiseta blanca". El
  modelo se apoya mucho en la ropa de la foto.
- **Describe la luz por dónde cae**: "la luz de la ventana le ilumina el lado izquierdo de la cara".
- **Una tarea por LoRA**: identidad con V2 naranja, edición por instrucción (quitar, recolorear) con
  `general_edit_v2`. No los cargues juntos a fuerza 1.0: se rompe la imagen.

---

## 6. Opcional: SPEED, la segunda palanca

[SPEED](https://github.com/aoleg/ComfyUI-SPEED) ejecuta los primeros pasos a media resolución y
expande al final. Da **1,67× a 832 px y 1,91× a 1024 px** con calidad casi idéntica.

Está instalado en `EXTENSIONS_PATH`, en el accordion **"speed (spectral progressive diffusion)"**.

**Configuración obligatoria** (por defecto no hace nada, y encima más lento):

| Campo | Valor |
|---|---|
| Enable | ✅ |
| Transform | `dct` |
| **Mode** | **`manual`** ← `neo_shift` no es válido y no arranca |
| Model preset | `krea-2` |
| Scales | `0.5,1.0` |
| **Manual sigmas** | **`0.55`** ← el umbral crítico |

> **Si la imagen sale idéntica y tarda igual, SPEED no se aplicó.** Mira el log: si dice
> `transitions: none`, el umbral está fuera del schedule. Con `0.55` debe decir
> `schedule split: 3/4 steps coarse (75%), 1 at full resolution`.

**Coste:** cambia ligeramente la composición. Para editar sobre una foto es aceptable; para
reproducir una imagen concreta, no.

---

## 7. Problemas frecuentes

| Síntoma | Causa | Arreglo |
|---|---|---|
| La edición tarda minutos | El encoder está en CPU | Arranca con `make krea2` (fuerza `TE=gpu`) |
| Identidad muy pobre | Falta el tag `<lora:...>` en el prompt | Añádelo; comprueba `Lora hashes` en el infotext |
| `Failed to recognize model` al generar | VAE con nombre que la extensión no reconoce | Selecciona TE y VAE a mano (sección 3.1) |
| `Python.h: No such file or directory` | Imagen sin `python3.13-dev` | `make build` (arreglado en esta rama) |
| Imagen quemada / sobreexpuesta | `ref_boost` bajo con LoRA v1.2, o scheduler `Turbo` | `ref_boost` 2 **con V2**; scheduler `beta` |
| `AssertionError` al muestrear | Has elegido `Euler a` | Usa `Euler` |
| OOM a 1536 px | El bias de referencia crece con la resolución | Baja a 1280 px o menos |
| SPEED no cambia nada | `mode` no es `manual` o el umbral es alto | Ver sección 6 |

---

## 8. Chuleta

```sh
make krea2          # arranca con la configuración correcta
make krea2-recipe   # imprime la receta y qué hay cargado ahora
make bench-edit     # mide un edit con el LoRA verificado
```

```
Modelo:      Krea/krea2_krea2Int4Convrot_v10Turbo
Text encoder: Huihui-Qwen3-VL-4B-abliterated-fp8_scaled
VAE:          krea2RealVae_v10

<lora:krea2_identity_edit_v2_r64:1> <lora:krea2_turbo_4step_rank_64_lora:1> <instrucción>

4 pasos · CFG 1 · Euler · beta · 832×832
grounding_px 768 · AR fit · ref_boost 2 · prep off
```
