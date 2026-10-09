# Última modificación: 2026-10-09

# Backends de atención en Forge Neo: qué se puede configurar y qué conviene

Investigación y cambios para elegir el backend de self/cross-attention más eficiente que el actual.
Todas las versiones de esta página están **leídas del código de la imagen** (`forge-neo:latest`), no
de documentación upstream.

## Contexto

| Dato | Valor |
|---|---|
| GPU | RTX 4060, 8 GB, compute capability **8.9 (sm89, Ada)** |
| Flags de build (Dockerfile) | `--xformers --flash` (slim añade `--onnxruntime-gpu`) |
| `EXTRA_ARGS` actual | `--cuda-malloc --lowvram --fp8_e4m3fn-unet --reserve-vram 2 --pin-shared-memory --mmap-torch-files` |
| torch | **2.13.0+cu130** |
| Instalado | `flash_attn 2.8.3`, `xformers 0.0.35`, `comfy_kitchen 0.2.35`, `triton 3.7.1`, `onnxruntime-gpu 1.28.0` |
| **NO instalado** | `sageattention`, `nunchaku`, `bitsandbytes` |

Conclusión de partida: **la atención activa hoy es FlashAttention** (porque `flash_attn` está
instalado), con xformers como fallback. `--use-ck-attention` solo se usa en `make wan`.

## Cómo elige Forge el backend

En `backend/attention.py`, la selección es **por disponibilidad del paquete**, no por velocidad
medida. Orden de prioridad (el primero disponible gana):

```
Comfy-Kitchen INT8  →  Sage  →  Flash  →  xformers  →  PyTorch SDPA  →  basic
```

- **`--xformers` / `--flash` / `--sage` son flags de *build***: instalan el paquete (vía `launch.py`).
  No son de runtime. Lo que decide en runtime es qué paquete está presente.
- **`--use-ck-attention` y `--use-pytorch-cross-attention` sí son de runtime.**
- `--sage-function {auto|fp16_triton|fp16_cuda|fp8_cuda|fp8_cuda++|sageattn3}` solo aplica si Sage
  es el backend activo.

## Opciones y veredicto

| Backend | Cómo se activa | Coste de integración | Rinde cuando… | Veredicto |
|---|---|---|---|---|
| **FlashAttention 2.8.3** | ya activo | — | secuencias medias/largas | **Base actual.** Correcto en sm89. |
| **CK INT8** (`--use-ck-attention`) | flag runtime, **ya en la imagen** | nulo | secuencias largas (vídeo / ≥1280 px) | **Probar primero.** Medido en Krea2 1280px: 26,3s → 21,9s; a 768px sin diferencia. |
| **Sparse Attention** (`sol_attn`) | script en UI, ya horneado | nulo | Wan / ≥1280 px | Útil en vídeo; **sustituye** al backend activo (no se suma). Exige `dim_head=128`, bf16/fp16. |
| **SageAttention 2.2.0** | instalar en build + `--sage-function` | medio (wheel cp313/cu130) | secuencias largas | **Experimental.** En sm89 el path FP8 por defecto es **inestable**; hay que forzar FP16. |
| **PyTorch SDPA** (`--use-pytorch-cross-attention`) | flag runtime | nulo | — | Estable, sin ventaja clara frente a Flash aquí. |
| Nunchaku / bitsandbytes | `--nunchaku` / `--bnb` | alto | cuantización de pesos | Fuera de alcance (fuerza torch distinto). |

### Por qué el margen es pequeño a baja resolución

Con `--lowvram` el cuello de botella es el **trasiego de pesos RAM↔VRAM, no el cálculo**. Y el número
de tokens es `(ancho/16)×(alto/16)`: a 768px son ~2.304 tokens, donde la atención es barata. Cambiar
de kernel apenas se nota por debajo de ~1024 px. **La atención sí domina en vídeo (Wan 2.2, ~32.000
tokens) y en alta resolución**; es ahí donde tiene sentido optimizar.

### El caso SageAttention en sm89 (importante)

El wheel de Sage 2.2.0 hace, en `core.py`, este dispatch por arquitectura:

```
sm80 → fp16_cuda (fp32 accum)
sm89 → sageattn_qk_int8_pv_fp8_cuda  ...  pv_accum_dtype="fp32+fp16"   # path FP8
sm90 → fp8_cuda_sm90
sm120→ fp8_cuda
```

Es decir, **en tu GPU `auto` elige el path FP8**, que la propia comunidad reporta como inestable en
sm89 (fallos `unspecified launch failure`, NaNs en cargas reales; ver thu-ml/SageAttention#309). El
path FP16 es estable. Por eso `make sage` arranca con **`--sage-function fp16_cuda`** de forma
explícita: nunca dejes `auto` en sm89.

Alternativa si el kernel CUDA diera problemas: `make sage-triton` usa
`--sage-function fp16_triton`, que mapea a `sageattention.sageattn_qk_int8_pv_fp16_triton` (INT8 QK +
FP16 PV por Triton). Hay que **medir**: Sage en sm89 no tiene kernel f16 con acumulador f16, así que
`sageattn_qk_int8_pv_fp16_cuda` con `pv_accum_dtype="fp16"` cae a los símbolos `sm80_compile` en
tiempo de import, que pueden no estar bien resueltos en sm89. El benchmark (`make bench-attn-sage-triton`)
lo dirá en tu máquina. Nota: ambas funciones (`fp16_cuda`/`fp16_triton`) **no** habilitan el path
*puro* de Triton (`sageattn`), que en `core.py` exige `dim_head > 128`; Forge filtra por `dim_head`
antes, así que en la práctica no se alcanza.

Nota: Sage no pierde calidad de forma apreciable en uso normal (cuantiza Q/K), pero **no es
bit-exacto**: a igual semilla da una imagen distinta. No es intercambiable a mitad de un trabajo.

## Cambios implementados

1. **Dockerfile / Dockerfile.cuda12**: `ARG INSTALL_SAGE=0` (por defecto **no** cambia nada) e
   instalación del wheel precompilado de `snw35/sageattention-wheel` (Linux cp313, CUDA 13). Verificado
   que el `.so` `_qattn_sm89` enlaza limpio contra el torch 2.13 de la imagen y que el wheel **no
   declara dependencias** (no pisa torch).
   - **No se usa el flag `--sage` de `launch.py`**: instala `sageattention==2.2.0` desde PyPI, que
     **no la publica** (solo hasta 1.0.6, sdist) → fallaría el build.
   - El `strip` de la limpieza final excluye `sageattention/*.so` (strippear kernels CUDA es arriesgado).
2. **docker-compose.yml**: `image: ${FORGE_IMAGE:-forge-neo:latest}` para poder arrancar la variante
   `:sage` sin pisar `latest`.
3. **Makefile**:
   - `make ck` → alias de `make wan` (CK INT8, sin build).
   - `make build-sage` → construye `forge-neo:sage` con Sage horneado.
   - `make sage` → arranca con la imagen `:sage` + `--sage-function fp16_cuda`.
   - `make push-sage` → publica `REGISTRY_IMAGE` con tag `:sage`.

## Plan de medición (antes de fijar nada)

La mejora solo aparece en secuencias largas, así que **mide en el caso que te interesa**, no en 768px.

Atajo: `make bench-attn-sweep` automatiza base (flash) vs CK INT8 (reinicia, mide y anexa filas a
`docs/bench-attn_09-10-2026.md`). Para Sage: `make bench-attn-sage` (fp16_cuda) o
`make bench-attn-sage-triton` (fp16_triton). Para medir el backend activo a
mano: `make bench-attn BENCH_ARGS="--width 1280 --height 1280 --steps 8 --runs 3"`. El benchmark usa
**semilla fija** y anota mediana, it/s, pico de VRAM y OOM.

Pasos equivalentes si lo haces a mano:

1. Fija **misma semilla, mismo prompt, misma resolución, mismos pasos**.
2. Genera una vez con cada perfil y anota el tiempo del log (`Total progress`/IT/s) y el pico de VRAM:
   - `make lowvram` (base, FlashAttention)
   - `make ck` (CK INT8)
   - `make build-sage && make sage` (Sage fp16_cuda)
   - `make sage-triton` (Sage fp16_triton, si el CUDA diera problemas)
3. Casos a medir, por orden de interés: **Wan 2.2 (vídeo)** → **Krea 2 a 1280/1536px** → (opcional) 768px
   para confirmar que "no hay diferencia".
4. Compara también **calidad**, no solo velocidad: a igual semilla la imagen cambiará si el backend
   cuantiza (CK INT8 y Sage). Revisa sobre todo detalle fino y texto.
5. Si usas Sparse Attention desde la UI, **no** lo combines con `make ck` ni con `make sage`: el script
   instala un `optimized_attention_override` que sustituye el backend activo.

## Riesgos y cosas a evitar

- **No habilitar Sage en la imagen CUDA 12** sin probar: el wheel `+cu12` se instala pero no está
  validado en este stack (sm89 + triton 3.7.1).
- **No dejar Sage en `auto`** en esta GPU (elegiría FP8, inestable).
- **`--sage-function` sin Sage instalado no hace nada** (el flag se ignora y sigue FlashAttention).
- El build con Sage **descarga un wheel de GitHub** en tiempo de build; si el release desaparece, el
  build falla. El SHA256 está fijado como ARG para detectar cambios.
