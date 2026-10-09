# Última modificación: 2026-10-10

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
instalado), con xformers como fallback. `--use-ck-attention` solo se usa en el preset `make wan`
(o con `make run ATTN=ck`).

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
| **CK INT8** (`--use-ck-attention`) | flag runtime, **ya en la imagen** | nulo | secuencias largas (vídeo / ≥1280 px) | **Ganador medido.** Krea2 1280px: 25,58 → 18,64s (**−27 %**); 1024px: 14,55 → 12,89s (−11 %). |
| **Sparse Attention** (`sol_attn`) | script en UI, ya horneado | nulo | Wan / ≥1280 px | Útil en vídeo; **sustituye** al backend activo (no se suma). Exige `dim_head=128`, bf16/fp16. |
| **SageAttention 2.2.0** | instalar en build + `--sage-function` | medio (wheel cp313/cu130) | secuencias largas | **Experimental.** En sm89, `fp16_cuda` da 20,71s a 1280px (mejor que flash, peor que CK); `fp16_triton` es **un lastre** (40,2s). No bit-exacto. |
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
path FP16 es estable. Por eso `make run ATTN=sage` arranca con **`--sage-function fp16_cuda`** de forma
explícita: nunca dejes `auto` en sm89.

Alternativa si el kernel CUDA diera problemas: `make run ATTN=sage-triton` usa
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
   - Ejes de arranque: `ATTN=flash|ck|sage|sage-triton` y `VRAM=auto|8gb|normal|high` (targets `make run`).
   - `make wan` arranca con `ATTN=ck` (CK INT8, sin build); `make build-sage` hornea Sage.
   - `make run ATTN=sage` arranca con la imagen `:sage` + `--sage-function fp16_cuda`.
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
   - `make run VRAM=8gb ATTN=flash` (base, FlashAttention)
   - `make run VRAM=8gb ATTN=ck` (CK INT8)
   - `make build-sage && make run VRAM=8gb ATTN=sage` (Sage fp16_cuda)
   - `make run VRAM=8gb ATTN=sage-triton` (Sage fp16_triton, si el CUDA diera problemas)
3. Casos a medir, por orden de interés: **Wan 2.2 (vídeo)** → **Krea 2 a 1280/1536px** → (opcional) 768px
   para confirmar que "no hay diferencia".
4. Compara también **calidad**, no solo velocidad: a igual semilla la imagen cambiará si el backend
   cuantiza (CK INT8 y Sage). Revisa sobre todo detalle fino y texto.
5. Si usas Sparse Attention desde la UI, **no** lo combines con `make wan` ni con `ATTN=ck` / `ATTN=sage`: el script
   instala un `optimized_attention_override` que sustituye el backend activo.

## Resultados medidos (2026-10-10)

Banco: `scripts/bench_attn.py` vía `make bench-attn*`, semilla fija 12345, 3 corridas tras 1 de
calentamiento, Krea 2 Int4 ConvRot v10 Turbo, RTX 4060 8 GB, perfil `8gb` (`--lowvram --cuda-stream`).
Tabla completa en [`docs/bench-attn_09-10-2026.md`](bench-attn_09-10-2026.md).

| Backend | 1024x1024 (mediana) | 1280x1280 (mediana) | vs flash @1280 | Pico VRAM |
|---|---|---|---|---|
| `flash` | 14,55 s | 25,58 s | — | 3,9–4,1 GB |
| `ck-int8` | **12,89 s** | **18,64 s** | **−27 %** | 3,9–4,1 GB |
| `sage-fp16_cuda` | 13,78 s | 20,71 s | −19 % | 3,8–4,1 GB |
| `sage-fp16_triton` | 13,50 s | 40,21 s | +57 % | 4,1 GB |

Conclusiones:

- **CK INT8 gana en ambas resoluciones**, y su ventaja **crece con la resolución** (−11 % a 1024 px,
  −27 % a 1280 px): confirma que el coste de atención pesa más a más tokens. Es el backend a usar a
  ≥1280 px, y no exige imagen extra.
- **Sage `fp16_cuda`** queda en medio; solo tendría sentido frente a CK en secuencias mucho más largas
  (vídeo), caso que este banco **no cubre** (es `txt2img`).
- **Sage `fp16_triton` es claramente peor que flash**, incluso a 1024 px: el kernel f16 de Sage en
  sm89 no está optimizado para esta carga. Descartado.
- **0 OOM** en todos los casos. El pico de VRAM es acumulado desde el arranque del contenedor, no por
  corrida (por eso las cifras se solapan).

### Bug de `/sdapi/v1/cmd-flags` (arreglado)

Forge Neo construye `FlagsModel` (en `modules/api/models.py`) tipando cada flag como `type(default)`.
Los flags numéricos con `default=None` —`--port`, `--reserve-vram`, `--cuda-stream`— quedan como
`str | None`, y al devolver un valor numérico FastAPI lanza `ResponseValidationError` → **HTTP 500**.
Afecta a cualquier consumidor del endpoint (nosotros, el propio bench). Se corrige con
`patches/cmd-flags-numeric-types.patch`, que relaja a `Any` los campos que quedan en ese tipo. Con él
`/cmd-flags` responde 200 y el benchmark ya no necesita degradar (el *fallback* best-effort en
`bench_attn.py` se mantiene por robustez).

### Bug de build de Sage (arreglado)

`make build-sage` **nunca instaló SageAttention**: descargaba el wheel como `/tmp/sage.whl` y pip lo
rechaza (`Invalid wheel filename (wrong number of parts): 'sage'`). Por eso la imagen `:sage` existía
pero no contenía Sage y jamás se había medido. El `Dockerfile` ahora descarga con el **basename de la
URL URL-decodificado** (la URL trae `%2B`) y verifica el SHA256; `Dockerfile.cuda12` ya usaba la URL
directa (funcionaba). Tras el arreglo, `forge-neo:sage` instala `sageattn_qk_int8_pv_fp16_cuda/_triton`.

## Riesgos y cosas a evitar

- **No habilitar Sage en la imagen CUDA 12** sin probar: el wheel `+cu12` se instala pero no está
  validado en este stack (sm89 + triton 3.7.1).
- **No dejar Sage en `auto`** en esta GPU (elegiría FP8, inestable).
- **`--sage-function` sin Sage instalado no hace nada** (el flag se ignora y sigue FlashAttention).
- El build con Sage **descarga un wheel de GitHub** en tiempo de build; si el release desaparece, el
  build falla. El SHA256 está fijado como ARG para detectar cambios.
