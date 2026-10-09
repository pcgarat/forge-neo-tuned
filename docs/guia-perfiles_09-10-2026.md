# Última modificación: 2026-10-09

# Guía de perfiles y órdenes: qué usar para Klein 9B, Krea 2 y Wan 2.2

Las órdenes de arranque del `Makefile` se organizan en **dos ejes** que antes estaban mezclados
(modelo y perfil de memoria/atención). Esta guía dice **qué orden usar, para qué modelo y bajo qué
circunstancias**, con el porqué de cada recomendación.

## La idea: intención vs mecanismo

| Eje | Qué es | Cómo se elige |
|---|---|---|
| **Intención** | Qué vas a generar (modelo) | preset: `make klein` / `krea2` / `wan` |
| **Mecanismo** | VRAM y backend de atención | variables: `VRAM=` y `ATTN=` |

Los presets aplican el mecanismo **recomendado para ese modelo**; si tu caso es límite, lo pisas
con una variable sin cambiar de orden.

```bash
make krea2                 # VRAM auto + atención recomendada de Krea 2 (flash)
make krea2 ATTN=ck         # Krea 2 a 1280 px, backend INT8
make wan                   # Wan 2.2, atención ck (INT8) por defecto
make run VRAM=8gb ATTN=sage # caso a la carta (sage exige `make build-sage`)
```

Variables:

| Variable | Valores | Default | Qué hace |
|---|---|---|---|
| `ATTN` | `flash` `ck` `sage` `sage-triton` | `flash` | Backend de atención (ver tabla abajo) |
| `VRAM` | `auto` `8gb` `normal` `high` | `auto` | Perfil de memoria (`auto` usa `nvidia-smi`) |
| `STREAM` | `on` `off` | `on` | `--cuda-stream` dentro del perfil 8 GB (solo medición) |

`make up` ignora los ejes: arranca con el `EXTRA_ARGS` del `.env` (RunPod o perfiles propios).

## Matriz de recomendación

| Modelo | Orden | Atención por defecto | Cuándo y por qué |
|---|---|---|---|
| **Flux.2 Klein 9B turbo** | `make klein` | `flash` | **Imagen, secuencias cortas** (2–6k tokens). CK/Sage no aportan hasta ≥1280 px. Turbo 4–8 pasos: CFG 1; no usar Spectrum. ImageStitch da multi-imagen. |
| **Krea 2 turbo** | `make krea2` | `flash` (→ `ATTN=ck` a ≥1280 px) | **Imagen + TE visión Qwen3-VL** (Moodboard / Identity Edit) + Depth/Pose LoRA. Sin `--fast-fp8`. Medido: CK INT8 a 1280 px baja 26,3 s → 21,9 s; a 768 px no cambia. |
| **Wan 2.2 turbo** | `make wan` | `ck` (INT8) | **Vídeo / I2V**; ~32k tokens, la atención domina. Alternativa: Sparse Attention desde la UI — **sustituye** el backend, no se suma. Para lotes largos a resolución fija, Torch Compile. |
| **API / lotes** | `make chatbot` | `flash` | Warmup `torch.compile` (`guard_filter_fn`, compatible con `--cuda-malloc`) al tamaño del último gen. |

### Backends de atención (`ATTN`)

| Valor | Flag de runtime | Imagen | Notas |
|---|---|---|---|
| `flash` | — | `:latest` | Estable, bit-exacto. Default de la imagen. |
| `ck` | `--use-ck-attention` | `:latest` | Comfy-Kitchen INT8. Rinde en secuencias largas. No bit-exacto. |
| `sage` | `--sage-function fp16_cuda` | `:sage` | Exige `make build-sage`. Experimental en sm89 (Ada); nunca dejar `auto`. |
| `sage-triton` | `--sage-function fp16_triton` | `:sage` | Kernel Triton; probar si el CUDA fp16 falla en sm89. |

`sage` y `sage-triton` arrancan la variante `:sage`; el resto, `:latest`.

### Perfiles de VRAM (`VRAM`)

| Valor | Flags | Para |
|---|---|---|
| `auto` | según `nvidia-smi`: ≥20 GB `high`, ≥12 GB `normal`, resto 8 GB | default de los presets |
| `8gb` | `--cuda-malloc --lowvram --fp8_e4m3fn-unet --reserve-vram 2 --cuda-stream --pin-shared-memory --mmap-torch-files` | RTX 4060/3060 y similares |
| `normal` | `--cuda-malloc --normalvram --bf16-unet …` | ≥12 GB, modelo no cabe en highvram |
| `high` | `--cuda-malloc --highvram --bf16-unet …` | ≥20 GB |

Con `--lowvram` **el cuello es el trasiego de pesos RAM↔VRAM, no el cálculo**: a igual modelo, pasar
de 768 a 1280 px apenas cambia el tiempo (medido con Krea 2: 25,9 s vs 26,3 s). Por eso optimizar la
atención solo paga en secuencias largas (Wan y ≥1280 px). Medir con `make bench-offload`.

## Decisiones por modelo

**Klein 9B.** Es el caso más "normal": imagen, resolución típica ≤1280, turbo. No gana nada con un
kernel de atención alternativo, así que se queda en `flash`. Si el checkpoint cabe con `normal`/`high`
según tu VRAM, `VRAM=auto` lo detecta; en 8 GB el `fp8_e4m3fn-unet` del perfil 8 GB es lo que lo mete.

**Krea 2.** Añade el text encoder de visión de Qwen3-VL, que en 8 GB es **muy justo** (ver
`docs/integracion-krea2-moodboard-identity-edit-forge-neo_23-07-2026.md`). El preset usa `flash` porque
a las resoluciones donde Krea 2 se usa habitualmente la atención no es el cuello. **Sube a `ATTN=ck`
cuando trabajes a ≥1280 px o con Moodboard/Identity Edit** (más tokens de contexto): ahí sí se mide la
mejora. No uses `--fast-fp8` (falla en Krea 2).

**Wan 2.2.** Es el único donde la atención domina de verdad: ~32.000 tokens por clip. Por eso el preset
activa `ck` (INT8) sin pedirlo. Dos interacciones a recordar: (1) **Sparse Attention de la UI compite**
con `ck`/`sage` (instala un override que sustituye el backend); elige una. (2) Para muchos clips a
resolución fija, **Torch Compile** amortiza la compilación; en uso interactivo cambiando de resolución,
no compensa porque con `--lowvram` el offload sigue mandando.

**chatBot / API.** Mismo modelo y resolución repetidos: el warmup de `torch.compile` al size del último
gen paga. Se mantiene en `flash` (no hay secuencia larga que justifique CK/Sage).

## Benchmarks (para no fijar esto a ciegas)

| Orden | Qué compara |
|---|---|
| `make bench-attn-sweep` | `flash` vs `ck` (reinicia y mide, semilla fija) |
| `make bench-attn-sage` / `-sage-triton` | Sage fp16_cuda / fp16_triton (exige `make build-sage`) |
| `make bench-offload-sweep` | perfil 8 GB con y sin `--cuda-stream` |

Anexan filas a `docs/bench-attn_09-10-2026.md` y `docs/bench-offload_09-10-2026.md`. Mide en el caso que
te interese (Wan, o imagen ≥1280 px), no a 768 px. CK INT8 y Sage **no son bit-exactos**: a igual semilla
la imagen cambia; compara también calidad.
