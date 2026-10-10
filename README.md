# forge-neo-tuned

Forge Neo ([sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic), rama **neo**) **curado y parchado**: Krea 2 completo, extensiones Neo horneadas, dependencias Python al día (sin los pins rotos de cada `install.py`) y fixes de runtime para CUDA 13 / Python 3.13 / lowvram.

Compose y RunPod son solo el empaquetado. Imagen: **`ghcr.io/pcgarat/forge-neo`**.

## Qué está tuneado (frente a Neo vanilla)

### Core parchado (en imagen)

| Pieza | Qué |
|-------|-----|
| **Krea 2 Moodboard + Identity Edit** | Patch regenerado del [toolkit](https://github.com/RedNodeAI/forge-neo-krea2-toolkit) sobre Neo pinneado (`FORGE_NEO_REF`); convive con el edit nativo upstream de img2img |
| **Qwen3-VL visión / lowvram** | Fallback a `attention_pytorch` cuando el encoder visual corre en CPU (evita `flash_attn` en CPU + spam de ERROR) |
| **Fixes del merge** | p. ej. `dynamic_args` sin `.pop()` (metaclase Neo); convivencia Moodboard/Edit vs `krea2_do_reference`; shape post-`_unpack_context` |
| **Shim A1111** | `generation_parameters_copypaste` → `infotext_utils` (extensiones pre-1.9 / dynamic-prompts) |
| **Sin `--nunchaku` en build** | Ese flag fuerza torch incompatible con wheels `flash_attn` cu130/cp313; se prioriza flash |

Detalle de regeneraciones y riesgos: [`patches/README.md`](patches/README.md).

### Extensiones Neo horneadas (`builtin-extensions/` → `extensions-builtin`)

Sobreviven a un volumen vacío (RunPod nuevo). Deps Python van en la imagen porque arranca con `--skip-install`.

| Extensión | Rol |
|-----------|-----|
| Krea2 Depth / Pose ControlNet-LoRA | Depth-Anything + adapter Krea |
| ADetailer-Neo | Caras / manos |
| State Manager Neo | Guardar/restaurar configs |
| CivitAI Browser Neo | Browser de modelos (sustituye Civitai Helper) |
| Agent Scheduler Neo | Cola de generaciones |
| Prompt All-in-One Neo | Historial, estilos, traducción |
| Lama Cleaner Neo | Borrado / inpaint de objetos |

### Semilla en volumen (`extensions/`)

Se copian solo si faltan: **IIB**, **Zoom Image**, **Krea2 Moodboard UI**, **Krea2 Identity Edit UI** (con fix `dynamic_args`).

Documentado (no en semilla automática): **sd-dynamic-prompts** + estilos fotográficos Krea2, **ReActor** (Codeberg).

### Dependencias “al día” y anti-pinchos

Con `--skip-install`, cada extensión no puede instalar lo suyo; se hornea a propósito:

| Área | Ajuste |
|------|--------|
| ReActor | `insightface==1.0.1` (wheel cp313), `albumentations`, **reafirma `onnxruntime-gpu`** (insightface pisa ORT CPU) |
| Depth/Pose | `easy-dwpose` con `--no-deps` (no pincha numpy / huggingface_hub viejos) |
| Dynamic prompts | `dynamicprompts` + `send2trash` **sin** extras MagicPrompt (evitarían `transformers[torch]` y pisarían torch) |
| Builtin Neo | ultralytics, sqlalchemy, etc.; boto3 y aliyun-python-sdk (traductores AWS/Aliyun de Prompt All-in-One) |
| Stack base | xformers, Flash Attention, onnxruntime-gpu (salvo slim), FFmpeg |
| Runtime | `gcc`/`g++` para Triton JIT (`torch.compile` / flash / sage) |

### Operativa

- Presets por modelo: `make klein` / `krea2` / `wan` (VRAM auto + atención recomendada) y `make run ATTN= VRAM=`
- `make up-interactive` deriva ejes (modelo/tamaño/imagen-vídeo/API/offload) en un asistente; `make preflight-gpu` valida driver y `nvidia-persistenced` antes de arrancar
- Estado en `/data`: base SQLite de Agent Scheduler y `ui-config.json` (State Manager) sobreviven al contenedor
- Perfil 8 GB con `--cuda-stream` + `--pin-shared-memory` (memlock holgado en compose) para solapar el trasiego RAM↔VRAM
- `PUID`/`PGID` + `setpriv` (output no queda root)
- Layout único `forge-data/` local ↔ RunPod
- Variantes `:cuda12`, `:slim` y `:sage` (backend SageAttention opt-in) para hosts/RunPod restrictivos

## Requisitos

| Requisito | Detalle |
|-----------|---------|
| Docker + Compose v2 | `make install-docker` en Ubuntu/Debian |
| NVIDIA Container Toolkit | `--gpus all` |
| Driver | CUDA **13** (o imagen `:cuda12`) |
| VRAM | ≥ 12–16 GB Krea 2 + TE visión; ≥ 20 GB highvram |
| Disco / RAM | Imagen + modelos; ≥ 16 GB RAM |

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

## Arranque rápido

```bash
cp .env.example .env
make build
make up
```

- **WebUI:** http://localhost:7860 · **API:** http://localhost:7860/docs · **Ayuda:** `make help`

## Krea 2 — pesos en el volumen

No van en la imagen:

| Asset | Carpeta |
|-------|---------|
| Checkpoint Krea 2 | `models/Stable-diffusion/` |
| TE visión (`qwen3vl_4b_bf16` / `fp8_scaled`) | `models/text_encoder/` |
| LoRA identity edit | `models/Lora/` |
| Depth ControlNet-LoRA (~862 MB) | `Models/ControlNet/Krea2/depth-control-lora.safetensors` |

```bash
make krea2-ext        # UI Moodboard + Edit → volumen
make krea2-depth-ext  # Depth/Pose en imagen → make build
make restart
```

Guía: [docs/integracion-krea2-moodboard-identity-edit-forge-neo_23-07-2026.md](docs/integracion-krea2-moodboard-identity-edit-forge-neo_23-07-2026.md).

## Perfiles, datos y extras

| Comando | Uso |
|---------|-----|
| `make klein` | Flux.2 Klein 9B turbo: VRAM auto + atención `flash` |
| `make krea2` | Krea 2 turbo: VRAM auto + `ck` (INT8; `ATTN=flash` para bit-exacto) |
| `make wan` | Wan 2.2 turbo (vídeo): VRAM auto + `ck` (INT8) |
| `make run ATTN= VRAM=` | Arranque manual con los ejes a la carta (`ATTN=flash|ck|sage|sage-triton`, `VRAM=auto|8gb|normal|high`, `STREAM=on|off`) |
| `make up-interactive` | Asistente: pregunta modelo, tamaño, imagen/vídeo, API y solape de offload… y arranca con lo recomendado |
| `make up` | Arranca con el `EXTRA_ARGS` del `.env`, sin ejes |
| `make restart` | Recrea el contenedor con la config exacta de la ejecución actual (imagen + `EXTRA_ARGS` + `COMMANDLINE_ARGS`); sin contenedor cae a `make up` |
| `make preflight-gpu` | Valida driver NVIDIA y socket de `nvidia-persistenced` (evita fallos crípticos de runc) |
| `make chatbot` | Perfil 8 GB + warmup `torch.compile` para API |
| `make bench-attn` / `bench-attn-sweep` | Medir backends de atención (misma semilla) → informe md |
| `make bench-offload` / `bench-offload-sweep` | Medir trasiego de pesos RAM↔VRAM (`--cuda-stream` on/off) → informe md |
| `make iib-access` | Permisos IIB a salidas |
| `make reactor-fix` | Reafirma ORT-GPU en contenedor vivo |
| `make seed-extensions` | Siembra `extensions/` si faltan |
| `make test-choose-profile` / `test-bench` / `test-bench-offload` | Tests de los helpers (sin GPU) |

`ATTN=sage`/`sage-triton` exigen `make build-sage` antes (imagen `:sage`, publicable con `make push-sage`).
`make help` agrupa todas las órdenes con su cuándo; detalle en [`docs/guia-perfiles_09-10-2026.md`](docs/guia-perfiles_09-10-2026.md).

`.env`: `DATA_PATH`, `EXTENSIONS_PATH`, `PUID`/`PGID`, `EXTRA_ARGS`, subdirs de modelos e IIB
(`IIB_ACCESS_CONTROL`, `IIB_ACCESS_CONTROL_ALLOWED_PATHS`). Tras cambios: `make up` (`make restart`
conserva la config del contenedor en marcha y no relee el `.env`).

**ReActor:** [codeberg.org/Gourieff/sd-webui-reactor](https://codeberg.org/Gourieff/sd-webui-reactor) — no el fork `-sfw` (ORT 1.17.1).

Dynamic prompts + estilos: ver [`extensions/README.md`](extensions/README.md).

## Publicar / RunPod

```bash
docker login ghcr.io -u TU_GITHUB_USER
make push            # :latest
make push-cuda12     # :cuda12
make push-slim       # :slim
make push-sage       # :sage (backend SageAttention)
```

Pod = esta imagen; volumen en `/workspace` → datos en `/workspace/forge-data`; HTTP **7860**.

## Docs

| Doc | Contenido |
|-----|-----------|
| [`patches/`](patches/) | Regeneración de patches Krea2 / Qwen3-VL |
| [Guía de perfiles](docs/guia-perfiles_09-10-2026.md) | Qué orden usar por modelo (Klein / Krea 2 / Wan) y por qué |
| [Krea 2 integración](docs/integracion-krea2-moodboard-identity-edit-forge-neo_23-07-2026.md) | Diseño del toolkit en esta stack |
| [Scripts txt2img](docs/guia-scripts-txt2img_23-09-2026.md) | Never OOM, Torch Compile, Depth… en 8 GB |
| [Backends de atención](docs/attention-backends_09-10-2026.md) | Opciones, riesgos en sm89 y cómo medir |
| [Offload RAM↔VRAM](docs/bench-offload_09-10-2026.md) | Trasiego de pesos: `--cuda-stream`, pinning y cómo medir |
| [Planteamiento](docs/planteamiento-docker-forge-neo_28-02-2025.md) | Arquitectura de imagen |
| [models.md](models.md) | Layout de modelos |
| [`builtin-extensions/`](builtin-extensions/) · [`extensions/`](extensions/) | Qué va en imagen vs volumen |
| [`.cursor/skills/`](.cursor/skills/) | Skills Forge Neo (extensiones compatibles y custom scripts) para el asistente |

## Upstream

- Forge Neo: [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
- Toolkit Krea2: [RedNodeAI/forge-neo-krea2-toolkit](https://github.com/RedNodeAI/forge-neo-krea2-toolkit)
- SageAttention (wheel cp313/cu13): [snw35/sageattention-wheel](https://github.com/snw35/sageattention-wheel)

Respeta licencias de Forge, extensiones y modelos.
