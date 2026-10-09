# sd-webui-forge-neo: imagen ligera con multi-stage (builder + runtime)
# Base build: CUDA devel. Imagen final: CUDA runtime (mucho más pequeña para pull en RunPod).

# -----------------------------------------------------------------------------
# Stage 1: builder — compila e instala todo (necesita devel por compilación)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1
RUN python3.13 -m ensurepip --upgrade

# Pin de Forge Neo compatible con patches/krea2-features-backend.patch (ver patches/README.md).
ARG FORGE_NEO_REF=41359cd4b8b89212b3dbad8c9af719160a12ed63

# SageAttention es un backend de atención opcional (ver docs/attention-backends_09-10-2026.md).
# INSTALL_SAGE=1 hornea el wheel en la imagen; por defecto 0 (imagen idéntica a la actual).
ARG INSTALL_SAGE=0

WORKDIR /app
COPY patches/krea2-features-backend.patch patches/qwen35-vision-attention-fix.patch patches/cmd-flags-numeric-types.patch /tmp/
RUN git clone --filter=blob:none --no-checkout https://github.com/Haoming02/sd-webui-forge-classic webui \
    && cd webui \
    && git fetch --depth 1 origin "${FORGE_NEO_REF}" \
    && git checkout FETCH_HEAD \
    && git apply --verbose /tmp/krea2-features-backend.patch \
    && git apply --verbose /tmp/qwen35-vision-attention-fix.patch \
    && git apply --verbose /tmp/cmd-flags-numeric-types.patch \
    && rm -f /tmp/krea2-features-backend.patch /tmp/qwen35-vision-attention-fix.patch /tmp/cmd-flags-numeric-types.patch \
    && rm -rf .git

WORKDIR /app/webui
# BUILD_SLIM=1: sin onnxruntime-gpu (reduce mucho tamaño para RunPod)
# Sin --nunchaku a propósito: ese flag fuerza torch 2.11, y los wheels precompilados de
# flash_attn para cu130/cp313 solo existen contra torch 2.13 → --flash fallaba en silencio.
ARG BUILD_SLIM=0
RUN if [ "$BUILD_SLIM" = "1" ]; then \
      export COMMANDLINE_ARGS="--exit --skip-torch-cuda-test --xformers --flash"; \
    else \
      export COMMANDLINE_ARGS="--exit --skip-torch-cuda-test --xformers --flash --onnxruntime-gpu"; \
    fi && python launch.py

RUN python -m pip install --no-cache-dir python-dotenv pillow-avif-plugin imageio_ffmpeg hnswlib

# sd-dynamic-prompts: con --skip-install su install.py no corre. Sin los extras
# [attentiongrabber,magicprompt], que arrastran transformers[torch] y pisarían el torch de la imagen.
RUN python -m pip install --no-cache-dir 'dynamicprompts~=0.31.0' 'send2trash~=1.8'

# Forge Classic renombró generation_parameters_copypaste → infotext_utils sin dejar alias.
# Mismo shim que mantiene A1111 upstream; lo necesitan extensiones pre-1.9 (sd-dynamic-prompts).
RUN printf 'from modules.infotext_utils import *  # noqa: F401\n' \
    > /app/webui/modules/generation_parameters_copypaste.py

# ReActor: insightface declara dependencia de onnxruntime (CPU) y pisa el pybind de
# onnxruntime-gpu → solo quedan Azure/CPU providers y el swap falla con CUDA.
# Con --skip-install, install.py de la extensión no corre (y además pincha ORT 1.17.1,
# incompatible con CUDA 13 / Py3.13). Horneamos deps aquí.
# insightface 0.7.3 es sdist sin wheel cp313; 1.0.1 sí tiene wheel y es API-compatible.
RUN python -m pip install --no-cache-dir --no-deps 'insightface==1.0.1' \
    && python -m pip install --no-cache-dir \
      'albumentations==1.4.3' \
      'opencv-python>=4.7.0.72' \
    && python -m pip uninstall -y onnxruntime \
    && if [ "$BUILD_SLIM" != "1" ]; then \
         python -m pip install --no-cache-dir --force-reinstall --no-deps 'onnxruntime-gpu==1.28.0'; \
       fi

# Krea2 Depth/Pose ControlNet-LoRA: install.py no corre con --skip-install.
# --no-deps evita que easy-dwpose pinche numpy/huggingface_hub antiguos.
RUN python -m pip install --no-cache-dir --no-deps 'easy-dwpose==1.0.2'

# Extensiones Neo en extensions-builtin (install.py no corre con --skip-install).
# mediapipe ya viene por otras deps; no pinchar 0.10.x encima.
# boto3/aliyun-python-sdk: traductores AWS/Aliyun de Prompt All-in-One, que si no los
# reporta como "No instalado" en su panel. El `find …/docs` de la limpieza borraba
# boto3.docs (paquete Python real que boto3 importa) y dejaba su import roto; ya no.
RUN python -m pip install --no-cache-dir \
      'ultralytics==8.3.253' \
      'sqlalchemy' \
      'ZipUnicode' \
      'beautifulsoup4' \
      'pysocks' \
      'chardet' \
      'PyExecJS' \
      'lxml' \
      'pathos' \
      'openai' \
      'boto3' \
      'aliyun-python-sdk-core' \
      'aliyun-python-sdk-alimt' \
    && python -c "import torch; assert '2.' in torch.__version__, torch.__version__"

# SageAttention 2.2.0 (opt-in con --build-arg INSTALL_SAGE=1).
# Wheel precompilado (Linux cp313, CUDA 13) de snw35/sageattention-wheel; trae kernels sm80/89/90/120.
# OJO: no usar el flag `--sage` de launch.py; ese instala `sageattention==2.2.0` desde PyPI, que
# solo publica hasta 1.0.6 (sdist) → fallaría el build. Por eso se instala el wheel explícitamente.
# En sm89 (RTX 40xx) el kernel por defecto (`sageattn`) elige el path FP8, reportado inestable
# (fallos de launch / NaN); en runtime hay que forzar `--sage-function fp16_cuda`|`fp16_triton`.
ARG SAGE_WHEEL_URL=https://github.com/snw35/sageattention-wheel/releases/download/cu12-2.2.0-cu13-2.2.0/sageattention-2.2.0%2Bcu13-cp313-cp313-linux_x86_64.whl
ARG SAGE_WHEEL_SHA256=c19e3bd8aef99fdb4916bc0afb58d1ced32a9fc854fc1d1b17dc9da9e880a4b9
# OJO: pip solo acepta un wheel local si el nombre de fichero es canónico
# (`name-version-...whl`, con `+` literal en la versión). Por eso se descarga con el
# basename de la URL ya URL-decodificado (la URL trae `%2B`) en vez de un nombre fijo:
# `pip install /tmp/sage.whl` fallaba con "Invalid wheel filename".
RUN if [ "$INSTALL_SAGE" = "1" ]; then \
      whl="/tmp/$(python -c "import sys,os,urllib.parse; print(urllib.parse.unquote(os.path.basename(sys.argv[1])))" "$SAGE_WHEEL_URL")" \
      && python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$SAGE_WHEEL_URL" "$whl" \
      && python -c "import hashlib,sys; h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); sys.exit(0 if h==sys.argv[2] else 'SAGE_SHA256 mismatch: '+h)" "$whl" "$SAGE_WHEEL_SHA256" \
      && python -m pip install --no-cache-dir --no-deps "$whl" \
      && rm -f "$whl" \
      && python -c "from sageattention import sageattn; print('sageattention OK')"; \
    fi

# Limpieza agresiva para reducir tamaño (RunPod tiene límite de disco para la imagen)
# No strippear onnxruntime: rompe providers CUDA.
# No borrar docs/doc que sean paquetes Python (con __init__.py): boto3/botocore y otros
# los importan en runtime, y borrarlos rompía su import.
RUN find /usr/local/lib/python3.13 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -name '*.pyc' -delete 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -type d \( -name 'docs' -o -name 'doc' \) ! -exec test -e {}/__init__.py \; -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -name '*.a' -delete 2>/dev/null || true \
    && find /usr/local/lib/python3.13 -name '*.so' ! -path '*/onnxruntime/*' ! -path '*/sageattention/*' -exec strip --strip-unneeded {} \; 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.13/site-packages/torch/share 2>/dev/null || true \
    && find /app/webui -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /app/webui -name '*.pyc' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: runtime — solo librerías CUDA runtime + Python + artefactos
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# gcc: Triton (flash/sage JIT) necesita un C compiler en runtime; sin él:
# RuntimeError: Failed to find C compiler. Please specify via CC ...
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ffmpeg \
    git \
    gcc \
    g++ \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

WORKDIR /app/webui

COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /app/webui /app/webui

# Extensiones custom en la imagen (Forge también carga extensions-builtin).
# El README.md de esta carpeta queda como fichero suelto e inofensivo.
COPY builtin-extensions/ /app/webui/extensions-builtin/

ENV COMMANDLINE_ARGS="--listen --port 7860 --data-dir /data --gradio-allowed-path /app/webui --gradio-allowed-path /data --agent-scheduler-sqlite-file /data/task_scheduler.sqlite3 --enable-insecure-extension-access --skip-prepare-environment --skip-install --api"
EXPOSE 7860
VOLUME ["/data"]

COPY entrypoint.sh /entrypoint.sh
COPY ensure_config_then_launch.py /app/webui/ensure_config_then_launch.py
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "ensure_config_then_launch.py"]
