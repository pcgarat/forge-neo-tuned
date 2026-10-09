# forge-neo — build, arranque y operación
# Uso: make [target] [VAR=valor];  make help

COMPOSE  = docker compose
SERVICE  = forge-neo
PORT     = 7860

# --- Ejes de configuración de arranque --------------------------------------
# ATTN: backend de atención.
#   flash (def)  ya activo en la imagen; estable y bit-exacto.
#   ck           Comfy-Kitchen INT8 (--use-ck-attention); rinde en secuencias
#                largas (vídeo Wan, imagen >=1280 px). No bit-exacto.
#   sage         SageAttention fp16_cuda; EXIGE `make build-sage`. Experimental en sm89.
#   sage-triton  Sage por Triton; EXIGE `make build-sage`.
# VRAM: perfil de memoria.
#   auto (def)  detecta por nvidia-smi: >=20 GB highvram, >=12 GB normalvram, resto 8 GB.
#   8gb         fuerza el perfil 8 GB (lowvram + offload asíncrono).
#   normal|high fuerza normalvram / highvram.
ATTN ?= flash
VRAM ?= auto

# STREAM=off quita --cuda-stream del perfil 8 GB (lo usa bench-offload-sweep).
STREAM ?= on

# Versiones para install-docker (repo estable de Docker). Vacío = última del repo.
# Para fijar: DOCKER_CE_VERSION=5:29.2.1-1~ubuntu.24.04~noble (ejemplo Ubuntu 24.04).
DOCKER_CE_VERSION ?=
DOCKER_COMPOSE_PLUGIN_VERSION ?=

# Imagen para push a registro. Ej.: make push REGISTRY_IMAGE=ghcr.io/pcgarat/forge-neo:latest
REGISTRY_IMAGE ?= ghcr.io/$(GITHUB_USER)/forge-neo:latest
REGISTRY_IMAGE_CUDA12 ?= ghcr.io/$(GITHUB_USER)/forge-neo:cuda12
GITHUB_USER ?= pcgarat

# Perfil GPU <=12 GB (p. ej. RTX 4060 8 GB): Klein 9B, Flux, Qwen y modelos grandes.
# cuda-malloc + lowvram + fp8 + offload RAM. Sin --fast-fp8 (falla en Krea2 y solo ralentiza).
# cuda-stream solapa la subida de pesos con el cómputo (hilos de offload); con lowvram el
# cuello es el trasiego RAM<->VRAM, no el cálculo. El pinning solo aplica con memlock holgado
# (ver docker-compose.yml). Medir con `make bench-offload`.
ARGS_8GB = --cuda-malloc --lowvram --fp8_e4m3fn-unet --reserve-vram 2 --cuda-stream --pin-shared-memory --mmap-torch-files
# Sin offload asíncrono, para aislar su efecto en el benchmark.
ARGS_8GB_NOSTREAM = $(filter-out --cuda-stream,$(ARGS_8GB))
ARGS_8GB_SEL = $(if $(filter off,$(STREAM)),$(ARGS_8GB_NOSTREAM),$(ARGS_8GB))
# Perfil >=12 GB: bf16 y sin offload de UNet. Igual para normal y high salvo el modo de VRAM.
ARGS_NORMAL = --cuda-malloc --normalvram --bf16-unet --pin-shared-memory --mmap-torch-files
ARGS_HIGH   = --cuda-malloc --highvram --bf16-unet --pin-shared-memory --mmap-torch-files

# Imagen a arrancar: la estándar, salvo Sage (variante horneada).
FORGE_IMAGE ?= forge-neo:latest
FORGE_IMAGE_SAGE ?= forge-neo:sage

# ATTN -> flags de runtime. flash no añade flag (es el backend por defecto de la imagen).
ATTN_FLAGS = $(strip $(if $(filter ck,$(ATTN)),--use-ck-attention,$(if $(filter sage,$(ATTN)),--sage-function fp16_cuda,$(if $(filter sage-triton,$(ATTN)),--sage-function fp16_triton,))))
# ATTN -> imagen. Sage/sage-triton requieren el wheel horneado (make build-sage).
ATTN_IMAGE = $(if $(filter sage sage-triton,$(ATTN)),$(FORGE_IMAGE_SAGE),$(FORGE_IMAGE))

.PHONY: help build build-no-cache build-cuda12 build-slim build-sage push push-cuda12 push-slim push-sage \
        run up down restart logs shell ps clean klein krea2 wan chatbot chatbot-warmup test-warmup \
        workspace seed-extensions preflight-gpu preflight-gpu-check \
        bench-attn bench-attn-sweep bench-attn-flash bench-attn-ck bench-attn-sage bench-attn-sage-triton test-bench \
        bench-offload bench-offload-sweep bench-offload-sweep-nostream bench-offload-sweep-stream test-bench-offload \
        iib-access krea2-ext krea2-depth-ext reactor-fix install-docker _launch-model

help:
	@echo "forge-neo — make [target] [VAR=valor];  make help"
	@echo ""
	@echo "ARRANQUE POR MODELO (fija VRAM y atención recomendadas; Ctrl+C sale de los logs)"
	@echo "  make klein   Flux.2 Klein 9B turbo — imagen; ImageStitch para multi-imagen."
	@echo "               CFG 1, 4-8 pasos. Atención flash (secuencias cortas: CK/Sage no aportan)."
	@echo "  make krea2   Krea 2 turbo — imagen + TE visión Qwen3-VL (Moodboard / Identity Edit)"
	@echo "               y Depth/Pose ControlNet-LoRA. A >=1280 px prueba ATTN=ck."
	@echo "  make wan     Wan 2.2 turbo — vídeo / I2V; ~32k tokens, la atención domina."
	@echo "               Por defecto ATTN=ck (INT8). No combinar con Sparse Attention (se sustituyen)."
	@echo ""
	@echo "ARRANQUE MANUAL"
	@echo "  make run      Arranca con VRAM y ATTN a mano (mismos ejes que los presets)."
	@echo "  make up       Arranca con el EXTRA_ARGS del .env, sin ejes (RunPod / perfiles propios)."
	@echo "  make down     Para y elimina el contenedor."
	@echo "  make restart  down + up (por defecto con el perfil del .env)."
	@echo "  make logs     Sigue los logs del servicio (Ctrl+C para salir)."
	@echo "  make shell    Abre una shell dentro del contenedor."
	@echo "  make ps       Estado del servicio."
	@echo ""
	@echo "  Ejes (para make run y presets):"
	@echo "    ATTN=flash|ck|sage|sage-triton   sage* exige 'make build-sage' antes"
	@echo "    VRAM=auto|8gb|normal|high        auto detecta por nvidia-smi"
	@echo "    STREAM=on|off                    --cuda-stream en el perfil 8 GB (medición)"
	@echo ""
	@echo "IMAGEN"
	@echo "  make build           Construye la imagen (primera vez o tras cambios)"
	@echo "  make build-no-cache  Reconstruye sin caché (entrypoint, fixes de config.json)"
	@echo "  make build-sage      Imagen + SageAttention horneado (tag :sage) para ATTN=sage*"
	@echo "  make build-cuda12    Variante CUDA 12.4 (RunPod con driver < CUDA 13)"
	@echo "  make build-slim      Variante slim (sin onnxruntime-gpu/nunchaku; menos tamaño)"
	@echo ""
	@echo "DATOS Y EXTENSIONES"
	@echo "  make workspace       Crea el árbol de datos y siembra extensions/ si faltan (antes del 1er up)"
	@echo "  make seed-extensions Copia extensiones a EXTENSIONS_PATH solo si falta cada carpeta"
	@echo "  make krea2-ext       Krea2 Moodboard + Identity Edit (UI) desde GitHub"
	@echo "  make krea2-depth-ext Depth/Pose ControlNet-LoRA (builtin); luego make build"
	@echo "  make iib-access      .env de Infinite Image Browsing con acceso a carpetas de salida"
	@echo "  make reactor-fix     Repara deps de ReActor (onnxruntime-gpu vs CPU) en contenedor vivo"
	@echo ""
	@echo "DIAGNÓSTICO"
	@echo "  make preflight-gpu   Verifica driver NVIDIA y socket de nvidia-persistenced (antes de up)"
	@echo "  make chatbot         Perfil 8 GB + warmup torch.compile al size del último gen (API)"
	@echo "  make chatbot-warmup  Solo warmup (Forge ya tiene que estar arriba)"
	@echo "  make test-warmup     Tests del parser/payload de chatbot-warmup"
	@echo ""
	@echo "BENCHMARKS (anexan filas a docs/*.md; requieren GPU)"
	@echo "  make bench-attn         Mide N gens (semilla fija) del backend activo"
	@echo "  make bench-attn-sweep   Compara flash vs CK INT8 (reinicia y mide)"
	@echo "  make bench-attn-sage    Mide Sage fp16_cuda (requiere 'make build-sage')"
	@echo "  make bench-attn-sage-triton  Mide Sage fp16_triton"
	@echo "  make test-bench         Tests del benchmark de atención"
	@echo "  make bench-offload      Mide it/s vs resolución y MB descargados del perfil activo"
	@echo "  make bench-offload-sweep  Compara --cuda-stream on/off (reinicia y mide)"
	@echo "  make test-bench-offload Tests del benchmark de offload"
	@echo ""
	@echo "PUBLICAR Y RUNPOD"
	@echo "  make push | push-cuda12 | push-slim | push-sage   Construyen, etiquetan y suben"
	@echo ""
	@echo "INSTALACIÓN"
	@echo "  make install-docker  Docker Engine + Compose plugin desde repo oficial (Ubuntu/Debian)"
	@echo ""
	@echo "  make clean           down y elimina la imagen local"
	@echo ""
	@echo "WebUI: http://localhost:$(PORT)   API: http://localhost:$(PORT)/docs"

# Cargar .env y exportar para que compose use DATA_PATH y EXTENSIONS_PATH en los volúmenes
ENV_LOAD = set -a && [ -f .env ] && . ./.env && set +a

# Resuelve VRAM a los flags del perfil y los deja en la variable shell `extra`.
# Un solo sitio para que presets, `run` y benchmarks compartan la misma lógica.
define resolve_vram
if [ "$(VRAM)" = "8gb" ]; then \
  echo "  VRAM: 8 GB (forzado)"; extra="$(ARGS_8GB_SEL)"; \
elif [ "$(VRAM)" = "normal" ]; then \
  echo "  VRAM: normalvram (forzado)"; extra="$(ARGS_NORMAL)"; \
elif [ "$(VRAM)" = "high" ]; then \
  echo "  VRAM: highvram (forzado)"; extra="$(ARGS_HIGH)"; \
else \
  mem=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); \
  if [ -z "$$mem" ]; then echo "  VRAM: sin nvidia-smi -> 8 GB"; extra="$(ARGS_8GB_SEL)"; \
  elif [ "$$mem" -ge 20000 ]; then echo "  VRAM: $$mem MB -> highvram"; extra="$(ARGS_HIGH)"; \
  elif [ "$$mem" -ge 12000 ]; then echo "  VRAM: $$mem MB -> normalvram"; extra="$(ARGS_NORMAL)"; \
  else echo "  VRAM: $$mem MB -> 8 GB"; extra="$(ARGS_8GB_SEL)"; fi; \
fi
endef

# --- Arranque ---------------------------------------------------------------
# Receta común de arranque: resuelve VRAM + ATTN y levanta el servicio.
_launch-model: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@$(ENV_LOAD); \
	$(resolve_vram); \
	echo "  Atención: $(ATTN) $(ATTN_FLAGS)"; \
	echo "  Imagen:   $(ATTN_IMAGE)"; \
	export EXTRA_ARGS="$$extra $(ATTN_FLAGS)" FORGE_IMAGE="$(ATTN_IMAGE)"; \
	$(COMPOSE) up -d$(if $(BENCH_DETACH),, && $(MAKE) --no-print-directory logs)

# Presets por modelo: fijan el ATTN recomendado y detectan VRAM. Override con ATTN=/VRAM=.
klein: ATTN = flash
krea2: ATTN = flash
wan:   ATTN = ck
klein krea2 wan run: _launch-model

build:
	$(COMPOSE) build

# Reconstruir sin caché (para que entren cambios en entrypoint, Dockerfile, etc.)
build-no-cache:
	$(COMPOSE) build --no-cache

# Variante CUDA 12.4: para RunPod (u otros hosts) donde el driver no soporta CUDA 13
build-cuda12:
	docker build -f Dockerfile.cuda12 -t forge-neo:cuda12 .

# Variante slim: sin onnxruntime-gpu ni nunchaku, menos tamaño (para no exceder límite RunPod)
build-slim:
	docker build --build-arg BUILD_SLIM=1 -t forge-neo:slim .

# Variante con SageAttention horneado (backend opcional; ver docs/attention-backends_09-10-2026.md).
# No pisa :latest; arranca con `make run ATTN=sage` o un preset con ATTN=sage.
build-sage:
	DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_SAGE=1 -t $(FORGE_IMAGE_SAGE) .

# Arranque con EXTRA_ARGS del .env, sin ejes (RunPod / perfiles propios).
up: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@$(ENV_LOAD) && $(COMPOSE) up -d$(if $(BENCH_DETACH),, && $(MAKE) --no-print-directory logs)

down:
	$(COMPOSE) down

restart: down up

# Comprueba en el host lo que runc necesita para montar la GPU en el contenedor.
# Evita fallos crípticos de `docker compose up` (p. ej. "open /run/nvidia-persistenced/socket:
# no such file or directory") cuando el driver NVIDIA está desincronizado o el daemon está parado.
preflight-gpu-check:
	@if ! nvidia-smi >/dev/null 2>&1; then \
	  mod=$$(sed -n 's/.*Module for x86_64 *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null); \
	  lib=$$(readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null | sed -n 's/.*libnvidia-ml[.]so[.]//p'); \
	  echo "ERROR: nvidia-smi no funciona; no se puede exponer la GPU a Docker."; \
	  echo "       Módulo cargado:  $$mod"; \
	  echo "       Librería NVML:   $$lib"; \
	  if [ -n "$$mod" ] && [ "$$mod" != "$$lib" ]; then \
	    echo "       Desajuste driver/kernel: el módulo en memoria no se recarga en caliente"; \
	    echo "       con el escritorio usando la GPU. Arréglalo con: sudo reboot"; \
	  elif [ -f /var/run/reboot-required ]; then \
	    echo "       Hay un reinicio pendiente; arranca con: sudo reboot"; \
	  else \
	    echo "       Revisa dmesg y que libnvidia-container-toolkit esté instalado."; \
	  fi; \
	  exit 1; \
	fi; \
	if [ ! -S /run/nvidia-persistenced/socket ]; then \
	  echo "ERROR: falta /run/nvidia-persistenced/socket (nvidia-persistenced parado)."; \
	  echo "       El CDI spec de NVIDIA lo monta en el contenedor; sin él, runc falla."; \
	  echo "       Arréglalo con: sudo systemctl restart nvidia-persistenced"; \
	  exit 1; \
	fi

preflight-gpu:
	@$(MAKE) --no-print-directory preflight-gpu-check

# Ojo: las recetas lo invocan como sub-make (`$(MAKE) preflight-gpu`) y NO como
# prerrequisito de `up:`. Con `make -j` los prerrequisitos corren en paralelo y la
# comprobación podría perder la carrera contra `docker compose up`; el sub-make no.

# --- API / chatBot ----------------------------------------------------------
# Llamadas API del chatBot: mismo modelo/size, N escenas seguidas.
# 8 GB + compile guard_filter_fn (compatible con --cuda-malloc; max-autotune no lo es).
# El warmup es 1 step al size de params.txt; no pisa el último gen (save_images=false + restaura params.txt).
chatbot: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil chatBot 8 GB: $(ARGS_8GB)"
	@$(ENV_LOAD) && export EXTRA_ARGS="$(ARGS_8GB)" && $(COMPOSE) up -d && $(MAKE) --no-print-directory chatbot-warmup && $(MAKE) --no-print-directory logs

chatbot-warmup:
	@$(ENV_LOAD); \
	data="$${DATA_PATH:-/workspace/forge-data}"; \
	python3 "$(CURDIR)/scripts/chatbot_warmup.py" --data-path "$$data" --base-url "http://127.0.0.1:$(PORT)"

test-warmup:
	python3 -m unittest tests.test_chatbot_warmup -v

# --- Benchmark de backends de atención -------------------------------------
# Mide N generaciones idénticas contra el Forge que ya está arriba y anexa una fila
# al informe. No reinicia nada: orquesta `make bench-attn-sweep` (requiere GPU).
BENCH_ARGS   ?=
BENCH_REPORT ?= docs/bench-attn_09-10-2026.md

bench-attn:
	@$(ENV_LOAD); \
	data="$${DATA_PATH:-/workspace/forge-data}"; \
	python3 "$(CURDIR)/scripts/bench_attn.py" \
	  --data-path "$$data" \
	  --base-url "http://127.0.0.1:$(PORT)" \
	  --out "$(CURDIR)/$(BENCH_REPORT)" \
	  $(BENCH_ARGS)

test-bench:
	python3 -m unittest tests.test_bench_attn -v

# --- Benchmark de trasiego de pesos (offload RAM<->VRAM) --------------------
# Mide la curva it/s vs resolución del perfil activo y anexa un bloque por perfil.
# Si el modelo cabe entero en VRAM no hay líneas de offload: fuerza VRAM=8gb para medir.
BENCH_OFFLOAD_ARGS   ?= --sizes 768x768,1024x1024,1280x1280 --steps 8 --runs 3
BENCH_OFFLOAD_REPORT ?= docs/bench-offload_09-10-2026.md

bench-offload:
	@$(ENV_LOAD); \
	data="$${DATA_PATH:-/workspace/forge-data}"; \
	python3 "$(CURDIR)/scripts/bench_offload.py" \
	  --data-path "$$data" \
	  --base-url "http://127.0.0.1:$(PORT)" \
	  --container "$(SERVICE)" \
	  --out "$(CURDIR)/$(BENCH_OFFLOAD_REPORT)" \
	  --label "$${BENCH_OFFLOAD_LABEL:-active}" \
	  $(BENCH_OFFLOAD_ARGS)

test-bench-offload:
	python3 -m unittest tests.test_bench_offload -v

# Compara base (flash) vs CK INT8. Requiere GPU operativa (nvidia-smi). Intervalo configurable.
BENCH_SLEEP ?= 30
bench-attn-sweep: bench-attn-flash bench-attn-ck

bench-attn-flash:
	@$(MAKE) --no-print-directory run VRAM=8gb ATTN=flash BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label flash"

bench-attn-ck:
	@$(MAKE) --no-print-directory run VRAM=8gb ATTN=ck BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label ck-int8"

bench-attn-sage: build-sage
	@$(MAKE) --no-print-directory run VRAM=8gb ATTN=sage BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label sage-fp16_cuda"

bench-attn-sage-triton: build-sage
	@$(MAKE) --no-print-directory run VRAM=8gb ATTN=sage-triton BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label sage-fp16_triton"

# Compara el perfil 8 GB con offload asíncrono (--cuda-stream) frente a sin él.
BENCH_OFFLOAD_SLEEP ?= 30
bench-offload-sweep: bench-offload-sweep-nostream bench-offload-sweep-stream

bench-offload-sweep-nostream:
	@$(MAKE) --no-print-directory run VRAM=8gb STREAM=off ATTN=flash BENCH_DETACH=1
	@sleep $(BENCH_OFFLOAD_SLEEP)
	@$(MAKE) --no-print-directory bench-offload BENCH_OFFLOAD_LABEL="8gb-nostream"

bench-offload-sweep-stream:
	@$(MAKE) --no-print-directory run VRAM=8gb STREAM=on ATTN=flash BENCH_DETACH=1
	@sleep $(BENCH_OFFLOAD_SLEEP)
	@$(MAKE) --no-print-directory bench-offload BENCH_OFFLOAD_LABEL="8gb-stream"

logs:
	$(COMPOSE) logs -f $(SERVICE)

shell:
	$(COMPOSE) exec $(SERVICE) /bin/bash

# Crea el árbol de datos (DATA_PATH / EXTENSIONS_PATH del .env, o defaults bajo /workspace)
# y siembra extensiones custom del repo si faltan en el destino.
# Override CLI: make workspace DATA_PATH=./d EXTENSIONS_PATH=./d/extensions
workspace:
	@cli_data="$(DATA_PATH)"; cli_ext="$(EXTENSIONS_PATH)"; \
	$(ENV_LOAD); \
	data="$${cli_data:-$${DATA_PATH:-/workspace/forge-data}}"; \
	ext="$${cli_ext:-$${EXTENSIONS_PATH:-$$data/extensions}}"; \
	mkdir -p "$$ext" "$$data/models" "$$data/output" "$$data/Models" 2>/dev/null || mkdir -p "$$ext" "$$data/models" "$$data/output"; \
	echo "Árbol listo: data=$$data extensions=$$ext"; \
	EXTENSIONS_PATH="$$ext" $(MAKE) seed-extensions

# Copia cada subcarpeta de ./extensions a EXTENSIONS_PATH solo si aún no existe en destino.
# No sobrescribe instalaciones existentes (DB IIB, .env, etc.). No toca builtins de la imagen.
# Override CLI: make seed-extensions EXTENSIONS_PATH=/ruta/limpia
seed-extensions:
	@cli_ext="$(EXTENSIONS_PATH)"; \
	set -e; \
	$(ENV_LOAD); \
	src="$(CURDIR)/extensions"; \
	ext_root="$${cli_ext:-$${EXTENSIONS_PATH:-/workspace/forge-data/extensions}}"; \
	if [ ! -d "$$src" ]; then \
	  echo "No hay $$src; nada que sembrar."; \
	  exit 0; \
	fi; \
	mkdir -p "$$ext_root"; \
	copied=0; skipped=0; \
	for d in "$$src"/*/; do \
	  [ -d "$$d" ] || continue; \
	  name=$$(basename "$$d"); \
	  dest="$$ext_root/$$name"; \
	  if [ -e "$$dest" ]; then \
	    echo "  skip  $$name (ya existe en $$dest)"; \
	    skipped=$$((skipped+1)); \
	  else \
	    echo "  copy  $$name → $$dest"; \
	    if command -v rsync >/dev/null 2>&1; then \
	      rsync -a \
	        --exclude '.git/' \
	        --exclude 'iib.db' \
	        --exclude 'iib_db_backup/' \
	        --exclude '*.log' \
	        --exclude '__pycache__/' \
	        "$$d" "$$dest/" || { \
	          echo "ERROR: no se pudo copiar $$name → $$dest (¿permisos?)."; \
	          echo "  Prueba: sudo chown -R \$$(whoami) \"$$ext_root\""; \
	          exit 1; \
	        }; \
	    else \
	      mkdir -p "$$dest"; \
	      cp -a "$$d"/. "$$dest/" || { \
	          echo "ERROR: no se pudo copiar $$name → $$dest (¿permisos?)."; \
	          echo "  Prueba: sudo chown -R \$$(whoami) \"$$ext_root\""; \
	          exit 1; \
	        }; \
	    fi; \
	    copied=$$((copied+1)); \
	  fi; \
	done; \
	echo "Semilla extensions: $$copied copiadas, $$skipped omitidas → $$ext_root"

# Crea .env en la extensión Infinite Image Browsing con acceso a carpetas de salida (/data/output, /data/Images).
# Requiere EXTENSIONS_PATH en .env y que la extensión sd-webui-infinite-image-browsing esté instalada.
iib-access:
	@$(ENV_LOAD); \
	ext_dir="$${EXTENSIONS_PATH:-/workspace/forge-data/extensions}/sd-webui-infinite-image-browsing"; \
	if [ ! -d "$$ext_dir" ]; then \
	  echo "No existe $$ext_dir. Instala antes la extensión Infinite Image Browsing desde la pestaña Extensiones."; \
	  exit 1; \
	fi; \
	printf '%s\n%s\n' 'IIB_ACCESS_CONTROL=enable' 'IIB_ACCESS_CONTROL_ALLOWED_PATHS=txt2img,img2img,extra,save,/data/output,/data/Images' > "$$ext_dir/.env"; \
	echo "Creado $$ext_dir/.env con acceso a carpetas de salida. Reinicia la WebUI o recarga la extensión.";

# Instala/actualiza las extensiones UI de Krea2 Moodboard + Identity Edit en EXTENSIONS_PATH.
# El backend patch ya va en la imagen (Dockerfile). Modelos/LoRA/TE van en DATA_PATH (manual).
# Docs: docs/integracion-krea2-moodboard-identity-edit-forge-neo_23-07-2026.md
# Nota: git fetch por SHA corto falla en GitHub; usar SHA completo o rama (main).
KREA2_TOOLKIT_REF ?= 8aac7a745202ae2eecdf4435b1a85fb5466ee51c
krea2-ext:
	@$(ENV_LOAD); \
	ext_root="$${EXTENSIONS_PATH:-/workspace/forge-data/extensions}"; \
	mkdir -p "$$ext_root"; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	echo "Clonando forge-neo-krea2-toolkit @ $(KREA2_TOOLKIT_REF)…"; \
	git clone --filter=blob:none --no-checkout https://github.com/RedNodeAI/forge-neo-krea2-toolkit "$$tmp/toolkit" \
	  && git -C "$$tmp/toolkit" fetch --depth 1 origin "$(KREA2_TOOLKIT_REF)" \
	  && git -C "$$tmp/toolkit" checkout FETCH_HEAD \
	  && rm -rf "$$ext_root/sd-forge-krea2-moodboard" "$$ext_root/sd-forge-krea2-edit" \
	  && cp -a "$$tmp/toolkit/extensions/sd-forge-krea2-moodboard" "$$ext_root/" \
	  && cp -a "$$tmp/toolkit/extensions/sd-forge-krea2-edit" "$$ext_root/" \
	  && python3 -c "from pathlib import Path; p=Path('$$ext_root')/'sd-forge-krea2-edit/scripts/krea2_edit.py'; s=p.read_text(); s2=s.replace('dynamic_args.pop(\"ref_boosts\", None)','dynamic_args[\"ref_boosts\"] = []').replace('dynamic_args.pop(\"ref_fit\", None)','dynamic_args[\"ref_fit\"] = []'); assert s2!=s, 'no se encontró dynamic_args.pop en krea2_edit.py'; p.write_text(s2)" \
	  && echo "Instaladas en $$ext_root:" \
	  && echo "  - sd-forge-krea2-moodboard" \
	  && echo "  - sd-forge-krea2-edit (fix dynamic_args.pop aplicado)" \
	  && echo "Reinicia la WebUI: make run krea2 (o make restart). Requiere imagen con el backend patch (make build)."

# Actualiza el vendor de Krea2 Depth/Pose ControlNet-LoRA (extensions-builtin de la imagen).
# Tras correrlo: make build && make restart. El peso ~862 MB no se descarga aquí.
KREA2_DEPTH_REF ?= forge-classic-2.28.1
KREA2_DEPTH_REPO ?= https://github.com/fabiencomte/Krea-2-controlnet.git
krea2-depth-ext:
	@set -e; \
	dest="$(CURDIR)/builtin-extensions/sd-forge-krea2-depth-controlnet"; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	echo "Clonando $(KREA2_DEPTH_REPO) @ $(KREA2_DEPTH_REF)…"; \
	git clone --depth 1 -b "$(KREA2_DEPTH_REF)" "$(KREA2_DEPTH_REPO)" "$$tmp/src"; \
	ref=$$(git -C "$$tmp/src" rev-parse HEAD); \
	rm -rf "$$dest"; \
	mkdir -p "$$dest"; \
	cp -a "$$tmp/src/scripts" "$$tmp/src/forge_krea2_depth" "$$tmp/src/install.py" "$$tmp/src/THIRD_PARTY_NOTICES.md" "$$dest/"; \
	printf '%s\n' \
	  '# Krea 2 Depth / Pose ControlNet-LoRA (Forge)' \
	  '' \
	  "Vendorado desde $(KREA2_DEPTH_REPO)" \
	  "rama \`$(KREA2_DEPTH_REF)\` @ \`$$ref\`." \
	  '' \
	  'Va en **extensions-builtin** de la imagen Docker. Modelo en:' \
	  '\`$$DATA_PATH/Models/ControlNet/Krea2/depth-control-lora.safetensors\`' \
	  '' \
	  'Tras actualizar: `make build` && `make restart`.' \
	  > "$$dest/README.md"; \
	echo "Actualizado $$dest @ $$ref"; \
	echo "Siguiente: make build && make restart"

# Repara deps de ReActor en el contenedor en marcha (sin rebuild).
# insightface instala onnxruntime CPU y rompe CUDAExecutionProvider; este target lo deshace.
# Permanente en imagen: make build (Dockerfile ya incluye el mismo arreglo).
# Extensión recomendada: https://codeberg.org/Gourieff/sd-webui-reactor (no el fork -sfw de GitHub).
reactor-fix:
	@docker exec -i $(SERVICE) sh < "$(CURDIR)/scripts/reactor_fix_deps.sh"
	@echo "Listo. Si la WebUI ya estaba arriba, reinicia: make restart"

ps:
	$(COMPOSE) ps

clean: down
	docker rmi forge-neo:latest 2>/dev/null || true

# Construye la imagen, la etiqueta con REGISTRY_IMAGE y la sube. Requiere docker login al registro.
push: build
	docker tag forge-neo:latest $(REGISTRY_IMAGE)
	docker push $(REGISTRY_IMAGE)

# Variante CUDA 12: construir, etiquetar y subir (para RunPod con driver que no soporta CUDA 13)
push-cuda12: build-cuda12
	docker tag forge-neo:cuda12 $(REGISTRY_IMAGE_CUDA12)
	docker push $(REGISTRY_IMAGE_CUDA12)

# Variante slim: subir como :slim (para RunPod cuando la imagen completa excede el límite)
push-slim: build-slim
	$(eval slim_image := $(patsubst %:latest,%:slim,$(REGISTRY_IMAGE)))
	docker tag forge-neo:slim $(slim_image)
	docker push $(slim_image)

# Variante Sage: subir como :sage
push-sage: build-sage
	$(eval sage_image := $(patsubst %:latest,%:sage,$(REGISTRY_IMAGE)))
	docker tag $(FORGE_IMAGE_SAGE) $(sage_image)
	docker push $(sage_image)

# Instala Docker Engine y Docker Compose (plugin) desde el repo oficial. Solo Ubuntu/Debian.
# Usa la última versión estable del repo; para fijar: make install-docker DOCKER_CE_VERSION=5:29.2.1-1~ubuntu.24.04~noble
install-docker:
	@. /etc/os-release 2>/dev/null && ([ "$$ID" = "ubuntu" ] || [ "$$ID" = "debian" ]) || (echo "Solo soportado Ubuntu/Debian. Ver https://docs.docker.com/engine/install/" && exit 1)
	@echo "Quitando paquetes que puedan conflictuar..."
	@sudo apt-get remove -y docker.io docker-doc docker-compose docker-compose-v2 podman-docker 2>/dev/null || true
	@sudo apt-get update
	@sudo apt-get install -y ca-certificates curl
	@sudo install -m 0755 -d /etc/apt/keyrings
	@id=$$(. /etc/os-release && echo "$$ID"); \
	suite=$$(. /etc/os-release && echo "$${UBUNTU_CODENAME:-$$VERSION_CODENAME}"); \
	if [ "$$id" = "ubuntu" ]; then \
	  gpg_url="https://download.docker.com/linux/ubuntu/gpg"; \
	  repo_url="https://download.docker.com/linux/ubuntu"; \
	elif [ "$$id" = "debian" ]; then \
	  gpg_url="https://download.docker.com/linux/debian/gpg"; \
	  repo_url="https://download.docker.com/linux/debian"; \
	else exit 1; fi; \
	sudo curl -fsSL "$$gpg_url" -o /etc/apt/keyrings/docker.asc; \
	sudo chmod a+r /etc/apt/keyrings/docker.asc; \
	printf 'Types: deb\nURIs: %s\nSuites: %s\nComponents: stable\nSigned-By: /etc/apt/keyrings/docker.asc\n' "$$repo_url" "$$suite" | sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null
	@sudo apt-get update
	@pkgs="docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"; \
	if [ -n "$(DOCKER_CE_VERSION)" ]; then \
	  pkgs="docker-ce=$(DOCKER_CE_VERSION) docker-ce-cli=$(DOCKER_CE_VERSION) containerd.io docker-buildx-plugin docker-compose-plugin"; \
	elif [ -n "$(DOCKER_COMPOSE_PLUGIN_VERSION)" ]; then \
	  pkgs="docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin=$(DOCKER_COMPOSE_PLUGIN_VERSION)"; \
	fi; \
	sudo apt-get install -y $$pkgs
	@(sudo systemctl enable docker 2>/dev/null && sudo systemctl start docker 2>/dev/null) || \
	  (sudo service docker start 2>/dev/null) || true
	@if ! sudo docker info >/dev/null 2>&1; then \
	  echo "Docker no está en marcha (entorno sin systemd, p. ej. WSL). Arranca el daemon:"; \
	  echo "  sudo dockerd &"; \
	  echo "  o en WSL2: configura systemd o ejecuta dockerd en segundo plano."; \
	fi
	@echo "---"; sudo docker --version; sudo docker compose version
