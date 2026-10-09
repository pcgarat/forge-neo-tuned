# sd-webui-forge-neo — objetivos para build y ejecución
# Uso: make [objetivo]; make help

COMPOSE  = docker compose
SERVICE  = forge-neo
PORT     = 7860

# Versiones para install-docker (repo estable de Docker). Vacío = última del repo.
# Para fijar: DOCKER_CE_VERSION=5:29.2.1-1~ubuntu.24.04~noble (ejemplo Ubuntu 24.04).
DOCKER_CE_VERSION ?=
DOCKER_COMPOSE_PLUGIN_VERSION ?=

# Imagen para push a registro. Ej.: make push REGISTRY_IMAGE=ghcr.io/pcgarat/forge-neo:latest
REGISTRY_IMAGE ?= ghcr.io/$(GITHUB_USER)/forge-neo:latest
REGISTRY_IMAGE_CUDA12 ?= ghcr.io/$(GITHUB_USER)/forge-neo:cuda12
GITHUB_USER ?= pcgarat

# Perfil GPU ≤12 GB (p. ej. RTX 4060 8 GB): Klein 9B, Flux, Qwen y modelos grandes.
# cuda-malloc + lowvram + fp8 + offload RAM. Sin --fast-fp8 (falla en Krea2 y solo ralentiza).
ARGS_8GB = --cuda-malloc --lowvram --fp8_e4m3fn-unet --reserve-vram 2 --pin-shared-memory --mmap-torch-files

# Perfil 8 GB + atención INT8 (Comfy-Kitchen), que sustituye a flash_attn.
# Solo rinde cuando la secuencia de atención es larga: vídeo (Wan) y alta resolución.
# Medido en Krea2 a 1280px: 26,3s → 21,9s; a 768px no hay diferencia medible.
# Ojo: a igual semilla produce una imagen distinta, no es intercambiable a mitad de un trabajo.
ARGS_INT8_ATTN = $(ARGS_8GB) --use-ck-attention

# Perfil 8 GB + SageAttention forzado a FP16 (el path FP8 por defecto es inestable en sm89).
# Requiere la imagen horneada con Sage (make build-sage); no basta con flags de runtime.
# fp16_cuda usa los kernels CUDA precompilados (_qattn_sm89); fp16_triton usa los Triton.
# Nota: Sage en sm89 no tiene kernel f16 con acumulador f16 -> `sageattn_qk_int8_pv_fp16_cuda`
# con pv_accum_dtype="fp16" cae a `sm80_compile` en tiempo de import; puede fallar en sm89.
ARGS_SAGE = $(ARGS_8GB) --sage-function fp16_cuda
ARGS_SAGE_TRITON = $(ARGS_8GB) --sage-function fp16_triton

# Imagen a arrancar: por defecto la estándar. `make sage` usa la variante con SageAttention.
FORGE_IMAGE ?= forge-neo:latest
FORGE_IMAGE_SAGE ?= forge-neo:sage

.PHONY: help build build-no-cache build-cuda12 build-slim build-sage push push-cuda12 push-slim push-sage up down restart logs shell workspace preflight-gpu preflight-gpu-check seed-extensions ps clean klein9b lowvram wan ck sage sage-triton chatbot chatbot-warmup test-warmup bench-attn bench-attn-sweep bench-attn-flash bench-attn-ck bench-attn-sage bench-attn-sage-triton test-bench iib-access krea2-ext krea2-depth-ext reactor-fix install-docker

help:
	@echo "sd-webui-forge-neo — objetivos disponibles:"
	@echo ""
	@echo "  make build         — Construir la imagen (primera vez o tras cambios)"
	@echo "  make build-no-cache — Reconstruir sin caché (entrypoint, fixes config.json, etc.)"
	@echo "  make up            — Arrancar el contenedor y seguir logs (Ctrl+C para salir)"
	@echo "  make down          — Parar y eliminar el contenedor"
	@echo "  make restart       — down + up y seguir logs (Ctrl+C para salir)"
	@echo "  make klein9b       — Arrancar optimizado para Klein 9B y seguir logs (Ctrl+C para salir)"
	@echo "  make lowvram       — Arrancar con perfil 8 GB y seguir logs (Ctrl+C para salir)"
	@echo "  make wan           — Perfil 8 GB + atención INT8 de Comfy-Kitchen (vídeo Wan / alta resolución)"
	@echo "  make ck            — Alias de 'make wan' (atención INT8 de Comfy-Kitchen, sin build)"
	@echo "  make sage          — Perfil 8 GB + SageAttention fp16_cuda (requiere 'make build-sage' antes)"
	@echo "  make sage-triton   — Igual que 'make sage' pero con el kernel Triton (fp16_triton)"
	@echo "  make chatbot       — Perfil 8 GB + warmup torch.compile (guard_filter_fn) al size del último gen"
	@echo "  make chatbot-warmup — Solo warmup (Forge ya tiene que estar arriba)"
	@echo "  make test-warmup   — Tests del parser/payload de chatbot-warmup"
	@echo "  make bench-attn    — Medir N gens (semilla fija) del backend activo y anexar fila al informe"
	@echo "  make bench-attn-sweep — Comparar base (flash) vs CK INT8 reiniciando y midiendo (requiere GPU)"
	@echo "  make bench-attn-sage — Mide SageAttention fp16_cuda (requiere 'make build-sage')"
	@echo "  make bench-attn-sage-triton — Mide SageAttention fp16_triton"
	@echo "  make test-bench    — Tests del benchmark de atención"
	@echo "  make logs          — Ver logs del servicio (Ctrl+C para salir)"
	@echo "  make shell         — Abrir una shell dentro del contenedor"
	@echo "  make workspace     — Crear árbol de datos y sembrar extensions/ si faltan (antes del primer up)"
	@echo "  make preflight-gpu — Verificar driver NVIDIA y socket de nvidia-persistenced en el host (antes de up)"
	@echo "  make seed-extensions — Copiar repo/extensions → EXTENSIONS_PATH solo si falta cada carpeta"
	@echo "  make iib-access   — Crear .env en la extensión IIB con acceso a carpetas de salida (/data/output, /data/Images)"
	@echo "  make krea2-ext    — Forzar actualización Krea2 Moodboard + Identity Edit desde GitHub"
	@echo "  make krea2-depth-ext — Actualizar Depth/Pose ControlNet-LoRA (builtin) desde GitHub; luego make build"
	@echo "  make reactor-fix  — Reparar deps ReActor (onnxruntime-gpu vs CPU) en contenedor en marcha"
	@echo "  make ps            — Estado del servicio"
	@echo "  make clean         — down y eliminar imagen local"
	@echo "  make install-docker — Instalar Docker Engine y Docker Compose (plugin) desde repo oficial (Ubuntu/Debian, requiere sudo)"
	@echo "  make push           — Construir imagen, etiquetar y subir a registro (REGISTRY_IMAGE)"
	@echo "  make build-cuda12   — Construir variante CUDA 12.4 (para RunPod con driver < CUDA 13)"
	@echo "  make push-cuda12   — Construir variante CUDA 12, etiquetar y subir (REGISTRY_IMAGE_CUDA12)"
	@echo "  make build-slim    — Construir variante slim (sin onnxruntime-gpu/nunchaku, menos tamaño para RunPod)"
	@echo "  make push-slim     — Construir slim, etiquetar y subir (REGISTRY_IMAGE, tag :slim)"
	@echo "  make build-sage    — Construir imagen estándar + SageAttention horneado (tag :sage; no pisa :latest)"
	@echo "  make push-sage     — Construir variante Sage, etiquetar y subir (REGISTRY_IMAGE, tag :sage)"
	@echo ""
	@echo "WebUI: http://localhost:$(PORT)   API: http://localhost:$(PORT)/docs"

# Cargar .env y exportar para que compose use DATA_PATH y EXTENSIONS_PATH en los volúmenes
ENV_LOAD = set -a && [ -f .env ] && . ./.env && set +a

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
# No pisa :latest; arranca con `make sage`.
build-sage:
	DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_SAGE=1 -t $(FORGE_IMAGE_SAGE) .

up: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@$(ENV_LOAD) && $(COMPOSE) up -d
	@$(if $(BENCH_DETACH),,@$(MAKE) logs)

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

# Flux 2 Klein 9B: según VRAM se aplican flags de memoria y precisión
klein9b: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@v=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); \
	if [ -z "$$v" ]; then \
	  echo "No se detectó nvidia-smi; usando perfil 8GB"; \
	  extra="$(ARGS_8GB)"; \
	elif [ "$$v" -ge 20000 ]; then \
	  echo "VRAM $$v MB: --highvram --bf16-unet"; \
	  extra="--cuda-malloc --highvram --bf16-unet --pin-shared-memory --mmap-torch-files"; \
	elif [ "$$v" -ge 12000 ]; then \
	  echo "VRAM $$v MB: --normalvram --bf16-unet"; \
	  extra="--cuda-malloc --normalvram --bf16-unet --pin-shared-memory --mmap-torch-files"; \
	else \
	  echo "VRAM $$v MB: perfil 8GB ($(ARGS_8GB))"; \
	  extra="$(ARGS_8GB)"; \
	fi; \
	$(ENV_LOAD) && export EXTRA_ARGS="$$extra" && $(COMPOSE) up -d && $(MAKE) logs

# Perfil fijo 8 GB (sin autodetección); útil para RTX 4060 / 3060 12GB límite, etc.
lowvram: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil 8GB: $(ARGS_8GB)"
	@$(ENV_LOAD) && export EXTRA_ARGS="$(ARGS_8GB)" && $(COMPOSE) up -d
	@$(if $(BENCH_DETACH),,@$(MAKE) logs)

wan: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil 8GB + atención INT8: $(ARGS_INT8_ATTN)"
	@$(ENV_LOAD) && export EXTRA_ARGS="$(ARGS_INT8_ATTN)" && $(COMPOSE) up -d
	@$(if $(BENCH_DETACH),,@$(MAKE) logs)

# Atención INT8 de Comfy-Kitchen. Mismo flag que `make wan`; alias con nombre explícito.
# No requiere rebuild: el módulo comfy_kitchen ya viene en la imagen.
ck: wan

# SageAttention fp16_cuda. Requiere la imagen con Sage (`make build-sage`).
# Fuerza FP16: en sm89 el path FP8 por defecto (`auto`) es inestable.
sage: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil 8GB + SageAttention fp16_cuda: $(ARGS_SAGE)"
	@$(ENV_LOAD) && export FORGE_IMAGE="$(FORGE_IMAGE_SAGE)" EXTRA_ARGS="$(ARGS_SAGE)" && $(COMPOSE) up -d
	@$(if $(BENCH_DETACH),,@$(MAKE) logs)

# Sage por Triton (kernels Triton, sin depender de los CUDA precompilados). Solo lo usa Triton,
# que ya viene en la imagen; probar si el kernel CUDA fp16 diera problemas en sm89.
sage-triton: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil 8GB + SageAttention fp16_triton: $(ARGS_SAGE_TRITON)"
	@$(ENV_LOAD) && export FORGE_IMAGE="$(FORGE_IMAGE_SAGE)" EXTRA_ARGS="$(ARGS_SAGE_TRITON)" && $(COMPOSE) up -d
	@$(if $(BENCH_DETACH),,@$(MAKE) logs)

# Llamadas API del chatBot: mismo modelo/size, N escenas seguidas.
# 8 GB + compile guard_filter_fn (compatible con --cuda-malloc; max-autotune no lo es).
# El warmup es 1 step al size de params.txt; no pisa el último gen (save_images=false + restaura params.txt).
chatbot: workspace
	@$(MAKE) --no-print-directory preflight-gpu
	@echo "Perfil chatBot 8GB: $(ARGS_8GB)"
	@$(ENV_LOAD) && export EXTRA_ARGS="$(ARGS_8GB)" && $(COMPOSE) up -d && $(MAKE) chatbot-warmup && $(MAKE) logs

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

# Compara base (flash) vs CK INT8. Requiere GPU operativa (nvidia-smi). Intervalo configurable.
BENCH_SLEEP ?= 30
bench-attn-sweep: bench-attn-flash bench-attn-ck

bench-attn-flash:
	@$(MAKE) --no-print-directory lowvram BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label flash"

bench-attn-ck:
	@$(MAKE) --no-print-directory ck BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label ck-int8"

bench-attn-sage: build-sage
	@$(MAKE) --no-print-directory sage BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label sage-fp16_cuda"

bench-attn-sage-triton: build-sage
	@$(MAKE) --no-print-directory sage-triton BENCH_DETACH=1
	@sleep $(BENCH_SLEEP)
	@$(MAKE) --no-print-directory bench-attn BENCH_ARGS="$(BENCH_ARGS) --label sage-fp16_triton"

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
	  && echo "Reinicia la WebUI (make restart). Requiere imagen con el backend patch (make build)."

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
