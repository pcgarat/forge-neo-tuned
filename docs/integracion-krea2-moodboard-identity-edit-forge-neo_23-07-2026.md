# Última modificación: 2026-07-30

# Integración de Krea 2 Moodboard + Identity Edit en docker-neo

Análisis de cómo meter el toolkit de [RedNodeAI/forge-neo-krea2-toolkit](https://github.com/RedNodeAI/forge-neo-krea2-toolkit) (publicado también en [Civitai](https://civitai.com/models/2794961/krea-2-moodboard-identity-edit-comfyui-nodes-forge-neo)) en esta imagen Docker de Forge Neo.

**Ámbito:** solo la variante **Forge Neo** (`Krea2-ForgeNeo-Toolkit`). El zip/nodes de ComfyUI (`ComfyUI-Krea2Moodboard`) **no aplica** a este repo.

**Estado (2026-07-23):** implementado en repo — patch en imagen (`FORGE_NEO_REF` pin), `make krea2-ext`, docs. Checklist: [docs/checklists/integracion-krea2-moodboard-identity-edit_23-07-2026.md](checklists/integracion-krea2-moodboard-identity-edit_23-07-2026.md). Pendiente smoke test en GPU.

---

## 1. Qué es (y qué no es)

| Pieza | Rol |
|-------|-----|
| Moodboard | Transferencia de estilo/vibe con referencias (sin LoRA). |
| Identity Edit | Edición con identidad preservada vía LoRA `krea2_edit` + conditioning dual. |
| Backend patch | Activa el path nativo Qwen3-VL de Neo (deepstack + interleaved mrope) y los hooks que consumen las extensiones. |
| Extensiones UI | Dos accordions en txt2img/img2img + Settings → Krea2 Moodboard. |

No es un wrapper de ComfyUI. No añade nodos. Todo vive en la UI normal de Forge.

**Crítica:** el “plug and play” del README oficial asume un Forge Neo instalado en disco con historial git. Aquí el WebUI está **horneado en la imagen** y el `Dockerfile` hace `rm -rf .git`. Eso cambia el procedimiento: el patch no se aplica “una vez en el host”; hay que decidir **dónde** y **cuándo** se aplica.

---

## 2. Fuentes oficiales

| Recurso | URL |
|---------|-----|
| Repo Forge Neo toolkit | https://github.com/RedNodeAI/forge-neo-krea2-toolkit |
| Install oficial | https://github.com/RedNodeAI/forge-neo-krea2-toolkit/blob/main/INSTALL.md |
| Civitai (zips + showcase) | https://civitai.com/models/2794961 |
| Hermano ComfyUI (no usar aquí) | https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard |
| Text encoder (visión) | https://huggingface.co/Comfy-Org/Krea-2 |
| LoRA Identity Edit | https://civitai.com/models/2761113 · https://huggingface.co/conradlocke/krea2-identity-edit |
| Forge Neo (rama `neo`) | https://github.com/Haoming02/sd-webui-forge-classic/tree/neo |

Contenido del toolkit (GitHub):

```
forge-neo-krea2-toolkit/
├── INSTALL.md
├── README.md
├── krea2-features-backend.patch   # ~680 líneas, 4 ficheros
└── extensions/
    ├── sd-forge-krea2-moodboard/
    └── sd-forge-krea2-edit/
```

El patch toca **solo** rutas K2/Qwen3-VL:

1. `backend/diffusion_engine/krea.py`
2. `backend/nn/krea.py`
3. `backend/nn/llm/llama.py`
4. `backend/text_processing/qwen3vl_engine.py`

Otros modelos no deberían verse afectados si el patch aplica limpio.

---

## 3. Cómo encaja con la arquitectura de docker-neo

```
┌─────────────────────────────────────────────────────────────┐
│ Imagen Docker (/app/webui)                                  │
│  - Código Forge Neo (clonado en build)                      │
│  - ★ AQUÍ debe vivir el backend patch                       │
│  - Deps Python / CUDA                                       │
└─────────────────────────────────────────────────────────────┘
         │
         │  volúmenes
         ▼
┌─────────────────────────────────────────────────────────────┐
│ DATA_PATH → /data                                           │
│  models/Stable-diffusion/   ← checkpoint Krea 2             │
│  models/text_encoder/       ← qwen3vl_4b_* (visión)         │
│  models/Lora/               ← krea2_identity_edit           │
│  models/VAE/                ← si hace falta                 │
│  config.json, output/, …                                    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ EXTENSIONS_PATH → /data/extensions                          │
│  sd-forge-krea2-moodboard/   ← copy / clone                 │
│  sd-forge-krea2-edit/                                       │
└─────────────────────────────────────────────────────────────┘
```

| Componente | Persistencia | ¿Rebuild imagen? |
|------------|--------------|------------------|
| Backend patch | Dentro de `/app/webui` | **Sí** |
| Extensiones | Volumen `EXTENSIONS_PATH` | No |
| Modelos / LoRA / TE | Volumen `DATA_PATH` | No |

**Conclusión de diseño:** separar **patch (imagen)** de **extensiones + pesos (volúmenes)**. Mezclarlos (p. ej. aplicar el patch solo en el volumen) es un antipatrón: el volumen no monta `backend/`.

---

## 4. Instalación oficial (referencia)

Del [INSTALL.md](https://github.com/RedNodeAI/forge-neo-krea2-toolkit/blob/main/INSTALL.md), en un Forge Neo nativo:

1. Desde la raíz de Forge: `git apply --verbose path/to/krea2-features-backend.patch`
2. Copiar `extensions/sd-forge-krea2-*` a `<forge>/extensions/`
3. Text encoder visión en `models/text_encoder/` y seleccionarlo con el checkpoint K2
4. LoRA identity edit a strength **1.0** (solo Identity Edit)
5. Reiniciar WebUI → accordions en txt2img/img2img; Settings → Krea2 Moodboard

Requisito de versión: **Forge Neo rama `neo`, builds ~julio 2026+**. Si `git apply` falla, el Neo es más viejo o más nuevo que el bundle → hay que buscar release más reciente o aplicar hunks a mano.

---

## 5. Plan de integración en docker-neo (recomendado)

### 5.1 Opción A — Patch en build (preferida)

Ventajas: reproducible, misma imagen en local/RunPod, no depende de estado del volumen.

Esqueleto orientativo para el `Dockerfile` (y el mismo patrón en `Dockerfile.cuda12`):

```dockerfile
# Tras el clone de Forge Neo, ANTES de borrar .git (o usar patch -p1 sin git):
WORKDIR /app/webui
COPY patches/krea2-features-backend.patch /tmp/krea2-features-backend.patch
RUN git apply --verbose /tmp/krea2-features-backend.patch \
    && rm -f /tmp/krea2-features-backend.patch \
    && rm -rf .git
```

Alternativa sin historial git (si se borra `.git` antes):

```dockerfile
RUN patch -p1 < /tmp/krea2-features-backend.patch
```

`patch -p1` suele bastar porque el diff es unificado estándar; `git apply` es lo que documenta el autor.

**Vendor del patch:** clonar/copiar `krea2-features-backend.patch` a algo como `patches/` en este repo (pin por commit/tag del toolkit). No descargar en caliente en cada build sin pin: un cambio upstream puede romper el build sin aviso.

**Crítica — pin de Forge Neo:** hoy el Dockerfile hace:

```dockerfile
git clone --depth 1 --branch neo https://github.com/Haoming02/sd-webui-forge-classic webui
```

Eso es “último commit de `neo`”. El toolkit está testado contra un Neo de julio 2026 concreto. Si Haoming02 mueve esos 4 ficheros, el build rompe. Mejor:

```dockerfile
ARG FORGE_NEO_REF=<sha-o-tag-compatible-con-el-patch>
RUN git clone --branch neo https://github.com/Haoming02/sd-webui-forge-classic webui \
    && cd webui && git checkout "$FORGE_NEO_REF"
```

Trade-off: builds más estables vs. quedarse atrás en fixes de Neo. Para un feature que **parchea el core**, estabilidad gana.

### 5.2 Extensiones en el volumen (sin rebuild)

Con el contenedor parado o vía `make shell`:

```bash
# En el HOST, donde apunte EXTENSIONS_PATH (.env)
EXT="${EXTENSIONS_PATH:-/workspace/forge-data/extensions}"
git clone --depth 1 https://github.com/RedNodeAI/forge-neo-krea2-toolkit /tmp/krea2-toolkit
cp -a /tmp/krea2-toolkit/extensions/sd-forge-krea2-moodboard "$EXT/"
cp -a /tmp/krea2-toolkit/extensions/sd-forge-krea2-edit "$EXT/"
```

O descargar solo el zip `Krea2-ForgeNeo-Toolkit-*.zip` de Civitai y copiar las dos carpetas.

Reiniciar: `make restart` (o reinicio de WebUI desde la UI).

**No** hace falta meter las extensiones en la imagen salvo que quieras una imagen “lista para RunPod sin pasos manuales”. En ese caso, `COPY` a `/app/webui/extensions/` **no** sirve si Compose monta `EXTENSIONS_PATH` encima de `/data/extensions`: Forge usa `data_path/extensions`. Hay que copiar al volumen o documentar el clone post-arranque.

Objetivo Make opcional (`make krea2-ext`) tendría sentido: clonar/actualizar las dos extensiones en `EXTENSIONS_PATH`, análogo a `make iib-access`.

### 5.3 Modelos (volumen `DATA_PATH`)

Rutas esperadas con el `.env` por defecto:

| Asset | Destino en host | Notas |
|-------|-----------------|-------|
| Checkpoint Krea 2 | `$DATA_PATH/models/Stable-diffusion/` | Turbo vs Raw según receta |
| `qwen3vl_4b_bf16.safetensors` (o `fp8_scaled`) | `$DATA_PATH/models/text_encoder/` | **Variante VISIÓN** (Comfy-Org/Krea-2). Log: `Detected Qwen3-VL-4B (vision) text encoder` |
| LoRA identity edit | `$DATA_PATH/models/Lora/` | Strength **1.0**; v1.2 recomendada |
| VAE Krea 2 | `$DATA_PATH/models/VAE/` | Según cómo empaquetes el checkpoint |

Sin el TE de visión, Moodboard e Identity Edit no tienen sentido: las extensiones armán estado que el engine solo consume si `moodboard_available` (torre `visual` presente).

### 5.4 Opción B — Patch en runtime (desaconsejada)

Aplicar el patch en `entrypoint.sh` sobre `/app/webui` en cada arranque:

- Capa writable del contenedor se pierde al recrear el contenedor → hay que reaplicar siempre.
- Imagen “oficial” queda divergente del runtime → debugging confuso.
- Fallos de `patch` a mitad de arranque = WebUI rota.

Solo tendría sentido como puente temporal mientras no se toca el Dockerfile.

### 5.5 Opción C — Solo extensiones sin patch

**No funciona.** Las extensiones “arman” moodboard/edit; el engine parcheado es quien las consume. Sin patch: UI visible o no, la feature real no está.

---

## 6. Uso tras integrar (resumen operativo)

### Moodboard (sin LoRA)

1. Cargar checkpoint K2 + TE visión.
2. Accordion **Krea2 Moodboard** → enable → 1–10 refs.
3. Receta “Krea vibe” del autor: extract **style**, strength **0.5**, **fine tiles 4×4**, directive on, position **after**.

### Identity Edit (con LoRA)

1. LoRA `krea2_*` a strength **1.0**.
2. Accordion **Krea2 Identity Edit** → imagen fuente + instrucción en el prompt.
3. Sampler: **Euler / Simple**. Turbo 8 steps CFG 1 (v1.2: 8–12). Removals: Raw 20–40 CFG 3. ≤2MP.
4. Con LoRA v1.2: AR **fit source to output**, `ref_boost` 2–6, `grounding_px` ~768 (1024+ personas).

### Fusión

Activar **ambos** accordions: identidad del edit source + estilo del moodboard. Receta moodboard sugerida para fusión: style, full image, **indirect ON**, directive ON.

Settings persistentes del moodboard: **Settings → Krea2 Moodboard**.

---

## 7. Riesgos y fricción con este proyecto

| Riesgo | Por qué importa aquí | Mitigación |
|--------|----------------------|------------|
| Patch frágil vs `neo` HEAD | Clone `--depth 1` sin pin | Pin `FORGE_NEO_REF` + pin del patch |
| `.git` borrado | `git apply` “oficial” no aplica tal cual | Aplicar **antes** de `rm -rf .git`, o usar `patch -p1` |
| VRAM | TE visión (~9 GB bf16) + K2 + LoRA + refs | En 8 GB (`make krea2 VRAM=8gb`) es **muy justo / inviable** en bf16; probar `fp8_scaled` del TE y perfil lowvram; Moodboard solo ya es pesado |
| Extensiones en volumen vacío | RunPod nuevo = sin carpetas | Documentar `make krea2-ext` o bake + sync al volumen |
| Confundir packs Civitai | Zip ComfyUI ≠ Forge Neo | Usar solo `Krea2-ForgeNeo-Toolkit` o el repo GitHub |
| Licencias | Toolkit AGPL-3.0 (como Neo); pesos Krea/LoRA aparte | No redistribuir pesos en la imagen |
| Auto face-ref prep | 2 generaciones anidadas + caché en `ref_cache/` de la extensión | El caché debe vivir en el volumen de extensiones (persistente) |

**Crítica VRAM:** los presets están pensados para Flux/Klein/Krea2 en ≤12 GB. Krea2 + Qwen3-VL visión es el perfil más exigente de memoria. No asumas que `ARGS_8GB` sirva igual; documenta `make krea2 VRAM=8gb` como el extremo ajustado o avisa que Moodboard/Edit piden ≥12–16 GB de forma realista.

---

## 8. Checklist de implementación

Checklist vivo: [docs/checklists/integracion-krea2-moodboard-identity-edit_23-07-2026.md](checklists/integracion-krea2-moodboard-identity-edit_23-07-2026.md).

### Hecho en repo

- [x] Vendorizar `krea2-features-backend.patch` en `patches/` (regenerado; original no aplicaba en neo HEAD)
- [x] Aplicar patch en `Dockerfile` y `Dockerfile.cuda12` (slim incluido vía mismo Dockerfile)
- [x] Pin `FORGE_NEO_REF=97ff3a4024be2f0d5316f16e868e5ef822768872`
- [x] `make krea2-ext`
- [x] README raíz + este doc
- [x] Imagen GHCR llevará el patch por defecto (va en el Dockerfile de build/push)

### Pendiente (GPU / datos de usuario)

- [ ] `make build` / `make build-cuda12` y verificar apply en build
- [ ] `make krea2-ext`
- [ ] Colocar TE visión + checkpoint K2 (+ LoRA si Identity Edit)
- [ ] Log `Detected Qwen3-VL-4B (vision) text encoder`
- [ ] Smoke test Moodboard e Identity Edit

**Nota de regeneración:** el patch oficial del toolkit fallaba en `backend/diffusion_engine/krea.py` (import `PredictionDiscreteFlow` eliminado upstream). Se regeneró por cherry-pick 3 vías desde `44ae1a4` y se corrigió `dynamic_args.pop()` incompatible con la metaclase actual. Detalle: `patches/README.md`.

---

## 9. Alternativa: no integrar en la imagen

Si solo quieres probar el toolkit:

1. Usar Forge Neo **nativo** (no Docker) y seguir el INSTALL.md al pie de la letra.
2. O montar un bind del código WebUI editable (hoy **no** es el diseño de docker-neo: el código va en imagen).

Forzar el patch fuera de Docker en esta arquitectura es pelear contra el diseño. Si el objetivo es “mi stack dockerizado tiene Moodboard/Edit”, el patch va en el Dockerfile.

---

## 10. Veredicto

| Pregunta | Respuesta |
|----------|-----------|
| ¿Se puede integrar? | Sí. |
| ¿Es solo copiar extensiones? | **No.** El patch de backend es obligatorio. |
| ¿Dónde va el patch? | En el **build** de la imagen (`/app/webui`). |
| ¿Dónde van las extensiones? | En **`EXTENSIONS_PATH`** (`/data/extensions`). |
| ¿Dónde van los pesos? | En **`DATA_PATH`/models/...**. |
| ¿Riesgo principal? | Desalineación patch ↔ commit de Forge Neo + VRAM del TE visión. |
| ¿ComfyUI pack? | Irrelevante para este repo. |

Siguiente paso: `make build` + `make krea2-ext` + modelos en `DATA_PATH` + smoke test en GPU.
