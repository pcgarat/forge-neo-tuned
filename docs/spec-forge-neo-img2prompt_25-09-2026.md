# Última modificación: 2026-09-25

# Spec: Forge Neo — Image → Prompt (Krea 2 v1)

## Forge Neo — cómo se crea una extensión (hallazgos)

Mecanismo heredado de A1111 / Forge Classic (`Haoming02/sd-webui-forge-classic` rama `neo`):

| Pieza | Rol |
|-------|-----|
| Carpeta en `extensions/` o `extensions-builtin/` | Unidad instalable; el nombre de carpeta es el id por defecto |
| `scripts/*.py` | El loader importa todo lo que hay aquí. Suele contener una subclase de `modules.scripts.Script` **o** registrar `script_callbacks` |
| Paquete hermano en la raíz (p. ej. `forge_krea2_depth/`) | Lógica fuera del script; patrón usado por Depth ControlNet en este repo |
| `install.py` (raíz) | Se ejecuta al arranque vía `run_extension_installer` **salvo** `--skip-install` |
| `metadata.ini` | Opcional: `Name`, orden `Before`/`After` entre extensiones |
| `javascript/`, `style.css`, `preload.py` | Opcionales |

**Dos formas de UI:**

1. **`scripts.Script` + `show() → AlwaysVisible`** — accordion dentro de txt2img/img2img (Moodboard, Identity Edit, Depth). Ideal si necesitas el **stack ya seleccionado** en esa pestaña.
2. **`script_callbacks.on_ui_tabs`** — pestaña top-level propia (p. ej. [Adeliox/forge-neo-image2prompt](https://github.com/Adeliox/forge-neo-image2prompt)). Mejor para tools pesados de VRAM; el “send to prompt” suele hacerse con JS sobre `#txt2img_prompt textarea`.

**Instalación en la práctica:**

- UI: Extensions → Install from URL (repo git) → Apply and restart UI.
- CLI: `git clone <url> $EXTENSIONS_PATH/<nombre>`.
- En `docker-neo`: `make seed-extensions` / `make up` copia `repo/extensions/*` → `EXTENSIONS_PATH` **solo si no existe**; la imagen arranca con **`--skip-install`**, así que deps de `install.py` **no** se instalan en runtime — hay que hornearlas en la imagen o no depender de ellas.

**Decisión v1 (este spec):** forma (1) AlwaysVisible, sin `install.py`, sin deps extra. No forkear Adeliox (carga VL propio + estilos tag-soup; distinto objetivo). Rellenar el prompt del tab activo con JS Gradio (`#txt2img_prompt` / `#img2img_prompt`), patrón send-to ya usado en el ecosistema Neo.

---

## Objective

Extensión de **Forge Neo** (formato WebUI estándar) que, a partir de una imagen (upload o pegado desde portapapeles en el control Gradio), produce un **prompt en prosa** listo para pegar en txt2img, afinado al stack **Krea 2** actualmente seleccionado (checkpoint + text encoder + turbo vs RAW).

**Instalación:** igual que el resto de extensiones — **Extensions → Install from URL**, o `git clone` en `extensions/` / `EXTENSIONS_PATH`. Tras instalar: **Apply and restart UI**. No requiere pasos Docker especiales ni ir en la imagen como builtin.

**Usuario:** operador local de Forge Neo (incl. `docker-neo`).

**Por qué:** dejar de adaptar a mano captions genéricos a Qwen3-VL / Krea 2.

**Éxito v1:** instalable por el flujo estándar + UI usable + detección de stack Krea 2 + escritura del prompt vía `PromptProvider` inyectable; el provider real (VL/API) no existe aún — stub que parte de una descripción opcional del usuario o de un template mínimo.

**Fuera de alcance v1:** FLUX.2 Klein 9B, backend VL/API, auto-aplicar sampler/steps/CFG (solo *hints* informativos), app desktop aparte, lógica basada en VAE, hornear la extensión en `builtin-extensions/` / imagen Docker.

---

## Tech Stack

| Pieza | Elección |
|-------|----------|
| Host | Forge Neo / A1111-compatible extension loader |
| Lenguaje | Python 3 (el del runtime Forge) |
| UI | Gradio + `modules.scripts.Script` / `InputAccordion` |
| Layout | Extensión WebUI clásica: raíz con `scripts/`, paquete Python importable, `README.md` |
| Distribución | Repo git clonable / Install from URL; opcionalmente también semilla en `docker-neo/extensions/` |
| Persistencia | Ninguna en v1 (sin DB, sin files de caché obligatorios) |
| Backend visión/LLM | **No** en v1; interfaz preparada (`PromptProvider`) |
| `install.py` | Solo si hace falta; v1 **sin** deps extra (usa Gradio/PIL/`modules` ya presentes). Si aparece `install.py`, debe ser no-op o mínimo — esta imagen usa `--skip-install` |

Referencia de prompting Krea 2 / FLUX prosa: `guia_img_prompts.md` (principios de prosa natural; perfil Krea 2 prioriza detalle de composición/luz/materiales y texto entre comillas).

---

## Commands

**Instalar (Forge Neo / host o contenedor):**

```bash
# Opción A — UI: Extensions → Install from URL → pegar URL del repo git → Install → Apply and restart UI

# Opción B — CLI en EXTENSIONS_PATH (ej. /data/extensions o forge-data/extensions)
git clone <URL_DEL_REPO> sd-forge-img2prompt
# Luego Apply and restart UI (o reiniciar el contenedor)
```

**Desarrollo en monorepo `docker-neo` (opcional):** el código puede vivir en `extensions/sd-forge-img2prompt/` como semilla que `make up` copia **solo si falta** la carpeta en `EXTENSIONS_PATH` (mismo patrón Moodboard). Eso no sustituye el flujo Install from URL.

```bash
make up
make shell
```

Tests unitarios (sin levantar WebUI):

```bash
python -m pytest extensions/sd-forge-img2prompt/tests -q
# o, si el cwd es la raíz de la extensión:
python -m pytest tests -q
```

Lint (Ask first si el repo aún no lo usa ahí):

```bash
ruff check extensions/sd-forge-img2prompt
```

---

## Project Structure

Raíz de la extensión = lo que Forge clona en `extensions/sd-forge-img2prompt/`:

```text
sd-forge-img2prompt/          # raíz instalable (git root de la extensión)
  README.md                   # Install from URL + uso v1 + cómo enchufar provider
  metadata.ini                # opcional; si el ecosistema Neo lo usa, incluirlo
  scripts/
    img2prompt.py             # Script Gradio (UI + callbacks) — requerido por el loader
  forge_img2prompt/
    __init__.py
    stack.py                  # Detección Krea 2 / turbo|RAW / TE
    profiles/
      __init__.py
      base.py                 # Protocolo Profile + PromptRequest/Result
      krea2.py                # Perfil Krea 2 (prosa, hints steps/CFG)
    providers/
      __init__.py
      base.py                 # Protocolo PromptProvider
      stub.py                 # Stub v1 (sin backend)
  tests/
    test_stack.py
    test_stub_provider.py
    test_krea2_profile.py

# En docker-neo (doc del producto, no parte del zip/clone de la extensión):
docker-neo/docs/spec-forge-neo-img2prompt_25-09-2026.md
```

Importante: `scripts/*.py` debe poder importar `forge_img2prompt` con la raíz de la extensión en `sys.path` (comportamiento habitual del loader; si hace falta un ajuste mínimo en el script, documentarlo).

---

## Code Style

- Extensión como las existentes: imports desde `modules.*`, accordion AlwaysVisible, `elem_id` con prefijo estable.
- Comentarios solo si hay complejidad no obvia.
- Nombres en inglés en código; UI y docs de usuario en español.
- Sin lógica de prompt acoplada a Gradio: UI → `PromptRequest` → `PromptProvider` → `PromptResult` → rellenar textbox.

Ejemplo de contrato (ilustrativo):

```python
from typing import Protocol
from dataclasses import dataclass
from PIL import Image

@dataclass(frozen=True)
class StackInfo:
    family: str          # "krea2" | "unknown"
    variant: str         # "turbo" | "raw" | "unknown"
    checkpoint: str
    text_encoder: str
    is_supported: bool

@dataclass(frozen=True)
class PromptRequest:
    image: Image.Image | None
    user_notes: str
    stack: StackInfo

@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_hint: str   # vacío o aviso si distilled/turbo
    sampler_hints: str   # texto informativo, no aplica settings

class PromptProvider(Protocol):
    def generate(self, request: PromptRequest) -> PromptResult: ...
```

Stub v1: si hay `user_notes`, reescribe/envuelve con plantilla Krea 2; si no, genera un prompt mínimo pidiendo completar + metadatos de stack en un comentario HTML de UI (no en el prompt final salvo que aporten).

---

## Testing Strategy

| Nivel | Qué | Dónde |
|-------|-----|-------|
| Unit | Detección `family`/`variant` a partir de nombres de checkpoint/TE | `tests/test_stack.py` |
| Unit | Stub genera prosa no-tag-soup y respeta perfil Krea 2 | `tests/test_stub_provider.py`, `test_krea2_profile.py` |
| Manual | Subir/pegar imagen, notes, Generate → prompt en txt2img con checkpoint Krea 2 | Smoke en Forge Neo GPU |

**No** tests e2e de WebUI en v1 (Ask first si se quieren).

Cobertura: no hay umbral numérico; sí deben pasar todos los unit tests del paquete antes de dar v1 por hecha.

---

## Boundaries

**Always**
- Detectar stack antes de generar; si no es Krea 2, mostrar aviso claro y no fingir optimización.
- Mantener `PromptProvider` desacoplado (un solo sitio para enchufar backend).
- Prompts en prosa natural; no booru/tag spam.
- Fecha en docs nuevos según convención del repo.

**Ask first**
- Publicar repo git aparte vs solo carpeta en `docker-neo/extensions/`.
- Añadir la extensión al seed de `make up` / target `make` de refresh.
- Meterla en `builtin-extensions/` (imagen) — por defecto **no**.
- Añadir dependencias Python / `install.py` no vacío.
- Auto-aplicar steps/CFG/sampler (v1 solo hints).
- Soporte Klein u otras familias.
- Implementar provider real (local VL o API).

**Never**
- Enviar imágenes a APIs sin decisión explícita.
- Acoplar prompts al VAE.
- Exigir un procedimiento de install distinto al de cualquier otra extensión WebUI.
- Sobrescribir extensiones ajenas o el patch de backend Krea2.
- Incluir secretos / API keys en el repo.

---

## Success Criteria

1. Se puede instalar con **Extensions → Install from URL** (o `git clone` en `extensions/`) y aparece tras **Apply and restart UI**, sin pasos Docker especiales.
2. Layout válido de extensión WebUI (`scripts/` + paquete; sin deps extra en v1).
3. Extensión visible en txt2img (y opcionalmente img2img) como accordion **Image → Prompt**.
4. Acepta imagen por upload y por pegado en el control de imagen Gradio.
5. Campo opcional **notas / descripción** del usuario.
6. Con checkpoint/TE reconocibles como Krea 2, el botón genera un prompt y lo escribe en el textbox positivo de la pestaña activa.
7. Muestra hints de sampler (Turbo ≈ 8 steps / CFG bajo; RAW ≈ más steps / CFG ~4.5) sin mutar settings.
8. Si el stack no es Krea 2, UI avisa y no promete prompt “optimizado”.
9. Existe `PromptProvider` + `StubProvider`; cambiar de provider no exige reescribir la UI.
10. `pytest` de `tests/` pasa sin levantar Forge.
11. README documenta Install from URL, uso v1 y el punto de extensión del provider.

---

## Open Questions

- ¿Repo git propio (URL pública/privada) en v1, o primero solo carpeta en `docker-neo/extensions/` lista para clonar/copiar, y el remoto después? **Default propuesto:** estructura instalable desde el día 1; remoto Ask first.

Pendientes diferidos (post-v1):

- ¿Provider local reutilizando Qwen3-VL cargado vs API?
- ¿Auto-aplicar preset `krea2` (sampler/steps/CFG)?
- Perfil Klein 9B (misma prosa, TE Qwen3-8B, 4 steps distilled).

---

## Assumptions (revisión humana)

1. Es una **extensión WebUI estándar** (Install from URL / clone en `extensions/`), no app en `tools/` ni builtin de imagen.
2. Puede además sembrarse desde `docker-neo/extensions/` como Moodboard, pero eso es opcional; el camino canónico es el del gestor de extensiones.
3. “Sin backend” = sin servicio/VL/API; el stub **sí** puede transformar `user_notes` + perfil Krea 2 en prosa usable.
4. Clipboard = pegar en el `gr.Image` del navegador; no API nativa del OS.
5. Detección Krea 2 por heurística de nombres de checkpoint/TE (p. ej. `krea`, `qwen3vl`); no hace falta leer arquitectura del state dict en v1.
6. Nombre de carpeta: `sd-forge-img2prompt`.
→ Corrige ahora o aprueba el spec actualizado.
