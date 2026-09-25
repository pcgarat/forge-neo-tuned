# Última modificación: 2026-09-25

# Todo: sd-forge-img2prompt

## Task 1: Esqueleto instalable + contratos

**Description:** Crear `extensions/sd-forge-img2prompt/` con layout WebUI (`scripts/`, `forge_img2prompt/`, `tests/`, `README.md`, opcional `metadata.ini`). Definir dataclasses/`Protocol` de stack, request/result y provider.

**Acceptance criteria:**
- [ ] Carpeta válida para seed / clone en `extensions/`
- [ ] Contratos importables sin Forge (`forge_img2prompt.profiles.base`, `providers.base`)
- [ ] README con Install from URL / clone / seed docker-neo

**Verification:**
- [ ] `python -c "from forge_img2prompt.providers.base import PromptProvider"` desde la raíz de la extensión (o pytest vacío verde)
- [ ] Manual: layout coincide con Moodboard/Depth (scripts + paquete)

**Dependencies:** None

**Files likely touched:**
- `extensions/sd-forge-img2prompt/README.md`
- `extensions/sd-forge-img2prompt/metadata.ini` (opcional)
- `extensions/sd-forge-img2prompt/forge_img2prompt/**/*.py`
- `extensions/sd-forge-img2prompt/scripts/.gitkeep` o stub mínimo

**Estimated scope:** Medium

---

## Task 2: Detección de stack Krea 2

**Description:** Heurística por nombres de checkpoint y text encoder → `StackInfo` (`krea2`/`unknown`, `turbo`/`raw`/`unknown`).

**Acceptance criteria:**
- [ ] Detecta variantes tipicas (`krea`, `krea2`, `turbo`, `raw`, `qwen3vl`)
- [ ] `is_supported` solo true para familia krea2
- [ ] Tests unitarios con nombres sintéticos

**Verification:**
- [ ] `python -m pytest extensions/sd-forge-img2prompt/tests/test_stack.py -q`

**Dependencies:** Task 1

**Files likely touched:**
- `extensions/sd-forge-img2prompt/forge_img2prompt/stack.py`
- `extensions/sd-forge-img2prompt/tests/test_stack.py`

**Estimated scope:** Small

---

## Task 3: Perfil Krea 2 + StubProvider

**Description:** Perfil de prosa (orden sujeto→entorno→luz→estilo; comillas para texto) y hints Turbo (~8 steps, CFG bajo) vs RAW (~28 steps, CFG ~4.5). Stub reescribe `user_notes` o plantilla mínima; no tag-soup.

**Acceptance criteria:**
- [ ] Con notes → prosa usable en inglés (estándar Krea prompting)
- [ ] Sin notes → prompt mínimo explícito / placeholder útil
- [ ] `sampler_hints` y aviso de negativo en turbo
- [ ] Tests sin Forge

**Verification:**
- [ ] `python -m pytest extensions/sd-forge-img2prompt/tests/test_stub_provider.py tests/test_krea2_profile.py -q`

**Dependencies:** Task 1–2

**Files likely touched:**
- `extensions/sd-forge-img2prompt/forge_img2prompt/profiles/krea2.py`
- `extensions/sd-forge-img2prompt/forge_img2prompt/providers/stub.py`
- `extensions/sd-forge-img2prompt/tests/test_*.py`

**Estimated scope:** Medium

---

## Task 4: Script Gradio AlwaysVisible

**Description:** Accordion Image → Prompt: imagen (upload/clipboard), notes, estado de stack, Generate, Apply to prompt (JS), hints. Lee checkpoint/TE vía `shared`/`sd_models` como el resto de Neo.

**Acceptance criteria:**
- [ ] Visible en txt2img (y img2img)
- [ ] Generate → texto en UI + opción de volcar a `#txt2img_prompt` / `#img2img_prompt`
- [ ] Stack no-Krea2 → aviso, no finge optimización
- [ ] Sin `install.py`; imports de `forge_img2prompt` OK

**Verification:**
- [ ] Manual en Forge Neo tras seed/restart
- [ ] Unit tests previos siguen pasando

**Dependencies:** Task 1–3

**Files likely touched:**
- `extensions/sd-forge-img2prompt/scripts/img2prompt.py`

**Estimated scope:** Medium

---

## Checkpoint: After Tasks 1–4

- [ ] pytest del paquete verde
- [ ] Extensión carga en Forge Neo
- [ ] Flujo imagen + notes → prompt en textbox con Krea2

---

## Task 5: Integración docs / seed docker-neo

**Description:** Listar la extensión en `extensions/README.md`; confirmar que entra en `make seed-extensions`. No meter en builtin ni cambiar Dockerfile.

**Acceptance criteria:**
- [ ] Documentada en `extensions/README.md`
- [ ] Spec/plan referenciados si aplica
- [ ] Sin cambios de imagen Docker

**Verification:**
- [ ] `make seed-extensions` copia `sd-forge-img2prompt` en destino limpio (o skip si ya existe)

**Dependencies:** Task 4

**Files likely touched:**
- `extensions/README.md`
- posiblemente nota corta en README raíz (Ask first si enreda)

**Estimated scope:** Small

---

## Fuera de este ciclo

- Provider VL/API real
- Perfil Klein 9B
- Auto-aplicar sampler/CFG
- Repo git aparte (Ask first)
- Target `make img2prompt-ext` de refresh forzado
