# Última modificación: 2026-09-26

# Todo: sd-forge-img2prompt (revisado)

## Task 1: Esqueleto + stack + stub (núcleo testeable)

**Description:** Crear `extensions/sd-forge-img2prompt/` instalable. Paquete slim: `stack.py`, `provider.py` (Protocol + StubProvider con prosa Krea2). Tests sin Forge.

**Acceptance criteria:**
- [ ] Layout: `scripts/`, `forge_img2prompt/`, `tests/`, `README.md` (Install from URL / clone / seed)
- [ ] `detect_stack(checkpoint, text_encoder) → StackInfo`
- [ ] Stub: notes → prosa; sin notes → plantilla mínima; hints Turbo/RAW; no tag-soup
- [ ] `is_supported` solo Krea2

**Verification:**
- [ ] `python -m pytest extensions/sd-forge-img2prompt/tests -q`

**Dependencies:** None

**Files:** `extensions/sd-forge-img2prompt/**`

**Scope:** Medium

---

## Task 2: Pestaña UI + paste_params

**Description:** `scripts/img2prompt.py` con `on_ui_tabs`: imagen, notes, status stack, Generate, prompt out, botones Send a txt2img/img2img vía `modules.infotext_utils`. Leer stack real de `shared`/`sd_models` en el click. Sin `install.py`. Guardar `EXT_DIR = scripts.basedir()` en import si hace falta.

**Acceptance criteria:**
- [ ] Pestaña visible tras restart
- [ ] Generate rellena el textbox de salida
- [ ] Send escribe en el prompt del tab destino (o copy fallback documentado)
- [ ] Stack no-Krea2 → aviso claro

**Verification:**
- [ ] Manual smoke Forge Neo
- [ ] Tests Task 1 siguen verdes

**Dependencies:** Task 1

**Files:** `extensions/sd-forge-img2prompt/scripts/img2prompt.py`, README

**Scope:** Medium

---

## Checkpoint: After Tasks 1–2

- [ ] Extensión cargable por seed/clone
- [ ] Flujo imagen + notes → prompt → Send txt2img con Krea2

---

## Task 3: Docs seed docker-neo

**Description:** Entrada en `extensions/README.md`. No builtin, no Dockerfile.

**Acceptance criteria:**
- [ ] Listada como extensión custom sembrable
- [ ] Spec/plan alineados (UI = on_ui_tabs)

**Verification:**
- [ ] `make seed-extensions` copia si falta

**Dependencies:** Task 2

**Files:** `extensions/README.md`, sync mínimo en spec si hace falta

**Scope:** Small

---

## Fuera de ciclo

- AlwaysVisible (solo si se pide explícitamente)
- Provider VL/API, perfil Klein, auto sampler/CFG
- Repo git aparte / `make img2prompt-ext`
- JS custom salvo que `infotext_utils` falle en smoke
