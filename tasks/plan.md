# Última modificación: 2026-09-26

# Plan: sd-forge-img2prompt (Forge Neo / Krea 2 v1) — revisado

Spec: [`docs/spec-forge-neo-img2prompt_25-09-2026.md`](../docs/spec-forge-neo-img2prompt_25-09-2026.md)  
Skill: `forge-neo-extensions`

## Crítica al plan anterior

| Decisión previa | Problema |
|-----------------|----------|
| AlwaysVisible “porque necesitamos el stack” | El checkpoint/TE viven en `shared` (globales). Una pestaña propia también los lee al pulsar Generate. AlwaysVisible brilla cuando hay hook `process`/`postprocess`; **esta tool no muta la generación**. |
| Rellenar prompt con JS ad-hoc | La skill marca JS como preferible en tabs, y en Neo el camino idiomático es `modules.infotext_utils.register_paste_params_button` (no `generation_parameters_copypaste`). |
| Árbol `profiles/` + `providers/` + muchos tests antes de UI | Correcto para TDD, pero v1 solo Krea2+stub: riesgo de over-structure. Protocol fino sí; carpetas profundas no aportan hasta Klein/backend. |
| Accordion junto a Moodboard/Edit/Depth/ADetailer | Más ruido en el panel de scripts para una tool de “preparar prompt”. |

## Enfoque revisado (recomendado)

**UI: `script_callbacks.on_ui_tabs`** — pestaña **Image → Prompt**.

- Encaja con “tool aparte” (skill + futuro provider VL con VRAM).
- Send-to txt2img/img2img vía **`infotext_utils.ParamBinding`**.
- Stack Krea2 se detecta al generar leyendo nombres de checkpoint/TE en `shared` / `sd_models`.
- Sin `process()` vacío; sin `install.py`; sin deps nuevas (`--skip-install`).

**Alternativa (solo si priorizas cero cambio de pestaña):** AlwaysVisible + textbox resultado + botones paste `infotext_utils` / copy — **sin** JS custom como path primario. Menos limpio para v2 con VL.

**Paquete (slim):**

```text
extensions/sd-forge-img2prompt/
  README.md
  scripts/
    img2prompt.py          # on_ui_tabs + wiring Gradio
  forge_img2prompt/
    __init__.py
    stack.py               # StackInfo + detect()
    provider.py            # Protocol + StubProvider (+ plantilla Krea2 inline o krea2.py corto)
  tests/
    test_stack.py
    test_stub_provider.py
  javascript/              # solo si paste_params no cubre un caso; preferir no
```

Opcional más adelante: `profiles/krea2.py` al añadir Klein; `metadata.ini` solo si hace falta orden.

## Orden (vertical)

1. Esqueleto instalable + `stack` + `StubProvider` + tests unitarios.
2. Pestaña Gradio: imagen (upload/clipboard), notes, status stack, Generate, prompt out, Send txt2img/img2img.
3. Seed docs (`extensions/README.md`); smoke en Forge Neo.

## Riesgos

| Riesgo | Mitigación |
|--------|------------|
| `--skip-install` | Sin `install.py` / deps |
| Heurística nombres | Aviso UI + substrings documentados |
| `infotext_utils` binding raro en Gradio 4 | Fallback: textbox + copy (`show_copy_button`) |
| Usuario quiere accordion in-tab | Cambiar a AlwaysVisible en 1 fichero de script; núcleo igual |

## Verificación

- `pytest extensions/sd-forge-img2prompt/tests -q`
- Seed/clone → restart → pestaña visible
- Krea2: Generate + Send rellena prompt
- No-Krea2: aviso, no fingir optimización
