# Última modificación: 2026-09-25

# Plan: sd-forge-img2prompt (Forge Neo / Krea 2 v1)

Spec: [`docs/spec-forge-neo-img2prompt_25-09-2026.md`](../docs/spec-forge-neo-img2prompt_25-09-2026.md)

## Enfoque

Extensión WebUI estándar en `extensions/sd-forge-img2prompt/`, sembrable por `make seed-extensions` e instalable por clone / Install from URL.

UI como **accordion AlwaysVisible** (mismo patrón que Moodboard/Edit), no pestaña top-level: necesitamos checkpoint/TE del tab de generación.

Backend de visión **no** en v1; `PromptProvider` + `StubProvider`. Relleno del prompt vía JS a `#txt2img_prompt` / `#img2img_prompt`.

## Componentes y dependencias

```
forge_img2prompt/stack.py          (detección Krea2 / turbo|raw)
        │
forge_img2prompt/profiles/*        (plantilla prosa + hints sampler)
        │
forge_img2prompt/providers/*       (Protocol + Stub)
        │
scripts/img2prompt.py              (Gradio Script AlwaysVisible)
        │
README + seed en docker-neo
```

Orden de implementación: núcleo puro (tests sin Forge) → Script UI → semilla/docs → smoke manual.

## Riesgos

| Riesgo | Mitigación |
|--------|------------|
| `--skip-install` en docker-neo | v1 sin `install.py` / sin deps nuevas |
| Heurística de nombres falla | Aviso UI claro + lista de substrings documentada; fácil ampliar |
| JS de prompt roto en Gradio 4 | Probar send-to; fallback: textbox editable + copy |
| Confusión con Adeliox image2prompt | README deja claro: stack-aware Krea2, stub, prosa; no VL propio |
| Paquete no importable desde `scripts/` | Extensión root en path (loader A1111); si falla, `sys.path` mínimo en el script |

## Verificación global

- `python -m pytest extensions/sd-forge-img2prompt/tests -q`
- Seed / clone en `EXTENSIONS_PATH` → restart → accordion visible
- Con checkpoint Krea2: Generate escribe prosa en el prompt
- Sin Krea2: aviso, no “optimizado”
