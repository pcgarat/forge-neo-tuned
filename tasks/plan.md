# Última modificación: 2026-09-26

# Plan: sd-forge-img2prompt (Forge Neo / Krea 2 v1)

Spec: [`docs/spec-forge-neo-img2prompt_25-09-2026.md`](../docs/spec-forge-neo-img2prompt_25-09-2026.md)  
Código: [`pcgarat/sd-forge-img2prompt`](https://github.com/pcgarat/sd-forge-img2prompt)  
Skill: `forge-neo-extensions`

## Enfoque

- Repo **dedicado** (no semilla en `docker-neo/extensions/`).
- UI: `on_ui_tabs` + send-to con `infotext_utils`.
- v1: `StubProvider` (notas → prosa Krea 2); imagen reservada para backend futuro.
- Sin `install.py` / deps (compatible `--skip-install`).

## Estado

| Ítem | Estado |
|------|--------|
| Repo GitHub + scaffold | Hecho (`main`) |
| Tests unitarios stack/stub | Hecho (9 passed) |
| Pestaña + paste_params | En código; **smoke Forge Neo pendiente** |
| Seed en docker-neo | No (install por URL) |

## Pendiente

1. Smoke en Forge Neo (Install from URL → Generate → Send txt2img con Krea 2).
2. Ajustar lectura de TE si el dropdown Neo no mapea a `opts.sd_vae` / `sd_text_encoder`.
3. (Ask first) `make img2prompt-ext` en docker-neo para clonar/actualizar.

## Fuera de ciclo

Provider VL/API, Klein 9B, auto sampler/CFG, AlwaysVisible.
