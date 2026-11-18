# Spec: UW Modular Wildcards en Forge Neo

Fecha: 2026-10-23

Extensión AlwaysVisible que lleva a Forge Neo el uso del pack
[UW Modular NSFW Wildcards v3.0.0](https://civitai.red/models/2789815)
(motor 3.1.0): prosa por cláusulas para Krea 2, tamaño desde un marcador
`[W:H]`, y descarga de las LoRA del pack la primera vez que el prompt las pide.

El corpus YAML, las miniaturas y los pesos no se versionan. Los trae quien
usa la extensión.

La extensión vive en `extensions/sd-forge-uw-wildcards/` de este workspace,
con su propio repositorio Git. No es semilla: `make seed-extensions` la
omite (archivo `.no-seed`) y se instala desde la URL del repo, como el resto
de extensiones de la interfaz. El zip local `uwModularV3/` es la variante
sin tags `<lora:…>` incrustados; la descarga se usa cuando el prompt sí los
trae.

## Objective

Quien genera con Krea 2 en txt2img/img2img puede activar un acordeón, elegir
un router o una plantilla del pack, fijar cláusulas (pin / skip / regenerar
un hueco) y generar. El prompt que llega al encoder es prosa, sin el marcador
de aspecto. Si el texto trae `<lora:ruta:peso>` y el fichero no está, se
descarga una vez a esa misma ruta.

El éxito de la v3 del pack es ese ciclo, no “un wildcard aleatorio”.
`sd-dynamic-prompts` sigue expandiendo el resto de `__tags__` (por ejemplo
`__film/…`). Esta extensión solo consume tags `uw-*` y corre antes.

### Fuera de este spec

- Portar los workflows ComfyUI Turbo/Raw, el catálogo de 358 miniaturas y
  PromptLoraExtractor. En Forge el tag `<lora:…>` se queda en el prompt.
- Vendor del YAML, de las miniaturas o de cualquier peso.
- LoRA de casting: el pack no la incrusta en el texto, así que no se descarga
  sola.
- Hijack del UNet. Solo se reescriben prompt, ancho y alto.

### Entregas

1. **Slice A — caparazón, aspecto y LoRA.** Acordeón, clave de API, manifiesto,
   descarga perezosa, marcador `[W:H]`. El texto UW que aún no sabemos expandir
   se deja intacto. Se puede probar sin el zip.
2. **Slice B — expansor de cláusulas.** Port del `ComfyUI-WildcardClause` del
   zip (17 huecos, pin / skip / regenerar, limpieza de puntuación, semilla por
   hueco). No empieza hasta tener ese código. Inventar el YAML sería otra
   extensión.

## Tech Stack

- Python 3 del contenedor Forge Neo. Sin dependencias nuevas y sin `install.py`.
- Gradio y `modules.scripts.Script` / `InputAccordion`, como Moodboard e
  Identity Edit.
- `unittest` de la biblioteca estándar, igual que `forge/tests/`.
- HTTP con `urllib` de la biblioteca estándar. No Aria2 ni el gestor de colas
  del browser de Civitai: ese gestor renombra y recoloca el fichero, y el tag
  dejaría de encontrarlo.
- Descarga: `GET https://{dominio}/api/download/models/{modelVersionId}` con
  `Authorization: Bearer …`, siguiendo el redirect solo si el destino es un
  fichero (la misma clasificación que `civitai_download._classify_download_redirect`).

## Commands

Los tests no arrancan el WebUI. Desde la raíz del repo de la extensión:

```sh
python3 -m unittest tests.test_api_key tests.test_aspect tests.test_lora_fetch -v
```

No hay lint ni build propios. `make seed-extensions` no copia esta carpeta.

## Project Structure

```text
forge/extensions/sd-forge-uw-wildcards/
  README.md
  metadata.ini                  # Before = sd-dynamic-prompts
  scripts/uw_wildcards.py       # Script AlwaysVisible + on_ui_settings
  forge_uw_wildcards/
    __init__.py
    api_key.py                  # resolución de la clave
    aspect.py                   # marcador [W:H] → texto + ancho + alto
    lora_tags.py                # parseo de <lora:…>
    lora_fetch.py               # manifiesto + descarga a la ruta del tag
    manifest.json               # alias → modelVersionId (versiones pinadas)

  tests/test_api_key.py
  tests/test_aspect.py
  tests/test_lora_fetch.py
  .no-seed
```

El slice B añadirá `forge_uw_wildcards/clauses.py` y `tests/test_clauses.py`.
Los YAML del usuario viven en el volumen de datos (`wildcards/`), no aquí.

`.no-seed` hace que la semilla de docker-neo ignore la carpeta. La
instalación de quien usa Forge es Install from URL.

## Code Style

Lógica en el paquete hermano. El script de Gradio solo construye el acordeón
y llama al paquete. Sin comentarios que reescriban el código. Nombres en
español solo en etiquetas de UI; identificadores en inglés.

La clave no se registra con el nombre `custom_api_key`. Esa opción ya es la
del browser y vive en `config.json`.

```python
def resolve_civitai_api_key(opts) -> str:
    """Primero la clave del browser; la de esta extensión solo si aquella está vacía."""
    data = getattr(opts, "data", None)
    browser = ""
    if isinstance(data, dict):
        browser = (data.get("custom_api_key") or "").strip()
    if not browser:
        browser = (getattr(opts, "custom_api_key", "") or "").strip()
    if browser:
        return browser
    return (getattr(opts, "uw_wildcards_civitai_api_key", "") or "").strip()
```

Un tag `<lora:krea2/nsfw/snofs_krea_v1_3D:0.6>` se guarda en la carpeta de
LoRA que Forge tenga configurada (`--lora-dir`, si no la primera de
`--lora-dirs`, si no `models/Lora`), dentro de `UWModularWildcards/`,
conservando el resto de la ruta:

`{lora_dir}/UWModularWildcards/krea2/nsfw/snofs_krea_v1_3D.safetensors`

El tag del prompt pasa a
`<lora:UWModularWildcards/krea2/nsfw/snofs_krea_v1_3D:0.6>` para que Forge lo
encuentre. Escritura a `*.partial` y `os.replace`. Un `<lora:…>` que no esté
en el manifiesto no se mueve ni se descarga. Si el alias no tiene
`modelVersionId`, no se busca “el último” en Civitai: el job se detiene y
nombra el alias.

## Testing Strategy

`unittest`, sin red y sin importar `modules.*`. El HTTP se sustituye por un
fake que devuelve bytes o un redirect a una página de login.

| Capa | Qué cubre |
|------|-----------|
| `test_uw_api_key` | Browser gana si las dos claves existen. Vacía del browser cae a la nuestra. Las dos vacías devuelven `""`. |
| `test_uw_aspect` | `[3:4]` final se quita y el tamaño cabe en el presupuesto, múltiplo de `divisible_by`. Sin marcador se usa `3:4`. Un `[3:4]` a mitad de frase no se toca. |
| `test_uw_lora_fetch` | Fichero ya presente no descarga. Falta + id conocido escribe la ruta del tag. Sin id, o redirect a página, lanza y no deja un `.partial`. |
| `test_uw_clauses` (slice B) | Pin no cambia al rerolar otro hueco. Skip omite la cláusula. La semilla de un hueco no altera los demás. |

No hay objetivo de cobertura global. Cada función nueva del paquete entra con
el test que fija su contrato. La UI de Gradio no se testea aquí: hace falta
el WebUI.

## Boundaries

- Always: tests de unidad del paquete antes de dar una slice por hecha. Versiones de Civitai pinadas en `manifest.json`. El tag `<lora:…>` permanece en el prompt. Tags que no sean `uw-*` no se reescriben.
- Ask first: nueva dependencia, mover la extensión a `builtin-extensions/`, cambiar el orden “clave del browser primero”, añadir un id de versión que no esté en la ficha, meter el corpus en git.
- Never: commitear claves, pesos, YAML del pack o miniaturas. Registrar la opción `custom_api_key`. Llamar al gestor de descargas del browser. Expandir cláusulas con un parser inventado. Descargar durante el preview. Re-tirar cláusulas en `process_before_every_sampling` (el hires fix volvería a cambiar la escena).

## Success Criteria

Slice A:

- Con el acordeón apagado, prompt, ancho y alto salen iguales.
- Con el acordeón encendido y aspecto activo, un prompt que termina en `[3:4]` pierde el marcador y `p.width` / `p.height` coinciden con el cálculo de `aspect.py` para los megapíxeles configurados (defecto 1.0, múltiplo 16, relación por defecto `3:4`).
- Settings muestra una clave propia, tipo password. Si `custom_api_key` tiene valor, las peticiones usan esa y no la propia.
- La primera generación cuyo prompt contiene un alias del manifiesto sin fichero lo descarga a `{lora_dir}/UWModularWildcards/…` y reescribe el tag. La segunda no vuelve a pedir el fichero.
- Sin clave, o si Civitai devuelve login / página, la generación no sigue. El mensaje nombra el alias. Si el browser está en SFW only (`civitai.com`), el mensaje lo dice.
- Un alias ausente del manifiesto no dispara una búsqueda por nombre.
- `python3 -m unittest` de los tres módulos de la slice A termina en OK.

Slice B (además):

- Un router `__uw-krea2-nsfw/router/…__` o una plantilla `__uw-krea2-nsfw/template/…__` se expande con el mismo texto que el nodo 3.1.0 para la misma semilla, los mismos overrides y un YAML de fixture.
- Regenerar un hueco cambia solo ese hueco. Pin lo deja escrito. Skip lo quita y no deja `BODY.,` ni comas dobles.
- El PNG guarda el prompt ya expandido y un extra con el JSON de overrides.
- El preview rellena el texto y lista las LoRA que faltan, sin descargar.

## Decisions

- Acordeón AlwaysVisible en txt2img e img2img, `sorting_priority` por debajo de Dynamic Prompts. `metadata.ini`: `Before = sd-dynamic-prompts`.
- Hook de generación: `before_process`. El batch comparte un solo ancho y alto; si hiciera falta variación de texto por imagen, será `before_process_batch` en el slice B, sin cambiar la longitud de `prompts` ni de `seeds`.
- Estado de cláusulas (slice B): un JSON, no 17 argumentos Gradio. Así un hueco nuevo no desplaza el infotext.
- Presupuesto de megapíxeles bajo por defecto (1.0). En esta máquina el VAE es el límite, no el tiempo de muestreo.
- Dominio y proxy se leen de las opciones que ya usa el browser: `civitai_sfw_only` elige `civitai.com` o `civitai.red`; `custom_civitai_proxy` se reenvía. No se importa `scripts.civitai_api`.
- Manifiesto inicial, ids comprobados en ficha. El resto queda con `version_id: null` hasta leer la ficha, y mientras tanto no se descarga:

| Alias del tag | Modelo | Versión pinada |
|---------------|--------|----------------|
| `krea2/nsfw/snofs_krea_v1_3D` | 1972981 | 3220691 (v1.3D) |
| `krea2/sliders/slider-photo__Photo_Detail_Slider_for_SNOFS_Krea` | 2823820 | 3185533 (v1.0) |
| `krea2/styles/photoreal/style-photoreal__Realism_Engine_Ideogram_4_Krea_2` | — | null (el pack pide v3.0; no usar la v2) |
| `krea2/styles/photoreal/style-photoreal__Mrpopo_Photoreal_Enhanced` | — | null |
| `krea2/face/face-expression__Snapchat_Face_Helper` | — | null |
| `krea2/nsfw/acts/nsfw-act-masturbation__krea2_sex` | — | null |
| `krea2/nsfw/acts/nsfw-act__krea2_sex_淫模_krea2_dildo_in_pussy` | — | null |

SNOFS v1.3D se publica como LoKR. El slice A solo garantiza que el fichero queda en la ruta. Si el loader de Forge lo rechaza, el log lo dice y no se marca como aplicado. Comprobarlo es manual, con el WebUI.

## Open Questions

1. Slice A en curso. Slice B cuando se porte `ComfyUI-WildcardClause` (está dentro del zip de `uwModularV3/`, variante sin tags `<lora:…>`).
2. Etiquetas del acordeón y de Settings: textos en español. Si se prefiere el inglés del resto de Forge, se cambian sin tocar la lógica.
