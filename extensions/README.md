# Extensiones custom (semilla)

Código de extensiones **no built-in** que `make up` / `make workspace` copian a `EXTENSIONS_PATH` **solo si falta** esa carpeta en el destino.

## Incluidas

| Carpeta | Descripción |
|---------|-------------|
| `sd-webui-infinite-image-browsing` | Infinite Image Browsing |
| `stable-diffusion-webui-zoomimage` | Zoom de imagen |
| `sd-forge-krea2-moodboard` | Krea2 Moodboard (UI) |
| `sd-forge-krea2-edit` | Krea2 Identity Edit (UI; fix `dynamic_args` aplicado) |

El patch de backend Krea2 va en la **imagen** (`patches/`), no aquí.

## Extensiones en la imagen (`builtin-extensions/`)

Viven en `extensions-builtin` de la imagen (no en este volumen). Incluyen Krea2 Depth/Pose,
ADetailer Neo, State Manager Neo, CivitAI Browser Neo, Agent Scheduler Neo,
Prompt All-in-One Neo y Lama Cleaner Neo. Ver [`builtin-extensions/README.md`](../builtin-extensions/README.md).

**No** instales Civitai Helper clásico: lo sustituye `sd-civitai-browser-neo` en la imagen.

## ReActor (no va en la semilla)

Instálalo en `EXTENSIONS_PATH` desde [codeberg.org/Gourieff/sd-webui-reactor](https://codeberg.org/Gourieff/sd-webui-reactor) (no el fork `-sfw` de GitHub). Las deps Python van en la **imagen** (`make build` / `make reactor-fix`); ver README § ReActor.

## sd-dynamic-prompts + estilos Krea2 (no van en la semilla)

Wildcards y biblioteca de estilos fotográficos para el encoder LLM de Krea2. Las deps Python van en la **imagen** (`dynamicprompts`, `send2trash`), junto al shim de `generation_parameters_copypaste` que la extensión necesita porque Forge Classic renombró ese módulo a `infotext_utils`.

Instalación en `EXTENSIONS_PATH` (`$ext`), una vez por volumen de datos:

```sh
git clone https://github.com/adieyal/sd-dynamic-prompts.git "$ext/sd-dynamic-prompts"
git clone https://github.com/aoleg/photographic-styles-and-wildcards-for-Krea-2.git "$ext/photographic-styles-and-wildcards-for-Krea-2"

# Estilos: Forge lee <data>/styles_integrated.csv además de styles.csv (que queda
# libre para los estilos propios guardados desde la UI).
ln -sfn extensions/photographic-styles-and-wildcards-for-Krea-2/styles.csv "$data/styles_integrated.csv"

# Wildcards fuera de la extensión: así desinstalarla no se lleva la biblioteca.
mkdir -p "$data/wildcards/film"
ln -sfn ../../wildcards "$ext/sd-dynamic-prompts/wildcards"
ln -sfn ../../extensions/photographic-styles-and-wildcards-for-Krea-2/photographica.yaml \
  "$data/wildcards/film/photographica.yaml"
```

Los symlinks apuntan al repo clonado, así que un `git pull` en él actualiza estilos y wildcards.

## Repo propio (no semilla)

`sd-forge-uw-wildcards/` tiene `.no-seed` y su propio Git. `make seed-extensions` no la copia. Se instala desde la interfaz con la URL del repositorio.

En la máquina de desarrollo, el repo se monta en el contenedor con `docker-compose.override.yml` (copia de `docker-compose.override.example.yml`). Con ese montaje activo, "Install from URL" falla: la carpeta destino ya existe.

## Reglas

- Añade aquí solo extensiones custom (una carpeta por extensión).
- No metas builtins de Forge.
- No versionar runtime: `.git/`, `iib.db`, `iib_db_backup/`, `*.log`, `__pycache__/`, `.env` locales.
- Si la extensión ya existe en `EXTENSIONS_PATH`, el seed **no la sobrescribe** (conserva DB, `.env`, etc.).
- Para forzar actualización de Krea2 desde GitHub: `make krea2-ext`.
