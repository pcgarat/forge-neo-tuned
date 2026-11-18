# Extensiones horneadas en la imagen (extensions-builtin)

A diferencia de [`extensions/`](../extensions/) (semilla → volumen `/data/extensions`),
lo que vive aquí se copia a `/app/webui/extensions-builtin/` en el **Dockerfile**.

Forge carga ambos directorios. Usa esta carpeta solo cuando la extensión debe
sobrevivir a un volumen vacío (RunPod nuevo, `DATA_PATH` limpio) sin pasos manuales.

| Carpeta | Descripción |
|---------|-------------|
| `sd-forge-krea2-depth-controlnet` | Depth/Pose ControlNet-LoRA para Krea 2 |
| `ADetailer-Neo` | Detección + inpaint de caras/manos ([Haoming02](https://github.com/Haoming02/ADetailer-Neo)) |
| `sd-webui-state-manager-neo` | Guardar/restaurar configs txt2img/img2img |
| `sd-civitai-browser-neo` | Browser/organizador de modelos (sustituye Civitai Helper) |
| `sd-webui-agent-scheduler-neo` | Cola de generaciones |
| `sd-webui-prompt-all-in-one-neo` | Historial, estilos y traducción de prompts |
| `forge-neo-lama-cleaner` | Borrado/inpaint de objetos (LaMa) |
| `forge-neo-rembg-cpu` | Parchea rembg (quitar fondo): CPU en vez de CUDA y sesión cacheada, para no competir por VRAM con `--lowvram` |

**No** metas aquí pesos de modelo (YOLO, big-lama, ControlNet-LoRA…): van en `DATA_PATH/Models/…`.
**No** dupliques la misma carpeta en `extensions/` ni en `EXTENSIONS_PATH` (Forge la cargaría dos veces).

Deps Python se hornean en el Dockerfile (`--skip-install` desactiva cada `install.py`).
