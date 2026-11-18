# forge-neo-rembg-cpu

Ajuste para la extensión [stable-diffusion-webui-rembg](https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg)
(quitar fondo). No aporta UI: solo parchea `rembg` al cargar los scripts.

## Qué corrige

| Problema | Efecto sin el parche |
|----------|----------------------|
| rembg elige `CUDAExecutionProvider` en cuanto ve `onnxruntime-gpu`, sin mirar la VRAM libre | Con `--lowvram` Forge deja <500 MiB y onnxruntime falla: `BFCArena::AllocateRawInternal ... Failed to allocate memory` |
| Crea una sesión ONNX por imagen dentro de `process()` | Cada postprocesado recarga el `.onnx` (~0.4 s) y pide memoria nueva |
| `scripts/api.py` no fija `U2NET_HOME` | Sus `.onnx` se descargan al HOME del contenedor (~1.2 GB) y se pierden al recrearlo |

El fallo de VRAM es especialmente engañoso: la excepción la lanza
`InferenceSession.run()`, no la creación de la sesión, así que onnxruntime no
hace fallback a CPU y el error emerge como un `Fail` opaco a mitad de inferencia.

## Comportamiento

- Inferencia en **CPU** (~0.3 s por imagen a 320×320). La GPU queda entera para Forge.
- Una sesión cacheada por modelo: la carga se paga una vez por proceso.
- `U2NET_HOME` se fija a `<DATA_PATH>/Models/u2net`, así los pesos persisten en el volumen.

## Configuración

| Variable | Valores | Default |
|----------|---------|---------|
| `REMBG_PROVIDER` | `cpu` \| `gpu` (o `cuda`) | `cpu` |

Usa `REMBG_PROVIDER=gpu` solo con VRAM de sobra (perfiles `normal`/`high`).

## Por qué no autodetecta la VRAM

Es tentador elegir proveedor según la VRAM libre, pero la sesión se crea con la
primera imagen y se cachea: un valor holgado en ese momento (Forge aún sin pesos)
llevaría a una sesión CUDA que falla en cuanto entra una generación. Preferimos
un default determinista y barato.

## Nota

No funciona con `BUILD_SLIM=1`: esa variante omite onnxruntime y `import rembg`
hace `sys.exit(1)`; el parche lo captura y se desactiva, dejando el aviso de la
propia extensión.
