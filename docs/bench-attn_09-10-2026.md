# Última modificación: 2026-10-10

# Benchmark de backends de atención

Generado con `make bench-attn` (misma semilla, mismos pasos, mismas dimensiones). Anexa una fila
por ejecución; para comparar, los perfiles base/CK/Sage se miden en el mismo informe.

Cómo se genera:

- `make bench-attn-sweep` — base (`flash`) vs CK INT8, reiniciando el contenedor entre ambos.
- `make bench-attn ARGS="--width 1280 --height 1280 --steps 8 --runs 3"` — medir el backend activo.
- `make bench-attn-sage` — construye la imagen con Sage y mide `sage-fp16_cuda`.

Notas:

- El **pico de VRAM** que reporta `/sdapi/v1/memory` es acumulado desde el arranque del contenedor,
  no por ejecución. Sirve para comparar perfiles arrancados de cero, no para medir una corrida aislada.
- A baja resolución el backend apenas cambia el tiempo (el `--lowvram` domina). Medir en el caso que
  interese: **Wan (vídeo)** y **Krea 2 a ≥1280px**.
- CK INT8 y Sage **no son bit-exactos**: a igual semilla la imagen cambia. Compara también calidad.
- Medición 2026-10-10: Krea 2 Int4 ConvRot v10 Turbo, RTX 4060 8 GB, perfil `8gb` (`--lowvram
  --cuda-stream`), 8 pasos, semilla 12345, 3 corridas tras 1 de calentamiento, a 1024x1024 y 1280x1280.

# Resultados

| Backend | Endpoint | Resolución | Pasos | Semilla | Runs (s) | Mediana (s) | it/s | Pico VRAM (GB) | OOM |
|---|---|---|---|---|---|---|---|---|---|
| flash | txt2img | 1280x1280 | 8 | 12345 | 25.58, 24.44, 25.77 | 25.58 | 0.31 | 3.94 | 0 |
| ck-int8 | txt2img | 1280x1280 | 8 | 12345 | 18.65, 18.63, 18.64 | 18.64 | 0.43 | 3.94 | 0 |
| sage-fp16_cuda | txt2img | 1280x1280 | 8 | 12345 | 20.68, 20.71, 20.72 | 20.71 | 0.39 | 3.84 | 0 |
| sage-fp16_triton | txt2img | 1280x1280 | 8 | 12345 | 40.15, 40.21, 40.28 | 40.21 | 0.20 | 4.09 | 0 |
| flash | txt2img | 1024x1024 | 8 | 12345 | 14.66, 14.55, 14.50 | 14.55 | 0.55 | 4.12 | 0 |
| ck-int8 | txt2img | 1024x1024 | 8 | 12345 | 12.89, 12.82, 12.96 | 12.89 | 0.62 | 4.09 | 0 |
| sage-fp16_cuda | txt2img | 1024x1024 | 8 | 12345 | 13.85, 13.76, 13.78 | 13.78 | 0.58 | 4.09 | 0 |
| sage-fp16_triton | txt2img | 1024x1024 | 8 | 12345 | 13.56, 13.35, 13.50 | 13.50 | 0.59 | 4.06 | 0 |
