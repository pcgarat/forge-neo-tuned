# Última modificación: 2026-10-09

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

# Resultados

| Backend | Endpoint | Resolución | Pasos | Semilla | Runs (s) | Mediana (s) | it/s | Pico VRAM (GB) | OOM |
|---|---|---|---|---|---|---|---|---|---|
