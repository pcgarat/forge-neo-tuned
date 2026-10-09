# Última modificación: 2026-10-09

# Benchmark de trasiego de pesos (offload RAM↔VRAM)

Generado con `make bench-offload` (misma semilla y mismos pasos por resolución).
Con `--lowvram` el coste dominante es la transferencia: una curva it/s plana indica
régimen limitado por offload, y una curva que cae al crecer los píxeles, por cómputo.

Cómo se genera:

- `make bench-offload` — mide el perfil activo y anexa un bloque. Args por defecto:
  `--sizes 768x768,1024x1024,1280x1280 --steps 8 --runs 3`.
- `make bench-offload-sweep` — compara el perfil 8 GB sin `--cuda-stream` (`8gb-nostream`)
  frente a con él (`8gb-stream`), reiniciando entre ambos.
- `make bench-offload ARGS=` mediante `BENCH_OFFLOAD_ARGS="..."` para fijar resoluciones.

Notas:

- El **pico de VRAM** que reporta `/sdapi/v1/memory` es acumulado desde el arranque del contenedor,
  no por ejecución. Sirve para comparar perfiles arrancados de cero.
- Los **MB descargados** salen del log del contenedor (`loaded partially` / `Unloaded partially`).
  Si el modelo cabe entero en VRAM no habrá líneas: fuerza `--lowvram` para medir.
- Con `--lowvram` el cuello es la transferencia: curva it/s plana → offload-bound; curva que cae → cómputo.
- `--pin-shared-memory` necesita `ulimit -l` holgado (ver `docker-compose.yml`); si no, no fija nada.

# Resultados

