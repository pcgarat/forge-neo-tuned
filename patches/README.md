# Parches Forge Neo (Krea2)

| Patch | Qué hace |
|-------|----------|
| `krea2-features-backend.patch` | Moodboard + Identity Edit (hooks K2 / Qwen3-VL). Regenerado para Neo `41359cd…` (ver "Segunda regeneración" abajo). |
| `krea2-features-backend.original.patch` | Original del toolkit (solo referencia; no aplicar en neo actual). |
| `qwen35-vision-attention-fix.patch` | Fix (2 partes): (1) `attention_function(...)` mal llamado en visión Qwen3-VL → `TypeError: attention_flash() missing k,v,heads` (upstream ya lo arregló por su cuenta en `41359cd`, ver abajo); (2) fallback a `attention_pytorch` cuando el encoder visual corre en CPU (lowvram offload) → `NotImplementedError: flash_attn::_flash_attn_forward ... 'CPU' backend`. |
| `cmd-flags-numeric-types.patch` | Fix `/sdapi/v1/cmd-flags` (HTTP 500 `ResponseValidationError`): `FlagsModel` tipa cada flag como `type(default)` y los numéricos con `default=None` (`--port`, `--reserve-vram`, `--cuda-stream`) quedan como `str | None`; se relajan a `Any` esos campos. Ver `docs/attention-backends_09-10-2026.md`. |

---

# Parche `krea2-features-backend`

## Origen

- Toolkit fuente: [RedNodeAI/forge-neo-krea2-toolkit](https://github.com/RedNodeAI/forge-neo-krea2-toolkit) @ `8aac7a745202` (commit del que se extrajo el patch original).
- Ficheros que toca: `backend/diffusion_engine/krea.py`, `backend/nn/krea.py`, `backend/nn/llm/llama.py`, `backend/text_processing/qwen3vl_engine.py`.
- El patch original vendorizado se conserva como `krea2-features-backend.original.patch` (solo referencia histórica, **no aplicar** contra el `neo` actual).

## Por qué se regeneró

El patch original dejó de aplicar limpio (`git apply --check`) contra Forge Neo (`Haoming02/sd-webui-forge-classic`, rama `neo`) por deriva de contexto:

- El import `from backend.modules.k_prediction import PredictionDiscreteFlow` en `krea.py` fue eliminado en el commit `644450e` ("k_predictor"), que introdujo `ForgeDiffusionEngine._get_predictor()` en `backend/diffusion_engine/base.py` como punto único de selección de predictor para todos los motores (`flux.py`, `qwen.py`, `wan.py`, etc.), no solo Krea2.
- Ese import solo aparecía como contexto en el hunk original (no como línea añadida/eliminada por el propio patch), por lo que el conflicto era puramente de "contexto que ya no existe", no un choque real de lógica.
- Fuera de ese refactor, no se detectaron más cambios estructurales relevantes en los ~50 commits entre la base original del toolkit y el HEAD actual de `neo` para estos 4 ficheros.

## Cómo se regeneró

1. Se clonó `neo` con profundidad suficiente (150 commits) y se identificó el commit `44ae1a4fd0d76d829df9689224b6e59f0711e9cb` ("empty") como la última base donde el patch original aplicaba limpio.
2. Se aplicó el patch original sobre ese commit (`git apply --verbose`, éxito en los 4 ficheros) y se creó un commit temporal con el resultado.
3. Se hizo `git cherry-pick` de ese commit sobre el HEAD actual de `neo`. Esto permite que Git resuelva un merge de 3 vías real (base = `44ae1a4`, ours = HEAD, theirs = patch aplicado) en vez de un simple parcheo por coincidencia de texto/contexto. Los 4 ficheros se fusionaron **sin conflictos**.
4. Se revisó manualmente diff a diff el resultado frente al HEAD original para detectar incompatibilidades silenciosas con el resto del código (ver bug corregido más abajo).
5. Se generó el diff final: `git diff <SHA-original> <SHA-con-features> -- <4 ficheros> > krea2-features-backend.patch`.
6. Se verificó en un `git worktree` limpio e independiente del mismo SHA: `git apply --check --verbose` (éxito) + `git apply --verbose` (aplicado limpio) + `python3 -m py_compile` sobre los 4 ficheros resultantes.

## Bug corregido durante la fusión

El patch original usaba `dynamic_args.pop("ref_boosts", None)` y `dynamic_args.pop("ref_fit", None)` en `backend/diffusion_engine/krea.py`, asumiendo que `dynamic_args` era un `dict` plano. En el `neo` actual (`backend/args.py`), `dynamic_args` es una **clase con metaclase** (`_DynamicArgsMeta`) que emula `__getitem__` / `__setitem__` / `__contains__` / `.get()`, pero **no implementa `.pop()`**.

Esto habría lanzado `AttributeError: type object 'dynamic_args' has no attribute 'pop'` en **cualquier generación normal con Krea2** (esa rama de código se ejecuta siempre que no hay moodboard ni edit armados, es decir, en el caso por defecto — no solo en un edge case). Se reprodujo el fallo de forma aislada con la metaclase real antes de corregir.

Corrección aplicada (comportamiento equivalente, compatible con la metaclase):

```python
dynamic_args["ref_boosts"] = []
dynamic_args["ref_fit"] = []
```

## SHA pinned

- Forge Neo (`neo`) SHA verificado: `41359cd4b8b89212b3dbad8c9af719160a12ed63` (ver "Segunda regeneración" abajo; SHA anterior `97ff3a4024be2f0d5316f16e868e5ef822768872`).
- El patch aplica limpio contra ese SHA exacto. Si `neo` avanza, **re-verificar** con `git apply --check --verbose` antes de dar por bueno el build; no asumir compatibilidad indefinida.

## Segunda regeneración (`97ff3a4` → `41359cd`, 81 commits, 2026-07-23 a 2026-09-18)

### Por qué se regeneró

Al intentar actualizar `FORGE_NEO_REF` al HEAD de `neo`, ninguno de los dos patches aplicaba limpio. A diferencia de la primera regeneración (deriva de contexto), esta vez hubo **choques de lógica reales**: upstream añadió su propia versión, más simple, de features que este patch ya cubría de forma más completa:

- **`7c866142` ("Krea 2 Edit")**: upstream agregó soporte nativo de conditioning por imagen de referencia para img2img normal (`opts.krea2_do_reference`, `self.ref_latents`/`self.ini_latent` inicializados en `ForgeDiffusionEngine.__init__` de `backend/diffusion_engine/base.py`, método `get_learned_conditioning_with_image(prompt, images)` + `encode_vision`). Es un mecanismo **automático y de una sola referencia**, distinto y no gated por el mismo flujo que Moodboard (`arm_moodboard`, activado explícitamente por la extensión) o Identity Edit (`arm_edit`).
- La misma idea, más simple, en `backend/nn/krea.py`: upstream metió su propio in-context edit path (`dynamic_args.ref_latents`, sin `ref_boosts`/`ref_fit`, usando `_imgids`/`adaptive_resize`) en el `forward()` del DiT, en el mismo punto donde este patch ya tenía su versión superset (boosts por-referencia, geometría "fit", `_imgids_offset`, `_ref_attn_bias`).
- **`backend/nn/llm/qwen35.py`**: upstream arregló por su cuenta la Parte 1 del bug de `qwen35-vision-attention-fix` (el `TypeError` de `attention_function(...)` mal llamado), pero moviendo la llamada a `attention_function(q, k, v, heads, skip_reshape=True)` **dentro** de `Qwen35VisionAttention.forward`, ya no en el `forward()` externo de la vision transformer. La Parte 2 (fallback a `attention_pytorch` en CPU/lowvram) seguía sin existir upstream.
- **`backend/text_processing/qwen3vl_engine.py`**: `tokenize()` cambió de firma — de `tokenize(self, texts, images=[])` (lista de tensores, con `self.image_template`) a `tokenize(self, texts: list[str], images: int = 0)` (imágenes como **conteo**, sin `image_template`, prependiendo `vision_block * images` directo al texto).

### Cómo se regeneró

Mismo método que la primera vez (cherry-pick para forzar un merge de 3 vías real en vez de parcheo por texto):

1. Se aplicó el patch existente sobre el SHA viejo (`97ff3a4`) — aplicó limpio, confirmando que no había drift adicional respecto a esa base.
2. Se hizo `git cherry-pick` de ese commit sobre el HEAD nuevo de `neo` (`41359cd`). Resultado: `backend/nn/llm/llama.py` y `backend/attention.py` se fusionaron **sin conflicto** (cambios upstream en áreas distintas: MRoPE intercalado de Qwen3-VL, DeepStack, offset de `attention_flash` con máscara). Hubo conflicto real en los otros 4 ficheros.
3. Cada conflicto se resolvió a mano, no por "tomar un lado":
   - `backend/diffusion_engine/krea.py`: se **conservaron ambos** mecanismos de referencia. El método nativo de upstream se renombró a `get_learned_conditioning_with_start_image` (mismo cuerpo) para no chocar de nombre con el `get_learned_conditioning_with_image` del patch (que es la ruta Moodboard, firma distinta). `get_learned_conditioning` ahora prueba, en orden: Identity Edit armado → Moodboard armado → referencia nativa de img2img (`opts.krea2_do_reference`) → texto plano. La rama de referencia nativa también limpia `dynamic_args["ref_boosts"]`/`["ref_fit"]` al activarse, para que boosts de un Identity Edit previo no se filtren a una referencia automática de img2img (esas dos claves no las limpia `dynamic_args.reset()`, ver `backend/args.py`).
   - `backend/nn/krea.py`: se tomó entera la versión del patch (superset funcional: boosts por-referencia + geometría "fit"; no-op cuando no hay refs armadas, igual que la versión nativa de upstream en ese caso). Se eliminó el helper `_imgids` y el import de `adaptive_resize` de upstream, que quedaban sin uso.
   - `backend/nn/llm/qwen35.py`: la Parte 2 (fallback CPU-safe) se reubicó dentro de `Qwen35VisionAttention.forward`, en el nuevo punto donde upstream ya llama a `attention_function` directo — la Parte 1 ya no hace falta, upstream la resolvió.
   - `backend/text_processing/qwen3vl_engine.py`: `tokenize()` se reescribió sobre la firma nueva (`images: int`), conservando el comportamiento del patch de no duplicar el vision block cuando el texto (Moodboard/Edit) ya lo trae incluido — el check pasó de comparar contra `self.image_template` (eliminado upstream) a comparar contra el propio `self.vision_block` en el texto ya construido. El hunk de `process_tokens` (expansión de multipliers de emphasis cuando hay imágenes) no dependía de nada de esto y se tomó tal cual del patch.
4. Se generó el diff final igual que antes: `git diff 41359cd 9fce477e -- <ficheros>`.
5. Se verificó en un clone limpio e independiente del SHA nuevo: `git apply --check --verbose` + `git apply --verbose` (ambos patches, ambos limpios) + `python3 -m py_compile` sobre los 6 ficheros resultantes.

### Riesgo específico de esta regeneración

Esta vez el conflicto era de **diseño**, no de contexto: hubo que decidir cómo conviven dos implementaciones independientes de "conditioning por imagen de referencia" (la nativa de upstream, simple y automática, vs. la de este patch, explícita y con más control) sin que una pise el estado de la otra. La lógica se revisó con cuidado (ver arriba).

### Bug real encontrado al probar con GPU (no detectable por `py_compile`)

A diferencia de la primera regeneración, esta vez sí se hizo una build completa y un `txt2img` real contra un checkpoint Krea2 (GPU RTX 4060, ver "Riesgos funcionales" — sigue siendo la única cobertura de runtime que existe). Primer intento: `AttributeError: 'SingleStreamDiT' object has no attribute '_unpack_context'`.

Causa: al resolver el primer conflicto de `backend/nn/krea.py` (bloque `get_learned_conditioning`/`encode_vision` de `diffusion_engine/krea.py`, no este), se copió literal la línea `context = self._unpack_context(context.squeeze(1))` del lado "theirs" (patch viejo) sin verificar que `_unpack_context` siguiera existiendo. **No existía**: upstream la eliminó porque movió el trabajo de "desempaquetar" el contexto (de `(b, seq, 12*2560)` plano a `(b, seq, 12, 2560)`) a `Qwen3VLTextProcessingEngine.__call__`, en `qwen3vl_engine.py` — el hunk nuevo `b, seq, fuse = z.shape; ...; z = z.reshape(b * seq, 12, 2560)` (no tocado por este patch, ver tabla de ficheros) ya entrega el contexto pre-desempaquetado por línea; `SingleStreamDiT.forward` upstream ahora usa `context` tal cual, sin reshape. Fix: se borró la línea (no hace falta reemplazarla por nada, `TextFusionTransformer.forward` sigue esperando el mismo shape 4D, solo que ya llega así).

Esto confirma lo que ya advertía este documento: la verificación estática (`git apply --check` + `py_compile`) prueba que el código **parsea**, no que sus asunciones de shape/contrato con el resto del archivo sigan siendo válidas tras un merge de 3 vías. Cualquier línea copiada de un lado del conflicto sin revisar si lo que llama sigue existiendo en el otro lado es sospechosa por defecto.

## Ficheros que toca el patch regenerado

- `backend/diffusion_engine/krea.py`
- `backend/nn/krea.py`
- `backend/nn/llm/llama.py`
- `backend/text_processing/qwen3vl_engine.py`

## Parche `qwen35-vision-attention-fix`

En `backend/nn/llm/qwen35.py`, el port de ComfyUI hacía originalmente:

```python
optimized_attention = attention_function(x.device, mask=False, small_input=True)
```

En Forge Neo, `attention_function` ya es la implementación concreta (`attention_flash` / sage / …) con firma `(q, k, v, heads, …)`, no un factory. Esa llamada disparaba el `TypeError` al usar Moodboard/Identity Edit (ruta visión).

### Parte 1: asignación, no llamada — **ya no aplica, resuelto upstream**

Desde `41359cd` (ver "Segunda regeneración" arriba), upstream movió la llamada dentro de `Qwen35VisionAttention.forward` como `attention_function(q, k, v, self.num_heads, skip_reshape=True)` — llamada correcta, sin el factory roto. Esta parte del fix ya no forma parte del patch; solo queda la Parte 2, reubicada en ese mismo punto.

### Parte 2: fallback CPU-safe (lowvram)

Con solo la Parte 1, el crash de `TypeError` desaparece, pero `attention_function` sigue siendo una **variable global fijada una vez al importar el módulo** según hardware (`attention_flash` si hay CUDA + flash-attn instalado). Si el encoder visual de Qwen3-VL se ejecuta en CPU (offload por `--lowvram` o por presión de VRAM durante Moodboard/Identity Edit), `attention_flash` llama a un kernel `flash_attn::_flash_attn_forward` que **solo tiene implementación registrada para CUDA** (y `Meta`, para tracing). El resultado es:

```
NotImplementedError: Could not run 'flash_attn::_flash_attn_forward' with arguments from the 'CPU' backend.
```

`attention_flash` en `backend/attention.py` ya captura esta excepción y reintenta con `operations.scaled_dot_product_attention` (fallback interno), pero eso implica: (a) un log de `ERROR` en cada forward de cada bloque visual (ruido, aparenta fallo real), y (b) el coste de lanzar y capturar la excepción de flash-attn en cada capa, en vez de decidir el kernel una sola vez por forward. Se corrige eligiendo el kernel según el device del tensor de entrada, replicando el patrón que ComfyUI usa para encoders visuales que pueden correr offloaded:

```python
from backend.attention import attention_function, attention_pytorch
...
optimized_attention = attention_function if q.device.type == "cuda" else attention_pytorch
```

(Desde `41359cd` esto vive dentro del loop `for q, k, v in zip(q_splits, k_splits, v_splits)` de `Qwen35VisionAttention.forward` — antes de la segunda regeneración se calculaba una sola vez en el `forward()` externo sobre `x.device`; ver "Segunda regeneración" arriba.)

`attention_pytorch` es CPU/CUDA-safe (usa `torch.nn.functional.scaled_dot_product_attention` sin forzar backend) y tiene la misma firma `(q, k, v, heads, skip_reshape=...)` que consume `Qwen35VisionAttention.forward`, por lo que es un *drop-in* sin tocar el resto de la ruta de visión.

Se descartaron las otras dos opciones evaluadas:

- **Forzar el encoder visual a CUDA en `llama.py` (`preprocess_embed`) y mover el resultado de vuelta**: rompe el propósito del offload por lowvram (el encoder visual es la parte más pesada en VRAM de la ruta Krea2/Moodboard) y es un cambio de gestión de memoria, no de selección de kernel de atención.
- **Parchear `attention_flash` para hacer fallback interno por device**: ya lo hace vía `try/except` genérico, pero de forma reactiva (excepción + log de error en cada capa) en vez de proactiva (decidir una vez por forward), y mezclaría lógica de selección de dispositivo dentro de una función pensada como kernel puro de atención.

Aplica sobre el mismo `FORGE_NEO_REF` que el patch Krea2.

---

## Riesgos funcionales no cubiertos por esta regeneración

La verificación ha sido **estática** (no hay `torch` en el entorno de trabajo para levantar el WebUI real), tanto en la primera regeneración (`97ff3a4`) como en la segunda (`41359cd`):

- No se ha probado generación real (moodboard, identity edit, ni fallback normal ni referencia nativa de img2img) en runtime.
- El bug de `dynamic_args.pop()` es la única incompatibilidad estructural detectada por inspección manual en la primera regeneración, pero no descarta regresiones sutiles derivadas de los commits de diferencia entre la base del toolkit y el HEAD actual (LoRA, cuantización, refactor de attention/convrot, `img2img refactor`) que no tocan directamente estos ficheros pero podrían interactuar en runtime (p.ej. `attention_function`, `UnetPatcher`, paths de cuantización).
- De la segunda regeneración, el punto que más se beneficiaría de una prueba real: la convivencia entre Identity Edit / Moodboard y la referencia nativa de img2img (`opts.krea2_do_reference`) en `get_learned_conditioning` de `backend/diffusion_engine/krea.py` — la lógica de reseteo de `dynamic_args["ref_boosts"]`/`["ref_fit"]` al pasar de una a otra se razonó por inspección de código, no se ejecutó.
- Recomendado: probar en un entorno con GPU/`torch` (moodboard, identity edit, e img2img con `krea2_do_reference` activado, incluyendo alternar entre los tres en la misma sesión) antes de considerar esto listo para producción.
