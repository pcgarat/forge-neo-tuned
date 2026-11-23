#!/usr/bin/env python3
"""Benchmark de Krea 2 Identity Edit (Forge Neo + Qwen3-VL visión).

Mide N ediciones idénticas (misma semilla, misma imagen fuente) contra un Forge ya en
marcha y anexa una fila por configuración a un informe markdown.

El punto delicado de este camino es que **el LoRA de edición se activa solo si aparece
como `<lora:...>` en el prompt**: la extensión arma la referencia pero el modelo edita
sin LoRA si falta el tag, y el tiempo baja ~3x sin que nada falle. Por eso el benchmark
verifica en el log del contenedor que `[LORA] Loaded <nombre>` ocurrió durante la
medición, y aborta si no se cargó. Sin esa comprobación los números no significan nada.

No arranca ni reinicia el contenedor; eso lo orquesta el Makefile (perfiles A/B).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_attn import _http_json, fetch_memory, ooms_from_memory, vram_peak_from_memory  # noqa: E402
from chatbot_warmup import wait_for_api  # noqa: E402

_GB = 1024**3
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_AR_MODES = (
    "match source (resize output)",
    "crop source to output AR",
    "fit source to output (v1.2)",
    "off",
)
_DEFAULT_PROMPT = "take this person and make them stand in a bright modern kitchen, natural light"
_DEFAULT_LORA = "krea2_identity_edit_v1_2_r64"


class BenchError(RuntimeError):
    """API caída, petición fallida o precondición incumplida."""


@dataclass
class EditResult:
    label: str
    width: int
    height: int
    steps: int
    seed: int
    grounding_px: int
    ref_boost: float
    lora: str | None = None
    te_device: str = "auto"
    loras_seen: list[str] = field(default_factory=list)
    runs: list[float] = field(default_factory=list)
    vram_peak: float | None = None
    ooms: int | None = None

    @property
    def median(self) -> float:
        return statistics.median(self.runs) if self.runs else float("nan")

    @property
    def minimum(self) -> float:
        return min(self.runs) if self.runs else float("nan")

    @property
    def it_s(self) -> float:
        return self.steps / self.median if self.runs and self.median > 0 else float("nan")


def lora_tag(name: str, weight: float = 1.0) -> str:
    """Tag que la extensión y Forge usan para aplicar el LoRA de edición."""
    return f"<lora:{name}:{weight:g}>"


def inject_lora(prompt: str, name: str | None, weight: float = 1.0) -> str:
    """Prefija el tag del LoRA si hay nombre y no estaba ya en el prompt."""
    if not name:
        return prompt
    tag = lora_tag(name, weight)
    if tag in prompt:
        return prompt
    return f"{tag} {prompt}"


def edit_script_args(
    grounding_px: int,
    ref_boost: float,
    ar_mode: str,
    source_b64: str,
    *,
    second_b64: str | None = None,
) -> list[Any]:
    """Argumentos posicionales del accordion, en el orden que declara `krea2_edit.py`."""
    return [
        True,  # enable
        source_b64,  # source image
        second_b64,  # 2nd reference (opcional)
        grounding_px,
        ar_mode,
        ref_boost,
        1.0,  # ref_boost (scene)
        False,  # auto face-ref prep (2 pasadas anidadas: no medir con esto activo)
    ]


def build_edit_body(
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    source_b64: str,
    grounding_px: int,
    ref_boost: float,
    ar_mode: str,
    lora: str | None,
    prompt: str = _DEFAULT_PROMPT,
    cfg_scale: float = 1.0,
    sampler_name: str = "Euler",
    scheduler: str = "Simple",
) -> dict[str, Any]:
    """Payload de txt2img con el accordion armado. Semilla y fuente fijas por comparación."""
    return {
        "prompt": inject_lora(prompt, lora),
        "negative_prompt": "",
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": False,
        "send_images": False,
        "do_not_save_samples": True,
        "do_not_save_grid": True,
        "alwayson_scripts": {
            "Krea2 Identity Edit": {
                "args": edit_script_args(grounding_px, ref_boost, ar_mode, source_b64)
            }
        },
    }


def active_checkpoint(options: dict[str, Any]) -> str:
    return str(options.get("sd_model_checkpoint") or "")


def is_krea2_checkpoint(name: str) -> bool:
    """Heurística: los checkpoints de Krea 2 en este árbol llevan `krea2` en el nombre."""
    return "krea2" in name.lower().replace(" ", "")


def require_krea2_checkpoint(options: dict[str, Any]) -> str:
    """Aborta si la UI no tiene cargado un checkpoint de Krea 2.

    `override_settings.sd_model_checkpoint` devuelve HTTP 500 en Neo 2.29 (bug del
    endpoint), así que no se puede fijar por petición: sin esta comprobación el benchmark
    mediría el modelo que hubiera cargado. Con otro modelo, los LoRA de Krea 2 dan
    `[LORA] Mismatch` y el edit se cae a los ~3 min con un error engañoso.
    """
    current = active_checkpoint(options)
    if not current:
        raise BenchError(
            "no se pudo leer el checkpoint activo de /sdapi/v1/options; "
            "carga un Krea 2 en la UI antes de medir."
        )
    if not is_krea2_checkpoint(current):
        raise BenchError(
            f"el checkpoint activo es {current!r}, no un Krea 2. Cárgalo en la UI "
            "(o con `make krea2`) y vuelve a lanzar: con otro modelo los LoRA de Krea 2 dan "
            "`[LORA] Mismatch`, el edit no se arma y la medición no vale."
        )
    return current


def te_device_from_flags(flags: dict[str, Any]) -> str:
    """Dónde corre el text encoder, tal como lo reporta `/cmd-flags`.

    Con `--lowvram` y sin `--text-enc-device` Forge lo pone en CPU, y eso domina el
    tiempo cuando hay tokens de visión: es la variable que más cambia el resultado.
    """
    explicit = flags.get("text_enc_device")
    if explicit:
        return str(explicit)
    if flags.get("lowvram"):
        return "cpu (implícito por --lowvram)"
    return "auto"


def lora_loaded(log_text: str, name: str | None) -> bool | None:
    """Compatibilidad: detecta `[LORA] Loaded` en el log del contenedor.

    Deprecado como guardián: Forge solo loguea la carga la **primera** vez que aplica un
    LoRA, así que con el LoRA ya residente la línea no aparece y daría un falso negativo.
    La verificación buena es `verify_edit_infotext` sobre el `info` de la respuesta.
    """
    if not name:
        return None
    clean = _ANSI.sub("", log_text or "")
    stem = name[:-len(".safetensors")] if name.endswith(".safetensors") else name
    pattern = re.compile(rf"\[LORA\] Loaded {re.escape(stem)}(\.safetensors)?\b")
    return bool(pattern.search(clean))


_LORA_HASHES = re.compile(r"Lora hashes:\s*\"([^\"]*)\"")
_KREA2_EDIT = re.compile(r"Krea2 Edit:\s*\"(\d+)\s+source\(s\)")
_KREA2_GROUNDING = re.compile(r"grounding_px\s+(\d+)")
_MODEL = re.compile(r"Model:\s*([^,\n]+)")


def parse_infotext(raw: str) -> dict[str, Any]:
    """Extrae del infotext las pruebas de que el edit fue real.

    - `Lora hashes: "n: h"` → el LoRA estaba aplicado (sin el tag en el prompt no aparece).
    - `Krea2 Edit: "N source(s)"` → la referencia se armó y llegó al DiT.
    """
    text = raw or ""
    out: dict[str, Any] = {"loras": [], "sources": None, "grounding_px": None}
    hashes = _LORA_HASHES.search(text)
    if hashes:
        for part in hashes.group(1).split(","):
            name = part.split(":", 1)[0].strip()
            if name:
                out["loras"].append(name)
    sources = _KREA2_EDIT.search(text)
    if sources:
        out["sources"] = int(sources.group(1))
    grounding = _KREA2_GROUNDING.search(text)
    if grounding:
        out["grounding_px"] = int(grounding.group(1))
    return out


def verify_edit_infotext(info: dict[str, Any], lora: str | None) -> dict[str, Any]:
    """Aborta si la respuesta no demuestra que hubo Identity Edit de verdad.

    Devuelve el infotext parseado (el llamante usa `loras` para registrar todos los que
    se aplicaron, no solo el esperado)."""
    infotexts = info.get("infotexts") or []
    raw = infotexts[0] if infotexts else ""
    parsed = parse_infotext(raw)
    if parsed["sources"] is None:
        model = _MODEL.search(raw)
        loaded = model.group(1).strip() if model else "desconocido"
        raise BenchError(
            "la respuesta no contiene `Krea2 Edit: \"N source(s)\"`: la referencia no llegó "
            f"al modelo. Checkpoint que generó: {loaded!r}. Comprueba que sea un Krea 2 y que "
            "el LoRA de edición no diera `[LORA] Mismatch` (un LoRA de Krea 2 sobre otro "
            "modelo no aplica y el edit no se arma)."
        )
    if lora:
        def stem(name: str) -> str:
            return name[:-len(".safetensors")] if name.endswith(".safetensors") else name

        wanted = stem(lora)
        found = {stem(n) for n in parsed["loras"]}
        if wanted not in found:
            raise BenchError(
                f"el LoRA {lora!r} no aparece en `Lora hashes` del infotext "
                f"(encontrados: {parsed['loras'] or 'ninguno'}): se habría medido el camino "
                "sin edición, que es ~3x más rápido. Revisa --lora y que el fichero exista."
            )
    return parsed


def container_log_since(container: str, since_epoch: float) -> str:
    """Log del contenedor desde un instante. Vacío si docker no está disponible."""
    try:
        done = subprocess.run(
            ["docker", "logs", "--since", f"{int(since_epoch)}", container],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (done.stdout or "") + (done.stderr or "")


def read_source_b64(path: str) -> str:
    raw = Path(path).read_bytes()
    if not raw:
        raise BenchError(f"imagen fuente vacía: {path}")
    return base64.b64encode(raw).decode("ascii")


def run_edit_once(base_url: str, body: dict[str, Any], timeout: float) -> tuple[float, dict[str, Any]]:
    """Lanza una edición y devuelve (segundos, `info` de la respuesta)."""
    url = f"{base_url.rstrip('/')}/sdapi/v1/txt2img"
    t0 = time.perf_counter()
    response = _http_json("POST", url, payload=body, timeout=timeout)
    elapsed = time.perf_counter() - t0
    info = response.get("info") if isinstance(response, dict) else None
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    return elapsed, info if isinstance(info, dict) else {}


def benchmark(
    base_url: str,
    *,
    label: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    runs: int,
    source_b64: str,
    grounding_px: int,
    ref_boost: float,
    ar_mode: str,
    lora: str | None,
    container: str = "",
    te_device: str = "auto",
    prompt: str = _DEFAULT_PROMPT,
    warmup: bool = True,
    timeout: float = 3600.0,
) -> EditResult:
    body = build_edit_body(
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        source_b64=source_b64,
        grounding_px=grounding_px,
        ref_boost=ref_boost,
        ar_mode=ar_mode,
        lora=lora,
        prompt=prompt,
    )
    # La verificación usa el `info` de cada respuesta (autoritativo). El log se mantiene
    # como dato extra opcional: Forge solo loguea la carga la primera vez.
    result = EditResult(
        label=label,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        grounding_px=grounding_px,
        ref_boost=ref_boost,
        lora=lora,
        te_device=te_device,
    )
    if warmup:
        print(f"[{label}] calentamiento…", file=sys.stderr, flush=True)
        _, info = run_edit_once(base_url, body, timeout)
        parsed = verify_edit_infotext(info, lora)
        if parsed["loras"]:
            result.loras_seen = parsed["loras"]

    for i in range(1, runs + 1):
        elapsed, info = run_edit_once(base_url, body, timeout)
        parsed = verify_edit_infotext(info, lora)
        if parsed["loras"]:
            result.loras_seen = parsed["loras"]
        result.runs.append(elapsed)
        print(f"[{label}] run {i}/{runs}: {elapsed:.2f}s", file=sys.stderr, flush=True)

    if container:
        # Informativo: ausencia de la línea NO implica que no se aplicara.
        if lora_loaded(container_log_since(container, time.time() - 900), lora) is False:
            print(
                "[aviso] el log no muestra la carga del LoRA; normal si ya estaba residente.",
                file=sys.stderr,
                flush=True,
            )

    memory = fetch_memory(base_url)
    result.vram_peak = vram_peak_from_memory(memory)
    result.ooms = ooms_from_memory(memory)
    return result


_TABLE_HEADER = (
    "| Config | Endpoint | Resolución | Pasos | grounding_px | ref_boost | LoRA | TE | "
    "Runs (s) | Mediana (s) | Pico VRAM (GB) | OOM |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def format_rows(results: list[EditResult]) -> str:
    rows = []
    for r in results:
        runs = ", ".join(f"{x:.2f}" for x in r.runs)
        vram = f"{r.vram_peak / _GB:.2f}" if r.vram_peak else "n/d"
        oom = "n/d" if r.ooms is None else str(r.ooms)
        # Todos los LoRA aplicados según el infotext: el de edición y cualquier extra que
        # venga en el prompt (p. ej. el distill de 4 pasos). Nunca solo el esperado.
        lora = ", ".join(r.loras_seen) if r.loras_seen else (r.lora or "—")
        rows.append(
            f"| {r.label} | identity-edit | {r.width}x{r.height} | {r.steps} | "
            f"{r.grounding_px} | {r.ref_boost:g} | {lora} | {r.te_device} | "
            f"{runs} | {r.median:.2f} | {vram} | {oom} |"
        )
    return "\n".join(rows)


def format_table(results: list[EditResult]) -> str:
    rows = format_rows(results)
    return _TABLE_HEADER + (rows + "\n" if rows else "")


def append_report(path: Path, results: list[EditResult]) -> None:
    """Crea o amplía el informe. Una sola cabecera: cada anexado añade filas."""
    today = time.strftime("%Y-%m-%d")
    new_rows = format_rows(results)
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = f"# Última modificación: {today}"
        idx = next((i for i, line in enumerate(lines) if line.startswith("| Config |")), None)
        if idx is None:
            body = "\n".join(lines).rstrip() + "\n\n# Resultados\n\n" + _TABLE_HEADER + new_rows + "\n"
        else:
            existing = [
                line
                for line in lines[idx + 2 :]
                if line.startswith("|") and not line.startswith("| Config |")
            ]
            body = (
                "\n".join(lines[:idx]).rstrip()
                + "\n\n"
                + _TABLE_HEADER
                + "\n".join(existing + new_rows.splitlines())
                + "\n"
            )
        path.write_text(body, encoding="utf-8")
        return
    path.write_text(
        f"# Última modificación: {today}\n\n"
        "# Benchmark de Krea 2 Identity Edit\n\n"
        "Generado con `make bench-edit` (misma semilla, misma imagen fuente, mismos pasos).\n"
        "El pico de VRAM es acumulado desde el arranque del contenedor, no por ejecución.\n"
        "El LoRA debe ir en el prompt (`<lora:...>`): sin el tag el modelo edita sin LoRA y el\n"
        "tiempo baja ~3x. El benchmark lo verifica en el `info` de la respuesta (`Lora hashes`\n"
        "y `Krea2 Edit: \"N source(s)\"`) y aborta si falta la prueba.\n\n"
        "# Resultados\n\n" + _TABLE_HEADER + new_rows + "\n",
        encoding="utf-8",
    )


def _parse_ar(value: str) -> str:
    if value not in _AR_MODES:
        raise argparse.ArgumentTypeError(f"AR inválido {value!r}; usa uno de: {', '.join(_AR_MODES)}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--container", default="", help="Contenedor a inspeccionar para verificar el LoRA")
    parser.add_argument("--label", default="identity-edit")
    parser.add_argument("--out", default="", help="Fichero markdown donde anexar resultados")
    parser.add_argument("--source", required=True, help="Imagen fuente del edit")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--lora", default=_DEFAULT_LORA, help="Nombre del LoRA ('' = sin LoRA)")
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-check-checkpoint",
        action="store_true",
        help="No exige un Krea 2 cargado (solo para depurar el propio benchmark)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--grounding-px", type=int, default=768)
    parser.add_argument("--ref-boost", type=float, default=4.0)
    parser.add_argument("--ar-mode", type=_parse_ar, default=_AR_MODES[2])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    parser.add_argument("--generate-timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    try:
        wait_for_api(args.base_url, args.wait_timeout)
        from bench_attn import fetch_cmd_flags_or_empty

        flags = fetch_cmd_flags_or_empty(args.base_url)
        # Falla rápido si la UI no tiene un Krea 2 cargado: con otro modelo la medición
        # tarda minutos y el error resultante es engañoso.
        if not args.no_check_checkpoint:
            try:
                options = _http_json("GET", f"{args.base_url.rstrip('/')}/sdapi/v1/options", timeout=60)
            except BenchError as exc:
                raise BenchError(f"no se pudo leer /sdapi/v1/options: {exc}") from exc
            require_krea2_checkpoint(options if isinstance(options, dict) else {})
        source_b64 = read_source_b64(args.source)
        result = benchmark(
            args.base_url,
            label=args.label,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            runs=args.runs,
            source_b64=source_b64,
            grounding_px=args.grounding_px,
            ref_boost=args.ref_boost,
            ar_mode=args.ar_mode,
            lora=args.lora or None,
            container=args.container,
            te_device=te_device_from_flags(flags),
            prompt=args.prompt,
            warmup=not args.no_warmup,
            timeout=args.generate_timeout,
        )
    except BenchError as exc:
        print(f"Benchmark falló: {exc}", file=sys.stderr)
        return 1

    table = format_table([result])
    print(table, end="")
    if args.out:
        append_report(Path(args.out), [result])
        print(f"Resultados anexados a {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
