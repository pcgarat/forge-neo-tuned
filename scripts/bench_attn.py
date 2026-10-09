#!/usr/bin/env python3
"""Benchmark del backend de atención de Forge Neo.

Mide el tiempo de pared de N generaciones idénticas (misma semilla) contra un Forge
ya en marcha, y anota el pico de VRAM y los OOM. Sirve para comparar perfiles que
solo cambian el backend de atención (`make lowvram` / `make ck` / `make sage`).

No arranca ni reinicia el contenedor; eso lo orquesta `make bench-attn-sweep`.
Escribe/actualiza un informe markdown (una fila por backend) para comparar de un vistazo.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chatbot_warmup import parse_last_gen, read_params_txt, wait_for_api  # noqa: E402

_DEFAULT_PROMPT = "attention benchmark"
_GB = 1024**3


class BenchError(RuntimeError):
    """API caída o petición fallida."""


@dataclass
class BenchResult:
    label: str
    endpoint: str
    width: int
    height: int
    steps: int
    seed: int
    runs: list[float] = field(default_factory=list)
    vram_peak: float | None = None
    ooms: int | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def median(self) -> float:
        return statistics.median(self.runs) if self.runs else float("nan")

    @property
    def minimum(self) -> float:
        return min(self.runs) if self.runs else float("nan")

    @property
    def maximum(self) -> float:
        return max(self.runs) if self.runs else float("nan")

    @property
    def it_s(self) -> float:
        return self.steps / self.median if self.runs and self.median > 0 else float("nan")


def backend_label(cmd_flags: dict[str, Any], override: str | None) -> str:
    """Etiqueta del backend. `override` manda; si no, se deduce de los flags de arranque.

    Solo distingue con fiabilidad CK / PyTorch SDPA / Sage con función explícita. Con los
    flags por defecto no se puede saber si el paquete activo es Flash o xformers, así que
    el Makefile pasa `--label` en cada perfil.
    """
    if override:
        return override
    if cmd_flags.get("use_ck_attention"):
        return "ck-int8"
    if cmd_flags.get("use_pytorch_cross_attention"):
        return "pytorch-sdpa"
    sage_function = cmd_flags.get("sage_function")
    if sage_function and sage_function != "auto":
        return f"sage-{sage_function}"
    return "default"


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise BenchError(f"HTTP {exc.code} {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise BenchError(f"Red {url}: {exc}") from exc


def fetch_cmd_flags(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    body = _http_json("GET", f"{base_url.rstrip('/')}/sdapi/v1/cmd-flags", timeout=timeout)
    return body if isinstance(body, dict) else {}


def fetch_memory(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    body = _http_json("GET", f"{base_url.rstrip('/')}/sdapi/v1/memory", timeout=timeout)
    return body if isinstance(body, dict) else {}


def build_body(
    width: int,
    height: int,
    steps: int,
    seed: int,
    *,
    cfg_scale: float = 1.0,
    sampler_name: str = "Euler",
    checkpoint: str | None = None,
    prompt: str = _DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Payload fijo salvo el backend. Semilla fija para que la comparación sea limpia."""
    body: dict[str, Any] = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_name": sampler_name,
        "save_images": False,
        "send_images": False,
    }
    if checkpoint:
        body["override_settings"] = {"sd_model_checkpoint": checkpoint}
        body["override_settings_restore_afterwards"] = True
    return body


def run_once(base_url: str, endpoint: str, body: dict[str, Any], timeout: float) -> float:
    url = f"{base_url.rstrip('/')}/sdapi/v1/{endpoint}"
    t0 = time.perf_counter()
    _http_json("POST", url, payload=body, timeout=timeout)
    return time.perf_counter() - t0


def vram_peak_from_memory(memory: dict[str, Any]) -> float | None:
    """Pico de VRAM reservada en bytes según /memory (None si no es CUDA)."""
    cuda = (memory or {}).get("cuda") or {}
    reserved = cuda.get("reserved") or {}
    peak = reserved.get("peak")
    return float(peak) if isinstance(peak, (int, float)) else None


def ooms_from_memory(memory: dict[str, Any]) -> int | None:
    cuda = (memory or {}).get("cuda") or {}
    events = cuda.get("events") or {}
    oom = events.get("oom")
    return int(oom) if isinstance(oom, (int, float)) else None


def benchmark(
    base_url: str,
    *,
    label: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    runs: int,
    endpoint: str = "txt2img",
    checkpoint: str | None = None,
    cfg_scale: float = 1.0,
    sampler_name: str = "Euler",
    warmup: bool = True,
    timeout: float = 900.0,
) -> BenchResult:
    body = build_body(
        width,
        height,
        steps,
        seed,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        checkpoint=checkpoint,
    )
    if warmup:
        print(f"[{label}] calentamiento…", file=sys.stderr, flush=True)
        run_once(base_url, endpoint, body, timeout)

    result = BenchResult(
        label=label,
        endpoint=endpoint,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
    )
    for i in range(1, runs + 1):
        elapsed = run_once(base_url, endpoint, body, timeout)
        result.runs.append(elapsed)
        print(f"[{label}] run {i}/{runs}: {elapsed:.2f}s", file=sys.stderr, flush=True)

    memory = fetch_memory(base_url)
    result.vram_peak = vram_peak_from_memory(memory)
    result.ooms = ooms_from_memory(memory)
    return result


def format_table(results: list[BenchResult]) -> str:
    header = (
        "| Backend | Endpoint | Resolución | Pasos | Semilla | Runs (s) | Mediana (s) | "
        "it/s | Pico VRAM (GB) | OOM |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        runs = ", ".join(f"{x:.2f}" for x in r.runs)
        vram = f"{r.vram_peak / _GB:.2f}" if r.vram_peak else "n/d"
        oom = "n/d" if r.ooms is None else str(r.ooms)
        rows.append(
            f"| {r.label} | {r.endpoint} | {r.width}x{r.height} | {r.steps} | {r.seed} | "
            f"{runs} | {r.median:.2f} | {r.it_s:.2f} | {vram} | {oom} |"
        )
    return header + "\n".join(rows) + "\n"


def append_report(path: Path, results: list[BenchResult]) -> None:
    """Crea o amplía el informe markdown. La primera línea lleva la fecha de modificación."""
    today = time.strftime("%Y-%m-%d")
    table = format_table(results)
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = f"# Última modificación: {today}"
        body = "\n".join(lines).rstrip() + "\n"
        if "# Resultados" in body:
            body = body + "\n" + table
        else:
            body = body + "\n# Resultados\n\n" + table
        path.write_text(body, encoding="utf-8")
        return
    path.write_text(
        f"# Última modificación: {today}\n\n"
        "# Benchmark de backends de atención\n\n"
        "Generado con `make bench-attn` (misma semilla, mismos pasos). Comparar por columnas.\n"
        "El pico de VRAM es acumulado desde el arranque del contenedor, no por ejecución.\n\n"
        "# Resultados\n\n" + table,
        encoding="utf-8",
    )


def resolve_size(
    args: argparse.Namespace, data_path: str
) -> tuple[int, int, int, str | None]:
    """Resolución, pasos y checkpoint: de los argumentos; si faltan, del último gen."""
    width, height, steps, checkpoint = args.width, args.height, args.steps, None
    if width and height and steps:
        return width, height, steps, checkpoint
    fields = parse_last_gen(read_params_txt(data_path))
    checkpoint = fields.get("sd_model_checkpoint")
    return (
        width or fields.get("width", 1024),
        height or fields.get("height", 1024),
        steps or 8,
        checkpoint,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="", help="DATA_PATH (params.txt del último gen)")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--label", default="", help="Etiqueta del backend (si vacío, se deduce)")
    parser.add_argument("--out", default="", help="Fichero markdown donde anexar resultados")
    parser.add_argument("--endpoint", default="txt2img", choices=["txt2img", "img2img"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", default="Euler")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    parser.add_argument("--generate-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    try:
        wait_for_api(args.base_url, args.wait_timeout)
        flags = fetch_cmd_flags(args.base_url)
        label = backend_label(flags, args.label.strip() or None)
        width, height, steps, checkpoint = resolve_size(args, args.data_path)
        result = benchmark(
            args.base_url,
            label=label,
            width=width,
            height=height,
            steps=steps,
            seed=args.seed,
            runs=args.runs,
            endpoint=args.endpoint,
            checkpoint=checkpoint,
            cfg_scale=args.cfg_scale,
            sampler_name=args.sampler,
            warmup=not args.no_warmup,
            timeout=args.generate_timeout,
        )
        result.flags = {
            k: flags.get(k)
            for k in ("use_ck_attention", "use_pytorch_cross_attention", "sage_function")
        }
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
