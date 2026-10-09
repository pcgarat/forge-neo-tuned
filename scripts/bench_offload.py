#!/usr/bin/env python3
"""Benchmark del trasiego de pesos RAM<->VRAM (offload) en Forge Neo.

Mide la curva it/s vs resolución del perfil activo y, leyendo el log del contenedor,
el volumen de pesos descargado por ciclo de lowvram. Sirve para comparar perfiles que
solo cambian el camino de offload (`--cuda-stream`, `--pin-shared-memory`) y para
distinguir régimen limitado por transferencia (curva it/s plana) de limitado por
cómputo (la curva cae al crecer los píxeles).

No arranca ni reinicia el contenedor; eso lo orquesta `make bench-offload-sweep`.
Escribe/actualiza un informe markdown (un bloque por perfil).
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_attn import (  # noqa: E402
    BenchError,
    build_body,
    fetch_memory,
    ooms_from_memory,
    run_once,
    vram_peak_from_memory,
)
from chatbot_warmup import parse_last_gen, read_params_txt, wait_for_api  # noqa: E402

_GB = 1024**3
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Líneas de lowvram de backend/memory_management.py: el "usable" solo aparece si el
# modelo no cupo entero. El resto de campos es estable, así que el prefijo es opcional.
_LOADED_PARTIALLY = re.compile(
    r"loaded partially;\s*(?:[\d.]+ MB usable,\s*)?([\d.]+) MB loaded,\s*"
    r"([\d.]+) MB offloaded,\s*([\d.]+) MB buffer reserved,\s*lowvram patches:\s*(\d+)"
)
_UNLOADED_PARTIALLY = re.compile(r"Unloaded partially:\s*([\d.]+) MB freed")
_PINNED = re.compile(r"Pinned Memory:\s*(\d+) MB")
_STREAMS = re.compile(r"Using async weight offloading with\s*(\d+) streams")
_TOTAL_MEM = re.compile(r"Total VRAM\s*(\d+) MB,\s*total RAM\s*(\d+) MB")


@dataclass
class SizeResult:
    label: str
    width: int
    height: int
    steps: int
    seed: int
    runs: list[float] = field(default_factory=list)
    vram_peak: float | None = None
    ooms: int | None = None

    @property
    def median(self) -> float:
        return statistics.median(self.runs) if self.runs else float("nan")

    @property
    def it_s(self) -> float:
        return self.steps / self.median if self.runs and self.median > 0 else float("nan")


@dataclass
class OffloadStats:
    pinned_memory_mb: int | None = None
    streams: int | None = None
    total_vram_mb: int | None = None
    total_ram_mb: int | None = None
    loaded_partially: int = 0
    max_loaded_mb: float = 0.0
    max_offloaded_mb: float = 0.0
    max_buffer_mb: float = 0.0
    max_patches: int = 0
    unloaded_events: int = 0
    total_freed_mb: float = 0.0

    @property
    def seen(self) -> bool:
        return self.loaded_partially > 0 or self.unloaded_events > 0 or self.streams is not None


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        width, sep, height = part.partition("x")
        if not sep or not width.isdigit() or not height.isdigit():
            raise ValueError(f"resolución inválida {part!r} (formato ANCHOxALTO)")
        sizes.append((int(width), int(height)))
    if not sizes:
        raise ValueError(f"sin resoluciones en {spec!r}")
    return sizes


def parse_offload_log(text: str) -> OffloadStats:
    """Extrae del log de Forge el pinning, los streams y el volumen descargado."""
    stats = OffloadStats()
    clean = _ANSI.sub("", text or "")
    for match in _LOADED_PARTIALLY.finditer(clean):
        loaded = float(match.group(1))
        offloaded = float(match.group(2))
        buffer_mb = float(match.group(3))
        stats.loaded_partially += 1
        stats.max_loaded_mb = max(stats.max_loaded_mb, loaded)
        stats.max_offloaded_mb = max(stats.max_offloaded_mb, offloaded)
        stats.max_buffer_mb = max(stats.max_buffer_mb, buffer_mb)
        stats.max_patches = max(stats.max_patches, int(match.group(4)))
    for match in _UNLOADED_PARTIALLY.finditer(clean):
        stats.unloaded_events += 1
        stats.total_freed_mb += float(match.group(1))
    if match := _PINNED.search(clean):
        stats.pinned_memory_mb = int(match.group(1))
    if match := _STREAMS.search(clean):
        stats.streams = int(match.group(1))
    if match := _TOTAL_MEM.search(clean):
        stats.total_vram_mb = int(match.group(1))
        stats.total_ram_mb = int(match.group(2))
    return stats


def _shell(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def probe_context(container: str = "") -> dict[str, str]:
    """Contexto de la máquina/contenedor. Best-effort: nunca debe tumbar el benchmark."""
    gpu = _shell(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.gen.max,"
            "pcie.link.width.current,pcie.link.width.max",
            "--format=csv,noheader",
        ]
    )
    try:
        swappiness = Path("/proc/sys/vm/swappiness").read_text().strip()
    except OSError:
        swappiness = "n/d"
    memlock = _shell(["docker", "exec", container, "sh", "-c", "ulimit -l"]) if container else ""
    return {
        "gpu": gpu or "n/d",
        "swappiness": swappiness or "n/d",
        "memlock_kb": memlock or "n/d",
    }


def read_container_log(container: str, lines: int = 20000) -> str:
    return _shell(["docker", "logs", "--tail", str(lines), container], timeout=60.0)


def benchmark_sizes(
    base_url: str,
    *,
    label: str,
    sizes: list[tuple[int, int]],
    steps: int,
    seed: int,
    runs: int,
    checkpoint: str | None = None,
    warmup: bool = True,
    timeout: float = 900.0,
) -> list[SizeResult]:
    results: list[SizeResult] = []
    for width, height in sizes:
        body = build_body(width, height, steps, seed, checkpoint=checkpoint)
        tag = f"{label} {width}x{height}"
        if warmup:
            print(f"[{tag}] calentamiento…", file=sys.stderr, flush=True)
            run_once(base_url, "txt2img", body, timeout)
        result = SizeResult(label=label, width=width, height=height, steps=steps, seed=seed)
        for i in range(1, runs + 1):
            elapsed = run_once(base_url, "txt2img", body, timeout)
            result.runs.append(elapsed)
            print(f"[{tag}] run {i}/{runs}: {elapsed:.2f}s", file=sys.stderr, flush=True)
        memory = fetch_memory(base_url)
        result.vram_peak = vram_peak_from_memory(memory)
        result.ooms = ooms_from_memory(memory)
        results.append(result)
    return results


def format_table(results: list[SizeResult]) -> str:
    header = (
        "| Perfil | Resolución | Pasos | Semilla | Runs (s) | Mediana (s) | it/s | "
        "Pico VRAM (GB) | OOM |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        run_times = ", ".join(f"{x:.2f}" for x in r.runs)
        vram = f"{r.vram_peak / _GB:.2f}" if r.vram_peak else "n/d"
        oom = "n/d" if r.ooms is None else str(r.ooms)
        rows.append(
            f"| {r.label} | {r.width}x{r.height} | {r.steps} | {r.seed} | {run_times} | "
            f"{r.median:.2f} | {r.it_s:.2f} | {vram} | {oom} |"
        )
    return header + "\n".join(rows) + "\n"


def format_context(context: dict[str, str], stats: OffloadStats) -> str:
    gpu = context.get("gpu", "n/d")
    pinned = f"{stats.pinned_memory_mb} MB" if stats.pinned_memory_mb is not None else "no aplicado"
    streams = f"{stats.streams} streams" if stats.streams else "desactivado"
    rows = [
        ("GPU (nombre, VRAM, PCIe gen/width actual/máx)", gpu),
        ("vm.swappiness", context.get("swappiness", "n/d")),
        ("ulimit -l (memlock, contenedor)", f"{context.get('memlock_kb', 'n/d')} KB"),
        ("Pinning según el log", pinned),
        ("Offload asíncrono según el log", streams),
    ]
    lines = ["| Contexto | Valor |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines) + "\n"


def format_offload_summary(stats: OffloadStats) -> str:
    if not stats.seen:
        return "Sin líneas de offload en el log (¿log vacío o modelo que cabía entero?).\n"
    lines = [
        f"- Cargas parciales (`loaded partially`): {stats.loaded_partially} — máx "
        f"{stats.max_offloaded_mb:.0f} MB descargados, {stats.max_buffer_mb:.0f} MB de buffer, "
        f"{stats.max_patches} parches lowvram.",
        f"- Descargas parciales (`Unloaded partially`): {stats.unloaded_events} — "
        f"{stats.total_freed_mb:.0f} MB liberados en total.",
    ]
    if stats.total_vram_mb and stats.total_ram_mb:
        lines.append(
            f"- Reportado por Forge: VRAM total {stats.total_vram_mb} MB, "
            f"RAM total {stats.total_ram_mb} MB."
        )
    return "\n".join(lines) + "\n"


def append_report(
    path: Path,
    *,
    label: str,
    context: dict[str, str],
    results: list[SizeResult],
    stats: OffloadStats,
) -> None:
    """Crea o amplía el informe markdown. La primera línea lleva la fecha de modificación."""
    today = time.strftime("%Y-%m-%d")
    block = (
        f"## Perfil `{label}`\n\n"
        + format_context(context, stats)
        + "\n"
        + format_table(results)
        + "\n"
        + format_offload_summary(stats)
        + "\n"
    )
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = f"# Última modificación: {today}"
        body = "\n".join(lines).rstrip() + "\n"
        body = body + "\n" + block if "# Resultados" in body else body + "\n# Resultados\n\n" + block
        path.write_text(body, encoding="utf-8")
        return
    path.write_text(
        f"# Última modificación: {today}\n\n"
        "# Benchmark de trasiego de pesos (offload RAM<->VRAM)\n\n"
        "Generado con `make bench-offload` (misma semilla y mismos pasos por resolución).\n"
        "Con `--lowvram` el coste dominante es la transferencia: una curva it/s plana indica\n"
        "régimen limitado por offload, y una curva que cae al crecer los píxeles, por cómputo.\n\n"
        "# Resultados\n\n" + block,
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="", help="DATA_PATH (params.txt del último gen)")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--label", default="offload", help="Etiqueta del perfil medido")
    parser.add_argument("--sizes", default="768x768,1024x1024,1280x1280", help="ANCHOxALTO separadas por comas")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--container", default="", help="Contenedor: leer su log y sus ulimits")
    parser.add_argument("--log-file", default="", help="Log ya volcado (alternativa a --container)")
    parser.add_argument("--out", default="", help="Fichero markdown donde anexar resultados")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    parser.add_argument("--generate-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        print(f"Argumentos inválidos: {exc}", file=sys.stderr)
        return 2

    checkpoint = None
    if args.data_path:
        checkpoint = parse_last_gen(read_params_txt(args.data_path)).get("sd_model_checkpoint")

    try:
        wait_for_api(args.base_url, args.wait_timeout)
        results = benchmark_sizes(
            args.base_url,
            label=args.label,
            sizes=sizes,
            steps=args.steps,
            seed=args.seed,
            runs=args.runs,
            checkpoint=checkpoint,
            warmup=not args.no_warmup,
            timeout=args.generate_timeout,
        )
    except BenchError as exc:
        print(f"Benchmark falló: {exc}", file=sys.stderr)
        return 1

    if args.log_file:
        log_text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
    elif args.container:
        log_text = read_container_log(args.container)
    else:
        log_text = ""
    stats = parse_offload_log(log_text)
    context = probe_context(args.container)

    print(format_table(results), end="")
    print()
    print(format_offload_summary(stats), end="")

    if args.out:
        append_report(
            Path(args.out), label=args.label, context=context, results=results, stats=stats
        )
        print(f"Resultados anexados a {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
