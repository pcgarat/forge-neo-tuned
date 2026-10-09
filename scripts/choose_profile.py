#!/usr/bin/env python3
"""Asistente de arranque: pregunta lo que importa y recomienda los ejes de `make run`.

Resuelve `ATTN` (backend de atención), `VRAM` (perfil de memoria) y `STREAM` (solape del
offload) a partir de lo que de verdad cambia la decisión: modelo, imagen vs vídeo, tamaño,
uso por API y longitud de la secuencia. Lo usa `make up-interactive`.

Tres modos:
  - Interactivo (por defecto): pregunta por stdin y resume por stderr.
  - `--make`: imprime solo `VAR=valor` por stdout, para `eval "$(make up-interactive)"`.
  - No interactivo: pasa las respuestas por flags (`--model`, `--resolution`, ...).

El resumen y las preguntas van a **stderr** a propósito: así stdout queda limpio para el eval.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

MODELS = {
    "klein": "Flux.2 Klein 9B turbo",
    "krea2": "Krea 2 turbo",
    "wan": "Wan 2.2 turbo",
}
DEFAULT_KIND = {"klein": "imagen", "krea2": "imagen", "wan": "video"}
# A partir de este lado mayor, CK INT8 empieza a rendir en imagen (el offload deja de dominar).
HIGH_RES_PX = 1280

_RES = re.compile(r"^(\d+)(?:x(\d+))?$", re.IGNORECASE)


@dataclass
class Answers:
    model: str = "klein"
    kind: str = "imagen"
    resolution: int = 1024  # lado mayor, en píxeles
    api: bool = False
    long_sequence: bool = False
    stream: bool = True  # solape de offload (--cuda-stream); off solo para medir


@dataclass
class Recommendation:
    attn: str
    attn_reason: str
    vram: str = "auto"
    warmup: bool = False
    stream: bool = True
    notes: list[str] = field(default_factory=list)

    def make_lines(self) -> list[str]:
        lines = [f"ATTN={self.attn}", f"VRAM={self.vram}"]
        if self.warmup:
            lines.append("WARMUP=1")
        if not self.stream:
            lines.append("STREAM=off")
        return lines


def parse_resolution(value: str) -> int:
    """Acepta `1024` o `1280x720` y devuelve el lado mayor."""
    match = _RES.match(value.strip())
    if not match:
        raise ValueError(f"resolución inválida {value!r} (usa 1024 o 1280x720)")
    width = int(match.group(1))
    height = int(match.group(2)) if match.group(2) else width
    return max(width, height)


def recommend(a: Answers) -> Recommendation:
    """Decide ATTN/VRAM/warmup y las notas según las respuestas."""
    is_video = a.kind == "video" or a.model == "wan"
    if is_video:
        attn, reason = "ck", "vídeo: ~32k tokens, la atención domina"
    elif a.long_sequence:
        attn, reason = "ck", "secuencias largas (Moodboard/Identity Edit): más contexto"
    elif a.resolution >= HIGH_RES_PX:
        attn, reason = "ck", f"imagen a {a.resolution} px: CK INT8 rinde (26,3→21,9 s medidos)"
    else:
        attn, reason = "flash", "secuencias cortas: CK/Sage no aportan y el offload domina"

    rec = Recommendation(attn=attn, attn_reason=reason, warmup=a.api, stream=a.stream)

    if a.model == "krea2":
        rec.notes.append("No añadas --fast-fp8 (falla en Krea 2).")
        if a.long_sequence:
            rec.notes.append(
                "Krea 2 + TE visión movido a VRAM: en 8 GB es muy justo; "
                "usa fp8_scaled del TE y vigila la VRAM."
            )
    if is_video and attn == "ck":
        rec.notes.append(
            "No lo combines con Sparse Attention de la UI: instala un override que sustituye el backend."
        )
    if attn == "ck":
        rec.notes.append("CK INT8 no es bit-exacto: a igual semilla la imagen cambia.")
    if a.api:
        rec.notes.append("Warmup torch.compile (guard_filter_fn) al size del último gen.")
    if not a.stream:
        rec.notes.append(
            "Offload asíncrono desactivado (STREAM=off): para comparar, no para uso normal."
        )
    return rec


def _ask(prompt: str, default, cast=str):
    # El prompt va a stderr: stdout lo captura el `eval` del Makefile y se lo tragaría.
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        print(f"{prompt}{suffix}: ", end="", file=sys.stderr, flush=True)
        try:
            raw = input().strip()
        except EOFError:
            print(file=sys.stderr)
            return default
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError as exc:
            print(f"  Valor inválido: {exc}", file=sys.stderr)


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    while True:
        raw = str(_ask(f"{prompt} ({'/'.join(choices)})", default)).strip().lower()
        if raw in choices:
            return raw
        print(f"  Elige una de: {'/'.join(choices)}", file=sys.stderr)


def _ask_bool(prompt: str, default: bool = False) -> bool:
    default_raw = "s" if default else "n"
    while True:
        raw = str(_ask(f"{prompt} (s/n)", default_raw)).strip().lower()
        if raw in ("s", "si", "sí", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Responde s o n.", file=sys.stderr)


def interactive() -> Answers:
    model = _ask_choice("¿Qué modelo vas a usar?", list(MODELS), "klein")
    kind = _ask_choice("¿Imagen o vídeo?", ["imagen", "video"], DEFAULT_KIND[model])
    resolution = _ask("¿Lado mayor de la imagen en píxeles (ej. 1024, 1280)", 1024, parse_resolution)
    long_sequence = _ask_bool(
        "¿Secuencias largas? (Moodboard/Identity Edit, muchas referencias)", False
    )
    api = _ask_bool("¿Lotes repetidos por API a la misma resolución?", False)
    stream = _ask_bool(
        "¿Solapar el trasiego de pesos RAM↔VRAM? (--cuda-stream; 'no' solo para medir)", True
    )
    return Answers(
        model=model,
        kind=kind,
        resolution=resolution,
        api=api,
        long_sequence=long_sequence,
        stream=stream,
    )


def format_summary(a: Answers, rec: Recommendation) -> str:
    lines = [
        "Configuración recomendada",
        f"  Modelo:   {MODELS[a.model]} ({a.kind})",
        f"  Tamaño:   lado mayor {a.resolution} px",
        f"  Atención: {rec.attn} — {rec.attn_reason}",
        f"  VRAM:     {rec.vram} (nvidia-smi decide el perfil)",
        f"  Offload:  {'solapado (--cuda-stream)' if rec.stream else 'sin solapar (STREAM=off)'}",
        f"  Warmup:   {'sí' if rec.warmup else 'no'}",
    ]
    if rec.notes:
        lines.append("Notas:")
        lines += [f"  - {note}" for note in rec.notes]
    warmup = " WARMUP=1" if rec.warmup else ""
    stream = "" if rec.stream else " STREAM=off"
    lines += [
        "",
        f"Arrancando con: make run ATTN={rec.attn} VRAM={rec.vram}{warmup}{stream}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), help="Responde sin preguntar")
    parser.add_argument("--kind", choices=["imagen", "video"])
    parser.add_argument("--resolution", type=parse_resolution, default=None)
    parser.add_argument("--long-sequence", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--no-stream", action="store_true", help="Desactiva --cuda-stream (medición)")
    parser.add_argument("--make", action="store_true", help="Imprime solo VAR=valor (para eval)")
    args = parser.parse_args(argv)

    if args.model:
        answers = Answers(
            model=args.model,
            kind=args.kind or DEFAULT_KIND[args.model],
            resolution=args.resolution or 1024,
            api=args.api,
            long_sequence=args.long_sequence,
            stream=not args.no_stream,
        )
    else:
        if not sys.stdin.isatty():
            print("Sin terminal interactiva: pasa al menos --model.", file=sys.stderr)
            return 2
        try:
            answers = interactive()
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130

    rec = recommend(answers)
    print(format_summary(answers, rec), file=sys.stderr)
    if args.make:
        print("\n".join(rec.make_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
