"""Fuerza a rembg a inferir en CPU y reutiliza las sesiones ONNX.

Tres problemas de la extensión RemBG que este parche corrige, todos visibles en
el perfil 8 GB (--lowvram):

1. VRAM. rembg elige CUDAExecutionProvider en cuanto onnxruntime-gpu está
   instalado, sin consultar la VRAM libre. Con Forge ocupando ~6.6 de 8 GB,
   onnxruntime no consigue reservar su arena y el postprocesado revienta con
   `BFCArena::AllocateRawInternal ... Failed to allocate memory`. La excepción
   sale de dentro de `InferenceSession.run()`, así que no hay fallback: la sesión
   ya se creó "con éxito" y falla a mitad de inferencia.
   En CPU, u2net resuelve 320x320 en ~0.3 s: despreciable frente a una
   generación, y la GPU queda entera para Forge.

2. Sesión por imagen. La extensión llama a `rembg.new_session()` dentro de
   `process()`, así que cada postprocesado paga la carga del .onnx y una reserva
   nueva de memoria. Aquí se cachea una sesión por modelo.

3. U2NET_HOME. Solo el script de postprocesado lo fija; los manejadores de la API
   (`scripts/api.py`) no, así que sus .onnx acababan en el HOME del contenedor y
   se perdían al recrearlo. Aquí se fija al directorio de modelos de Forge.

No se autodetecta VRAM a propósito: la sesión se cachea en la primera imagen, y
la VRAM de ese instante no predice la del resto de la sesión. CPU por defecto,
GPU solo si se pide explícitamente.
"""

from __future__ import annotations

import os
import threading

_DEFAULT_MODEL = "bria-rmbg"
_CPU_PROVIDERS = ("CPUExecutionProvider",)
# CPU de respaldo: si el arena de CUDA falla a mitad de sesión, onnxruntime cae.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

_sessions: dict = {}
_lock = threading.Lock()


def providers() -> list:
    """Proveedores ONNX según REMBG_PROVIDER (cpu por defecto)."""
    if os.environ.get("REMBG_PROVIDER", "cpu").strip().lower() in {"gpu", "cuda"}:
        return list(_CUDA_PROVIDERS)
    return list(_CPU_PROVIDERS)


def _session_key(model_name: str, args: tuple, kwargs: dict) -> tuple:
    """Clave de caché: el modelo y lo que altere qué .onnx se carga."""
    extra = sorted(
        (name, repr(value)) for name, value in kwargs.items() if name != "providers"
    )
    return (model_name, repr(args), tuple(extra))


def _install_u2net_home() -> None:
    """Modelos bajo DATA_PATH/Models/u2net (respeta --data-dir, vale en RunPod)."""
    try:
        from modules.paths_internal import models_path
    except Exception:
        return
    os.environ.setdefault("U2NET_HOME", os.path.join(models_path, "u2net"))


def apply() -> bool:
    """Parchea `rembg.new_session`. Idempotente; False si rembg no está instalado."""
    _install_u2net_home()

    try:
        import rembg
        from rembg import bg, session_factory
    except (ImportError, SystemExit):
        # rembg hace sys.exit(1) si falta el backend ONNX (caso de BUILD_SLIM).
        return False

    if getattr(rembg, "_forge_rembg_patch", False):
        return True

    original = session_factory.new_session

    def new_session(model_name: str = _DEFAULT_MODEL, *args, sess_opts=None, **kwargs):
        kwargs["providers"] = providers()
        key = _session_key(model_name, args, kwargs)
        with _lock:
            session = _sessions.get(key)
            if session is None:
                session = original(model_name, *args, sess_opts=sess_opts, **kwargs)
                _sessions[key] = session
            return session

    # bg.py hizo `from .session_factory import new_session`, así que guarda su
    # propia referencia: hay que reasignar en el paquete y en los dos módulos.
    for module in (session_factory, bg, rembg):
        module.new_session = new_session
    rembg._forge_rembg_patch = True
    return True


try:
    if not apply() and os.environ.get("REMBG_PROVIDER"):
        # Solo avisa si alguien pidió un proveedor y no hay rembg donde aplicarlo.
        print("[rembg-cpu] rembg no instalado; parche no aplicado")
except Exception as exc:  # noqa: BLE001 - Forge debe arrancar aunque el parche falle
    print(f"[rembg-cpu] no se pudo aplicar el parche: {exc}")
