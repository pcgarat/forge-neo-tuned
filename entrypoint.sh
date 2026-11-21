#!/bin/sh
set -e
# Misma raíz de datos en local (Compose → /data) y RunPod (volume en /workspace).
# Compose define DATA_DIR=/data; en RunPod el default es /workspace/forge-data.
data_dir="${DATA_DIR:-/workspace/forge-data}"
export COMMANDLINE_ARGS="$(echo "$COMMANDLINE_ARGS" | sed "s|/data|$data_dir|g")"

mkdir -p "$data_dir" "$data_dir/extensions" "$data_dir/models" "$data_dir/output" \
  "$data_dir/cache" "$data_dir/tmp"

# Evitar JSONDecodeError: Forge lee config.json y falla si existe pero está vacío
for f in config.json ui-config.json; do
  path="$data_dir/$f"
  if [ ! -f "$path" ] || [ ! -s "$path" ]; then
    printf '{}' > "$path"
  fi
done

if [ -n "$EXTRA_ARGS" ]; then
  export COMMANDLINE_ARGS="$COMMANDLINE_ARGS $EXTRA_ARGS"
fi

# Bajar a PUID:PGID para que output/cache en el host no queden root:root.
# El entrypoint debe arrancar como root (sin `user:` en compose); Forge escribe
# también bajo /app/webui (p. ej. backend/huggingface).
puid="${PUID:-1000}"
pgid="${PGID:-1000}"
video_gid="${VIDEO_GID:-44}"
render_gid="${RENDER_GID:-992}"

# rembg (quitar fondo): el Dockerfile lo hornea, esto solo cubre imágenes publicadas
# antes de ese cambio. Si falta, se instala en un venv persistente del volumen y se
# añade por PYTHONPATH. El venv usa --system-site-packages para arrastrar únicamente
# lo que falta (rembg + jsonschema/pooch/pymatting/scikit-image, ~50 MB): no toca
# numpy/pillow/scipy de la imagen y el siguiente arranque no pisa nada. Se paga una
# vez; tras recrear el contenedor la comprobación es local y no depende de PyPI.
# INSTALL_REMBG=0 lo desactiva (volumen de solo lectura o sin red).
rembg_version="${REMBG_VERSION:-2.0.87}"
rembg_venv="$data_dir/packages/rembg-venv"

rembg_missing() {
  ! python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("rembg") else 1)'
}

# El venv del volumen ya sirve: no hay que reinstalar nada (ni tocar PyPI).
rembg_venv_ready() {
  [ -x "$rembg_venv/bin/python" ] \
    && "$rembg_venv/bin/python" -c 'import rembg' >/dev/null 2>&1
}

if [ "${INSTALL_REMBG:-auto}" != "0" ] && rembg_missing && ! rembg_venv_ready; then
  # rembg hace sys.exit(1) sin backend ONNX: inútil en BUILD_SLIM (sin onnxruntime).
  if ! python -c 'import onnxruntime' >/dev/null 2>&1; then
    echo "[deps] onnxruntime ausente (BUILD_SLIM): se omite rembg"
  elif [ "$(id -u)" -ne 0 ]; then
    echo "[deps] AVISO: rembg falta y sin root no se puede instalar; usa 'make build'" >&2
  else
    echo "[deps] rembg no está en la imagen; instalando rembg==$rembg_version en $rembg_venv"
    mkdir -p "$data_dir/packages"
    # Venv reutilizable: pip resuelve sin volver a bajar lo ya instalado. Un venv a
    # medias de un intento fallido no importa rembg, así que se rehace desde cero.
    rm -rf "$rembg_venv"
    python -m venv --system-site-packages "$rembg_venv" >/dev/null 2>&1 || rm -rf "$rembg_venv"
    if [ -x "$rembg_venv/bin/python" ] \
       && "$rembg_venv/bin/python" -m pip install --quiet --no-cache-dir "rembg==$rembg_version"; then
      chown -R "$puid:$pgid" "$rembg_venv" 2>/dev/null || true
      echo "[deps] rembg==$rembg_version listo (venv cacheado; no se repite al reiniciar)"
    else
      echo "[deps] AVISO: no se pudo instalar rembg; la extensión quedará desactivada" >&2
    fi
  fi
fi

# Solo cuando la imagen no lo trae: el venv aporta rembg y las deps que faltaban.
if [ -x "$rembg_venv/bin/python" ] && rembg_missing \
   && rembg_site="$("$rembg_venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"; then
  export PYTHONPATH="$rembg_site${PYTHONPATH:+:$PYTHONPATH}"
fi

if [ "$(id -u)" -eq 0 ]; then
  # No hacer chown -R de todo DATA_PATH (incluye Models enormes).
  # Solo rutas que Forge escribe en runtime + leftovers root.
  for d in output cache tmp; do
    if [ -e "$data_dir/$d" ]; then
      chown -R "$puid:$pgid" "$data_dir/$d" 2>/dev/null || true
    fi
  done
  for f in config.json ui-config.json params.txt; do
    if [ -e "$data_dir/$f" ]; then
      chown "$puid:$pgid" "$data_dir/$f" 2>/dev/null || true
    fi
  done
  # Necesario: decompress de tokenizers/modelos embebidos bajo la imagen
  if [ -d /app/webui ]; then
    chown -R "$puid:$pgid" /app/webui
  fi

  # setpriv no cambia HOME: el proceso hereda HOME=/root y Triton/HF fallan con
  # PermissionError al crear /root/.triton (uid PUID no puede escribir ahí).
  home_dir="$(getent passwd "$puid" 2>/dev/null | cut -d: -f6)"
  if [ -z "$home_dir" ] || [ ! -d "$home_dir" ]; then
    home_dir="$data_dir/home"
    mkdir -p "$home_dir"
    chown "$puid:$pgid" "$home_dir"
  fi
  export HOME="$home_dir"
  mkdir -p "$data_dir/cache/triton" "$data_dir/cache/torchinductor"
  chown -R "$puid:$pgid" "$data_dir/cache/triton" "$data_dir/cache/torchinductor" 2>/dev/null || true
  export TRITON_CACHE_DIR="$data_dir/cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$data_dir/cache/torchinductor"
  # Triton JIT: si no hay CC explícito busca gcc/cc en PATH.
  if [ -z "${CC:-}" ] && command -v gcc >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
  fi
  if [ -z "${CXX:-}" ] && command -v g++ >/dev/null 2>&1; then
    export CXX="$(command -v g++)"
  fi

  groups="$pgid,$video_gid,$render_gid"
  exec setpriv --reuid="$puid" --regid="$pgid" --groups="$groups" -- "$@"
fi

exec "$@"
