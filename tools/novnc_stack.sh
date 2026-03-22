#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOVNC_RESOLUTION="${NOVNC_RESOLUTION:-1920x1080}"
NOVNC_DEPTH="${NOVNC_DEPTH:-24}"
XVFB_SCREEN="${NOVNC_RESOLUTION}x${NOVNC_DEPTH}"
NOVNC_DISPLAY="${NOVNC_DISPLAY:-:1}"
NOVNC_DISPLAY_NUM="${NOVNC_DISPLAY#:}"
NOVNC_RFB_PORT="${NOVNC_RFB_PORT:-5901}"
NOVNC_WEB_PORT="${NOVNC_WEB_PORT:-6080}"
NOVNC_LOG_DIR="${NOVNC_LOG_DIR:-}"
if [ -n "$NOVNC_LOG_DIR" ]; then
  LOG_DIR="$NOVNC_LOG_DIR"
  mkdir -p "$LOG_DIR"
else
  DEFAULT_LOG_DIR="${HOME}/.cache/novnc"
  if mkdir -p "$DEFAULT_LOG_DIR" 2>/dev/null && [ -w "$DEFAULT_LOG_DIR" ]; then
    LOG_DIR="$DEFAULT_LOG_DIR"
  else
    LOG_DIR="/tmp/novnc-${USER}"
    mkdir -p "$LOG_DIR"
  fi
fi

if ! touch "$LOG_DIR/.writetest" 2>/dev/null; then
  LOG_DIR="/tmp/novnc-${USER}"
  mkdir -p "$LOG_DIR"
  touch "$LOG_DIR/.writetest"
fi
rm -f "$LOG_DIR/.writetest"
if [ ! -w "$LOG_DIR" ]; then
  LOG_DIR="/tmp/novnc-${USER}"
  mkdir -p "$LOG_DIR"
fi

CONDA_BIN="${CONDA_BIN:-}"

resolve_conda_bin() {
  if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN" ]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    return 0
  fi

  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"
  do
    if [ -x "$candidate" ]; then
      CONDA_BIN="$candidate"
      return 0
    fi
  done

  return 1
}

ensure_conda_bin() {
  if resolve_conda_bin; then
    return 0
  fi
  echo "error: conda executable not found in PATH or common locations." >&2
  echo "hint: set CONDA_BIN=/absolute/path/to/conda and retry." >&2
  return 1
}

stop_stack_processes() {
  pkill -f "websockify .*${NOVNC_WEB_PORT}" || true
  pkill -f "novnc_proxy --listen 127.0.0.1:${NOVNC_WEB_PORT}" || true
  pkill -f "x11vnc -display ${NOVNC_DISPLAY} -rfbport ${NOVNC_RFB_PORT}" || true
  pkill -f 'startfluxbox' || true
  pkill -f 'fluxbox' || true
  pkill -f "Xvfb ${NOVNC_DISPLAY} -screen 0" || true
}

stop_all_stacks() {
  pkill -f 'websockify .*127\.0\.0\.1:[0-9]+' || true
  pkill -f 'novnc_proxy --listen 127\.0\.0\.1:[0-9]+' || true
  pkill -f 'x11vnc -display :[0-9]+' || true
  pkill -f 'Xvfb :[0-9]+ -screen 0' || true
  pkill -f 'startfluxbox' || true
  pkill -f 'fluxbox' || true
  pkill -f 'python -m emergent.salmon_abm.realtime_viewer' || true
  pkill -f 'python -m emergent.salmon_abm.rl_training_viewer' || true
}

wait_for_display() {
  local tries=100
  while [ "$tries" -gt 0 ]; do
    if DISPLAY="$NOVNC_DISPLAY" xdpyinfo >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
    tries=$((tries - 1))
  done
  return 1
}

latest_h5() {
  local latest
  latest="$(ls -1t "$REPO_ROOT"/outputs/production/*.h5 "$REPO_ROOT"/outputs/test/*.h5 2>/dev/null | head -1 || true)"
  printf '%s' "$latest"
}

start() {
  stop_stack_processes
  rm -f "/tmp/.X${NOVNC_DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${NOVNC_DISPLAY_NUM}" || true

  # Use -ac so GUI apps launched from separate shell sessions can attach
  # to the virtual display without Xauthority handoff issues.
  echo "starting Xvfb display ${NOVNC_DISPLAY} at ${XVFB_SCREEN}"
  setsid -f Xvfb "$NOVNC_DISPLAY" -screen 0 "$XVFB_SCREEN" -ac >>"$LOG_DIR/xvfb.log" 2>&1
  if ! wait_for_display; then
    local alt_display=":99"
    if [ "$NOVNC_DISPLAY" = ":99" ]; then
      alt_display=":100"
    fi
    echo "error: Xvfb did not become ready on ${NOVNC_DISPLAY}" >&2
    echo "hint: if display is occupied, retry with NOVNC_DISPLAY=${alt_display}" >&2
    tail -n 80 "$LOG_DIR/xvfb.log" || true
    return 1
  fi

  setsid -f env DISPLAY="$NOVNC_DISPLAY" fluxbox >>"$LOG_DIR/fluxbox.log" 2>&1
  sleep 1
  setsid -f x11vnc -display "$NOVNC_DISPLAY" -rfbport "$NOVNC_RFB_PORT" -localhost -forever -shared -nopw >>"$LOG_DIR/x11vnc.log" 2>&1
  sleep 1
  setsid -f /usr/share/novnc/utils/novnc_proxy --listen "127.0.0.1:${NOVNC_WEB_PORT}" --vnc "localhost:${NOVNC_RFB_PORT}" >>"$LOG_DIR/novnc.log" 2>&1
  sleep 1

  status
}

stop() {
  stop_all_stacks
  echo "stopped"
}

status() {
  echo "processes:"
  pgrep -af "Xvfb ${NOVNC_DISPLAY}|fluxbox|x11vnc|websockify" || true
  echo "viewer:"
  pgrep -af 'python -m emergent.salmon_abm.realtime_viewer' || true
  pgrep -af 'python -m emergent.salmon_abm.rl_training_viewer' || true
  echo "ports:"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp | grep -E ":(${NOVNC_RFB_PORT}|${NOVNC_WEB_PORT})\\b" || true
  fi
  echo "display:"
  DISPLAY="$NOVNC_DISPLAY" xdpyinfo | sed -n '1,5p' || true
  echo "url:"
  echo "http://127.0.0.1:${NOVNC_WEB_PORT}/vnc.html"
}

logs() {
  for f in xvfb.log fluxbox.log x11vnc.log novnc.log viewer.log rl_viewer.log; do
    echo "--- $f ---"
    tail -n 60 "$LOG_DIR/$f" || true
  done
}

viewer() {
  local h5_path="${1:-}"
  if [ -z "$h5_path" ]; then
    h5_path="$(latest_h5)"
  fi

  start
  ensure_conda_bin
  pkill -f 'python -m emergent.salmon_abm.realtime_viewer' || true

  if [ -n "$h5_path" ]; then
    if [ ! -f "$h5_path" ]; then
      echo "error: output file not found: $h5_path" >&2
      return 1
    fi
    echo "launching viewer with: $h5_path"
    setsid -f env DISPLAY="$NOVNC_DISPLAY" "$CONDA_BIN" run -n emergent \
      python -m emergent.salmon_abm.realtime_viewer "$h5_path" >>"$LOG_DIR/viewer.log" 2>&1
  else
    echo "no .h5 found in outputs/production or outputs/test; launching empty viewer"
    setsid -f env DISPLAY="$NOVNC_DISPLAY" "$CONDA_BIN" run -n emergent \
      python -m emergent.salmon_abm.realtime_viewer >>"$LOG_DIR/viewer.log" 2>&1
  fi

  sleep 2
  if ! pgrep -af 'python -m emergent.salmon_abm.realtime_viewer' >/dev/null; then
    echo "error: viewer did not stay running. check logs:" >&2
    echo "  $0 logs" >&2
    return 1
  fi
  echo "viewer running"
  echo "open: http://127.0.0.1:${NOVNC_WEB_PORT}/vnc.html"
}

rlviewer() {
  start
  ensure_conda_bin
  pkill -f 'python -m emergent.salmon_abm.rl_training_viewer' || true
  setsid -f env DISPLAY="$NOVNC_DISPLAY" "$CONDA_BIN" run -n emergent \
    python -m emergent.salmon_abm.rl_training_viewer >>"$LOG_DIR/rl_viewer.log" 2>&1
  sleep 2
  if ! pgrep -af 'python -m emergent.salmon_abm.rl_training_viewer' >/dev/null; then
    echo "error: rl viewer did not stay running. check logs:" >&2
    echo "  $0 logs" >&2
    return 1
  fi
  echo "rl viewer running"
  echo "open: http://127.0.0.1:${NOVNC_WEB_PORT}/vnc.html"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  viewer) viewer "${2:-}" ;;
  rlviewer) rlviewer ;;
  *) echo "usage: $0 {start|stop|status|logs|viewer [h5_path]|rlviewer}"; exit 1 ;;
esac
