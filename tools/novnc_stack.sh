#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${HOME}/.cache/novnc"
mkdir -p "$LOG_DIR"

start() {
  pkill -f 'Xvfb :1 -screen 0 1920x1080x24' || true
  pkill -f 'x11vnc -display :1 -rfbport 5901' || true
  pkill -f 'websockify .*6080' || true
  pkill -f 'startfluxbox' || true
  rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 || true

  setsid -f Xvfb :1 -screen 0 1920x1080x24 >>"$LOG_DIR/xvfb.log" 2>&1
  sleep 1
  setsid -f env DISPLAY=:1 fluxbox >>"$LOG_DIR/fluxbox.log" 2>&1
  sleep 1
  setsid -f x11vnc -display :1 -rfbport 5901 -localhost -forever -shared -nopw >>"$LOG_DIR/x11vnc.log" 2>&1
  sleep 1
  setsid -f /usr/share/novnc/utils/novnc_proxy --listen 127.0.0.1:6080 --vnc localhost:5901 >>"$LOG_DIR/novnc.log" 2>&1
  sleep 1

  status
}

stop() {
  pkill -f 'websockify .*6080' || true
  pkill -f 'x11vnc -display :1 -rfbport 5901' || true
  pkill -f 'startfluxbox' || true
  pkill -f 'Xvfb :1 -screen 0 1920x1080x24' || true
  echo "stopped"
}

status() {
  echo "processes:"
  pgrep -af 'Xvfb :1|fluxbox|x11vnc|websockify' || true
  echo "ports:"
  ss -ltnp | rg ':5901|:6080' || true
  echo "display:"
  DISPLAY=:1 xdpyinfo | sed -n '1,5p' || true
  echo "url:"
  echo "http://127.0.0.1:6080/vnc.html"
}

logs() {
  for f in xvfb.log fluxbox.log x11vnc.log novnc.log; do
    echo "--- $f ---"
    tail -n 60 "$LOG_DIR/$f" || true
  done
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  *) echo "usage: $0 {start|stop|status|logs}"; exit 1 ;;
esac
