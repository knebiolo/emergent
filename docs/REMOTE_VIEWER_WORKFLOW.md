# Remote Viewer Workflow (PowerShell + noVNC)

Use this workflow when running on a remote compute node with no physical display and a managed work laptop.

## Viewer Scope (Important)

- `realtime_viewer` is a playback UI for existing `.h5` files.
- Number of agents and number of timesteps are set in the simulation command, not in the viewer UI.
- Viewer controls are play/pause/scrub/open-file.

## Terminal Layout

Use three terminals total:
- Laptop PowerShell: SSH tunnel only.
- Node terminal A: noVNC stack + viewer.
- Node terminal B: simulation run.

## 1) Start tunnel from laptop (PowerShell)

If using the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start-tunnel.ps1
```

Manual equivalent (replace values; do not use `<...>` placeholders literally):

```powershell
Start-Process "http://127.0.0.1:6080/vnc.html"
ssh -N -L 6080:127.0.0.1:6080 kevinnebiolo@192.168.102.157
```

Keep this PowerShell window open while using the viewer.

## 2) Start virtual desktop + viewer on node (single command)

```bash
cd /home/kevinnebiolo/emergent
tools/novnc_stack.sh viewer
```

For laptop-sized browser windows, prefer starting the virtual desktop at a laptop resolution:

```bash
cd /home/kevinnebiolo/emergent
NOVNC_RESOLUTION=1366x768 tools/novnc_stack.sh rlviewer
```

Behavior:
- Restarts `Xvfb + fluxbox + x11vnc + noVNC`.
- Launches the realtime viewer on `DISPLAY=:1`.
- If no file is provided, auto-opens the most recent `.h5` from `outputs/production/` or `outputs/test/`.

Optional: launch a specific output file:

```bash
tools/novnc_stack.sh viewer outputs/production/<model_name>.h5
```

## 3) Run simulation in a second node terminal

HECRAS direct mode (preferred):

```bash
cd /home/kevinnebiolo/emergent
conda run -n emergent python tools/run_salmon_production.py \
  --nagents 2000 \
  --nsteps 500 \
  --dt 1.0 \
  --hecras-plan /home/kevinnebiolo/emergent/data/salmon_abm/<plan_file>.hdf \
  --start-polygon /home/kevinnebiolo/emergent/data/salmon_abm/shapes/start_loc_river_right.shp \
  --longitudinal-profile /home/kevinnebiolo/emergent/data/salmon_abm/shapes/longitudinal.shp \
  --hecras-time-mode loop \
  --backend process \
  --write-mode minimal
```

Notes:
- In this canonical runner, async + `--write-mode minimal` still writes `agent_data/X`, `agent_data/Y`, `agent_data/battery`, and `agent_data/heading` for fatigue-aware playback.
- Keep viewer and simulation in separate node terminals.

## 4) Transfer large gitignored inputs to node

```powershell
scp "C:\path\to\plan.hdf" <user>@<node-host-or-ip>:/home/<user>/emergent/data/salmon_abm/
scp "C:\path\to\shapes.zip" <user>@<node-host-or-ip>:/home/<user>/emergent/data/salmon_abm/
```

`data/salmon_abm/` is gitignored by design for large local inputs.

## Troubleshooting

- `ssh: connect to host ... port 22: Connection refused`:
  - Run the tunnel command from the laptop terminal, not from a shell already on the node.
  - Verify host/IP and SSH service availability.

- Browser opens but screen is black:
  - On node: `tools/novnc_stack.sh status`
  - On node: `tools/novnc_stack.sh logs`
  - If display/auth looks stale, restart stack:
    - `tools/novnc_stack.sh viewer`
  - In browser noVNC toolbar, refresh/reconnect.
  - Large H5 files can take noticeable time for first draw.

- RL viewer opens too small or maximize is too large:
  - This is usually a geometry mismatch: browser viewport (Windows) vs Xvfb desktop size (Linux).
  - In the RL viewer menu, use `View -> Fit Window to noVNC Desktop` (shortcut: `Ctrl+0`).
  - Avoid title-bar maximize in noVNC sessions; use Fit Window instead.
  - If mismatch persists, restart stack with a smaller virtual desktop (example: `NOVNC_RESOLUTION=1366x768`).

- Output file size looks stalled during run:
  - HDF5 allocation/chunking can make apparent size changes bursty.
  - Confirm completion via `<model>_stats.json` and viewer playback rather than shell size alone.
