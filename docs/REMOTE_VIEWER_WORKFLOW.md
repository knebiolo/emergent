# Remote Viewer Workflow (PowerShell + noVNC)

Use this workflow when running on a remote compute node with no physical display and a managed work laptop.

## 1) Start tunnel from laptop (PowerShell)

If using the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start-tunnel.ps1
```

Manual equivalent:

```powershell
Start-Process "http://127.0.0.1:6080/vnc.html"; ssh -N -L 6080`:127.0.0.1`:6080 kevinnebiolo@192.168.102.157
```

Keep this PowerShell window open.

## 2) Start virtual desktop + viewer on node

```bash
cd /home/kevinnebiolo/emergent
tools/novnc_stack.sh start
DISPLAY=:1 conda run -n emergent python -m emergent.salmon_abm.realtime_viewer
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
  --hecras-time-mode loop \
  --backend process \
  --write-mode minimal
```

## 4) Transfer large gitignored HDF inputs to node

```powershell
scp "C:\path\to\plan.hdf" kevinnebiolo@192.168.102.157:/home/kevinnebiolo/emergent/data/salmon_abm/
```

`data/salmon_abm/` is gitignored by design for large local inputs.

## Troubleshooting

- If browser page does not load, verify tunnel:
  - `Test-NetConnection 127.0.0.1 -Port 6080`
- If viewer is blank, check node services:
  - `tools/novnc_stack.sh status`
  - `tools/novnc_stack.sh logs`
