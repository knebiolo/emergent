@echo off
REM Batch file to connect to Spark via noVNC and display instructions for launching the RL Viewer
REM Double-click this file to open noVNC tunnel in your browser

echo.
echo ========================================
echo   Connecting to Spark via noVNC Tunnel
echo   (192.168.102.157 via localhost:6080)
echo ========================================
echo Starting noVNC server on Spark...
ssh kevinnebiolo@192.168.102.157 "cd /home/kevinnebiolo/emergent && tools/novnc_stack.sh start"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start noVNC server on Spark
    echo Please check your SSH connection and try again.
    echo.
    pause
    exit /b 1
)

echo noVNC server started successfully!
echo.
echo Starting SSH tunnel and opening browser...

REM Start the tunnel in the background using PowerShell job
start /B powershell -ExecutionPolicy Bypass -Command "Start-Process 'http://127.0.0.1:6080/vnc.html'; ssh -N -L 6080:127.0.0.1:6080 kevinnebiolo@192.168.102.157"

echo.
echo Waiting for tunnel to establish...
timeout /t 3 /nobreak >nul

echo.
echo Launching RL Training Viewer on Spark (this may take 10-15 seconds)...
ssh kevinnebiolo@192.168.102.157 "cd /home/kevinnebiolo/emergent && nohup env DISPLAY=:1 bash -c 'conda activate emergent && python -m emergent.salmon_abm.rl_training_viewer --model-dir data/salmon_abm --start-polygon data/salmon_abm/start_loc_river_right.shp' > /tmp/rl_viewer.log 2>&1 &"

echo.
echo ========================================
echo   RL Viewer should now be starting!
echo ========================================
echo.
echo INSTRUCTIONS:
echo - Check the browser window for Spark's desktop
echo - The RL Viewer GUI should appear within 10-15 seconds
echo - If it doesn't appear, check the log with:
echo     ssh kevinnebiolo@192.168.102.157 "cat /tmp/rl_viewer.log"
echo.
echo - Press any key to stop everything and clean up
echo ========================================
pause >nul

REM Cleanup when tunnel closes (user pressed Ctrl+C or closed window)
echo.
echo Cleaning up...
echo Stopping RL Viewer and noVNC server on Spark...

REM Kill the RL viewer process
ssh kevinnebiolo@192.168.102.157 "pkill -f rl_training_viewer"

REM Stop the noVNC server
ssh kevinnebiolo@192.168.102.157 "cd /home/kevinnebiolo/emergent && tools/novnc_stack.sh stop"

echo Cleanup complete.
pause

REM Use the existing start-tunnel.ps1 script
powershell -ExecutionPolicy Bypass -File "%~dp0tools\start-tunnel.ps1"

REM Note: The tunnel connects localhost:6080 to Spark's noVNC server
REM Once connected in browser, open a terminal on Spark and run:
REM   cd /emergent
REM   conda activate emergent
REM   python -m emergent.salmon_abm.rl_training_viewer --model-dir data/salmon_abm --start-polygon data/salmon_abm/start_loc_river_right.shp
