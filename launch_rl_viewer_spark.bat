@echo off
setlocal

REM One-click launcher from laptop:
REM - Starts/restarts Spark noVNC stack
REM - Launches RL viewer on Spark
REM - Opens local browser tunnel to Spark noVNC

set "SPARK_HOST=192.168.102.157"
set "SPARK_USER=kevinnebiolo"
set "SPARK_REPO=/home/kevinnebiolo/emergent"
set "LOCAL_WEB_PORT=6080"

echo.
echo ========================================
echo   Spark RL Viewer Launcher
echo ========================================
echo   Host: %SPARK_HOST%
echo   User: %SPARK_USER%
echo ========================================
echo.

echo [1/2] Starting noVNC + RL viewer on Spark...
ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && tools/novnc_stack.sh rlviewer"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch noVNC/RL viewer on Spark.
    echo Check SSH access and retry.
    echo.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting local SSH tunnel and opening browser...
start "Spark noVNC Tunnel" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0tools\start-tunnel.ps1" -HostName "%SPARK_HOST%" -UserName "%SPARK_USER%" -LocalPort %LOCAL_WEB_PORT% -RemotePort 6080

echo.
echo ========================================
echo   Launch complete
echo ========================================
echo Browser URL: http://127.0.0.1:%LOCAL_WEB_PORT%/vnc.html
echo.
echo Keep the "Spark noVNC Tunnel" PowerShell window open while you use the viewer.
echo.
pause
exit /b 0
