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
set "PRIMARY_DISPLAY=:1"
set "PRIMARY_RFB_PORT=5901"
set "PRIMARY_WEB_PORT=6080"
set "FALLBACK_DISPLAY=:99"
set "FALLBACK_RFB_PORT=5999"
set "FALLBACK_WEB_PORT=6081"
set "ACTIVE_REMOTE_WEB_PORT=%PRIMARY_WEB_PORT%"

echo.
echo ========================================
echo   Spark RL Viewer Launcher
echo ========================================
echo   Host: %SPARK_HOST%
echo   User: %SPARK_USER%
echo ========================================
echo.

echo [1/2] Starting noVNC + RL viewer on Spark (display %PRIMARY_DISPLAY%, web port %PRIMARY_WEB_PORT%)...
ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%PRIMARY_DISPLAY% NOVNC_RFB_PORT=%PRIMARY_RFB_PORT% NOVNC_WEB_PORT=%PRIMARY_WEB_PORT% tools/novnc_stack.sh rlviewer"
if errorlevel 1 (
    echo.
    echo Primary display/port in use. Retrying with fallback stack...
    echo Display %FALLBACK_DISPLAY%, web port %FALLBACK_WEB_PORT%
    ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%FALLBACK_DISPLAY% NOVNC_RFB_PORT=%FALLBACK_RFB_PORT% NOVNC_WEB_PORT=%FALLBACK_WEB_PORT% tools/novnc_stack.sh rlviewer"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to launch noVNC/RL viewer on Spark (primary and fallback).
        echo Check SSH access and retry.
        echo.
        pause
        exit /b 1
    )
    set "ACTIVE_REMOTE_WEB_PORT=%FALLBACK_WEB_PORT%"
)

echo.
echo [2/2] Starting local SSH tunnel and opening browser...
start "" "http://127.0.0.1:%LOCAL_WEB_PORT%/vnc.html"
start "Spark noVNC Tunnel" cmd /k "echo Starting SSH tunnel localhost:%LOCAL_WEB_PORT% ^> %SPARK_HOST%:%ACTIVE_REMOTE_WEB_PORT% && ssh -N -L %LOCAL_WEB_PORT%:127.0.0.1:%ACTIVE_REMOTE_WEB_PORT% %SPARK_USER%@%SPARK_HOST%"

echo.
echo ========================================
echo   Launch complete
echo ========================================
echo Browser URL: http://127.0.0.1:%LOCAL_WEB_PORT%/vnc.html
echo Spark noVNC remote port: %ACTIVE_REMOTE_WEB_PORT%
echo.
echo Keep the "Spark noVNC Tunnel" command window open while you use the viewer.
echo.
pause
exit /b 0
