@echo off
setlocal EnableExtensions

if /I "%~1"=="--inner" (
  shift /1
) else (
  echo %CMDCMDLINE% | find /I " /c " >nul
  if not errorlevel 1 (
    start "Spark RL Viewer Launcher" cmd /k ""%~f0" --inner"
    exit /b 0
  )
)

REM One-click launcher from laptop:
REM - Starts/restarts Spark noVNC stack
REM - Launches RL viewer on Spark
REM - Opens local browser tunnel to Spark noVNC

set "SPARK_HOST=192.168.102.157"
set "SPARK_USER=kevinnebiolo"
set "SPARK_REPO=/home/kevinnebiolo/emergent"
set "LOCAL_WEB_PORT=6080"
set "LOCAL_WEB_PORT_ALT1=6081"
set "LOCAL_WEB_PORT_ALT2=6082"
set "PRIMARY_DISPLAY=:1"
set "PRIMARY_RFB_PORT=5901"
set "PRIMARY_WEB_PORT=6080"
set "FALLBACK_DISPLAY=:99"
set "FALLBACK_RFB_PORT=5999"
set "FALLBACK_WEB_PORT=6081"
set "ACTIVE_REMOTE_WEB_PORT=%PRIMARY_WEB_PORT%"
set "ACTIVE_LOCAL_WEB_PORT=%LOCAL_WEB_PORT%"
set "EXIT_CODE=0"
set "LOGFILE=%TEMP%\spark_rl_viewer_launcher.log"

> "%LOGFILE%" echo [%DATE% %TIME%] Launcher start

echo.
echo ========================================
echo   Spark RL Viewer Launcher
echo ========================================
echo   Host: %SPARK_HOST%
echo   User: %SPARK_USER%
echo ========================================
echo.
echo Log file: %LOGFILE%
echo.

echo [1/2] Starting noVNC + RL viewer on Spark (display %PRIMARY_DISPLAY%, web port %PRIMARY_WEB_PORT%)...
>> "%LOGFILE%" echo [%DATE% %TIME%] Step 1 start primary
ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%PRIMARY_DISPLAY% NOVNC_RFB_PORT=%PRIMARY_RFB_PORT% NOVNC_WEB_PORT=%PRIMARY_WEB_PORT% tools/novnc_stack.sh rlviewer" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo.
    echo Primary display/port in use. Retrying with fallback stack...
    echo Display %FALLBACK_DISPLAY%, web port %FALLBACK_WEB_PORT%
    >> "%LOGFILE%" echo [%DATE% %TIME%] Primary failed, trying fallback
    ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%FALLBACK_DISPLAY% NOVNC_RFB_PORT=%FALLBACK_RFB_PORT% NOVNC_WEB_PORT=%FALLBACK_WEB_PORT% tools/novnc_stack.sh rlviewer" >> "%LOGFILE%" 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to launch noVNC/RL viewer on Spark (primary and fallback).
        echo Check SSH access and retry.
        echo.
        >> "%LOGFILE%" echo [%DATE% %TIME%] Step 1 failed
        set "EXIT_CODE=1"
        goto finish
    )
    set "ACTIVE_REMOTE_WEB_PORT=%FALLBACK_WEB_PORT%"
    >> "%LOGFILE%" echo [%DATE% %TIME%] Step 1 fallback succeeded
)
if not errorlevel 1 >> "%LOGFILE%" echo [%DATE% %TIME%] Step 1 completed

echo.
echo [2/2] Opening browser and starting local SSH tunnel...
>> "%LOGFILE%" echo [%DATE% %TIME%] Step 2 start
where ssh >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: ssh.exe not found on this machine PATH.
    echo Install OpenSSH client and retry.
    echo.
    >> "%LOGFILE%" echo [%DATE% %TIME%] ssh.exe missing
    set "EXIT_CODE=1"
    goto finish
)

call :pick_local_port
if "%ACTIVE_LOCAL_WEB_PORT%" NEQ "%LOCAL_WEB_PORT%" (
    echo Local port %LOCAL_WEB_PORT% is busy; using %ACTIVE_LOCAL_WEB_PORT% instead.
    >> "%LOGFILE%" echo [%DATE% %TIME%] Local port fallback to %ACTIVE_LOCAL_WEB_PORT%
)

start "" "http://127.0.0.1:%ACTIVE_LOCAL_WEB_PORT%/vnc.html"

echo.
echo ========================================
echo   Launch complete
echo ========================================
echo Browser URL: http://127.0.0.1:%ACTIVE_LOCAL_WEB_PORT%/vnc.html
echo Spark noVNC remote port: %ACTIVE_REMOTE_WEB_PORT%
echo.
echo Tunnel is running in THIS window.
echo Press Ctrl+C to stop tunnel when done.
echo.
echo Starting SSH tunnel localhost:%ACTIVE_LOCAL_WEB_PORT% ^> %SPARK_HOST%:%ACTIVE_REMOTE_WEB_PORT%
>> "%LOGFILE%" echo [%DATE% %TIME%] Starting tunnel local=%ACTIVE_LOCAL_WEB_PORT% remote=%ACTIVE_REMOTE_WEB_PORT%
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -L %ACTIVE_LOCAL_WEB_PORT%:127.0.0.1:%ACTIVE_REMOTE_WEB_PORT% %SPARK_USER%@%SPARK_HOST%
set "EXIT_CODE=%ERRORLEVEL%"
>> "%LOGFILE%" echo [%DATE% %TIME%] Tunnel exit code %EXIT_CODE%

echo.
if "%EXIT_CODE%"=="0" (
    echo Tunnel exited cleanly.
) else (
    echo Tunnel exited with code %EXIT_CODE%.
    echo Check log: %LOGFILE%
)
goto finish

:pick_local_port
set "ACTIVE_LOCAL_WEB_PORT=%LOCAL_WEB_PORT%"
call :is_port_busy %LOCAL_WEB_PORT%
if "%PORT_BUSY%"=="0" exit /b 0

set "ACTIVE_LOCAL_WEB_PORT=%LOCAL_WEB_PORT_ALT1%"
call :is_port_busy %LOCAL_WEB_PORT_ALT1%
if "%PORT_BUSY%"=="0" exit /b 0

set "ACTIVE_LOCAL_WEB_PORT=%LOCAL_WEB_PORT_ALT2%"
call :is_port_busy %LOCAL_WEB_PORT_ALT2%
if "%PORT_BUSY%"=="0" exit /b 0

echo.
echo ERROR: Local ports %LOCAL_WEB_PORT%, %LOCAL_WEB_PORT_ALT1%, and %LOCAL_WEB_PORT_ALT2% are all in use.
echo Free one of these ports and retry.
set "EXIT_CODE=1"
goto finish

:is_port_busy
set "PORT_BUSY=0"
netstat -ano | findstr /R /C:":%~1 .*LISTENING" >nul
if not errorlevel 1 set "PORT_BUSY=1"
exit /b 0

:finish
echo.
if "%EXIT_CODE%"=="0" (
    echo Launcher complete. Log file: %LOGFILE%
    >> "%LOGFILE%" echo [%DATE% %TIME%] Launcher complete
) else (
    echo Launcher finished with errors. Log file: %LOGFILE%
    >> "%LOGFILE%" echo [%DATE% %TIME%] Launcher failed
)
echo Press any key to close.
pause >nul
exit /b %EXIT_CODE%
