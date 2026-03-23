@echo off
setlocal EnableExtensions

if /I "%~1"=="--inner" goto main
start "Spark RL Viewer Launcher" cmd /k ""%~f0" --inner"
exit /b 0

:main
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
set "NOVNC_QUERY=?autoconnect=1&resize=scale"
set "EXIT_CODE=0"
set "LOGFILE=%TEMP%\spark_rl_viewer_launcher.log"

call :log Launcher start

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
call :log Step 1 start primary
ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%PRIMARY_DISPLAY% NOVNC_RFB_PORT=%PRIMARY_RFB_PORT% NOVNC_WEB_PORT=%PRIMARY_WEB_PORT% tools/novnc_stack.sh rlviewer" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto step1_fallback
call :log Step 1 primary succeeded
goto step2

:step1_fallback
echo.
echo Primary display/port in use. Retrying with fallback stack...
echo Display %FALLBACK_DISPLAY%, web port %FALLBACK_WEB_PORT%
call :log Primary failed, trying fallback
ssh %SPARK_USER%@%SPARK_HOST% "cd %SPARK_REPO% && NOVNC_DISPLAY=%FALLBACK_DISPLAY% NOVNC_RFB_PORT=%FALLBACK_RFB_PORT% NOVNC_WEB_PORT=%FALLBACK_WEB_PORT% tools/novnc_stack.sh rlviewer" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto startup_failed
set "ACTIVE_REMOTE_WEB_PORT=%FALLBACK_WEB_PORT%"
call :log Step 1 fallback succeeded

:step2
echo.
echo [2/2] Opening browser and starting local SSH tunnel...
call :log Step 2 start
where ssh >nul 2>&1
if errorlevel 1 goto missing_ssh

call :pick_local_port
if errorlevel 1 goto finish_error

if /I not "%ACTIVE_LOCAL_WEB_PORT%"=="%LOCAL_WEB_PORT%" echo Local port %LOCAL_WEB_PORT% is busy; using %ACTIVE_LOCAL_WEB_PORT% instead.
if /I not "%ACTIVE_LOCAL_WEB_PORT%"=="%LOCAL_WEB_PORT%" call :log Local port fallback to %ACTIVE_LOCAL_WEB_PORT%

set "NOVNC_URL=http://127.0.0.1:%ACTIVE_LOCAL_WEB_PORT%/vnc.html%NOVNC_QUERY%"
start "" "%NOVNC_URL%"

echo.
echo ========================================
echo   Launch complete
echo ========================================
echo Browser URL: http://127.0.0.1:%ACTIVE_LOCAL_WEB_PORT%/vnc.html
echo View mode: autoconnect + scale-to-fit
echo Spark noVNC remote port: %ACTIVE_REMOTE_WEB_PORT%
echo.
echo Tunnel is running in THIS window.
echo Press Ctrl+C to stop tunnel when done.
echo.
echo Starting SSH tunnel localhost:%ACTIVE_LOCAL_WEB_PORT% ^> %SPARK_HOST%:%ACTIVE_REMOTE_WEB_PORT%
call :log Starting tunnel local=%ACTIVE_LOCAL_WEB_PORT% remote=%ACTIVE_REMOTE_WEB_PORT%
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -L %ACTIVE_LOCAL_WEB_PORT%:127.0.0.1:%ACTIVE_REMOTE_WEB_PORT% %SPARK_USER%@%SPARK_HOST%
set "EXIT_CODE=%ERRORLEVEL%"
call :log Tunnel exit code %EXIT_CODE%

echo.
if "%EXIT_CODE%"=="0" echo Tunnel exited cleanly.
if not "%EXIT_CODE%"=="0" echo Tunnel exited with code %EXIT_CODE%.
if not "%EXIT_CODE%"=="0" echo Check log: %LOGFILE%
goto finish

:startup_failed
echo.
echo ERROR: Failed to launch noVNC/RL viewer on Spark (primary and fallback).
echo Check SSH access and retry.
echo.
call :log Step 1 failed
goto finish_error

:missing_ssh
echo.
echo ERROR: ssh.exe not found on this machine PATH.
echo Install OpenSSH client and retry.
echo.
call :log ssh.exe missing
goto finish_error

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
call :log Local ports all busy
exit /b 1

:is_port_busy
set "PORT_BUSY=0"
netstat -ano | findstr /R /C:":%~1 .*LISTENING" >nul
if not errorlevel 1 set "PORT_BUSY=1"
exit /b 0

:log
>> "%LOGFILE%" echo %*
exit /b 0

:finish_error
set "EXIT_CODE=1"
goto finish

:finish
echo.
if "%EXIT_CODE%"=="0" echo Launcher complete. Log file: %LOGFILE%
if not "%EXIT_CODE%"=="0" echo Launcher finished with errors. Log file: %LOGFILE%
if "%EXIT_CODE%"=="0" call :log Launcher complete
if not "%EXIT_CODE%"=="0" call :log Launcher failed
echo Press any key to close.
pause >nul
exit /b %EXIT_CODE%
