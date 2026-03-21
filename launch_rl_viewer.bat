@echo off
REM Batch file to launch the Salmon ABM RL Training Viewer
REM Double-click this file to start the viewer

echo.
echo ========================================
echo   Salmon ABM RL Training Viewer
echo ========================================
echo.

REM Change to the project directory (where this batch file is located)
cd /d "%~dp0"

echo Activating conda environment 'emergent'...
call conda activate emergent

if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate conda environment 'emergent'
    echo Please ensure conda is installed and the environment exists.
    echo.
    pause
    exit /b 1
)

echo.
echo Launching RL Training Viewer...
echo.

REM Launch the RL training viewer with default parameters
python -m emergent.salmon_abm.rl_training_viewer --model-dir data/salmon_abm --start-polygon data/salmon_abm/start_loc_river_right.shp

REM If the viewer exits with an error, pause so the user can see the error message
if errorlevel 1 (
    echo.
    echo ERROR: RL Training Viewer exited with an error.
    echo.
    pause
)
