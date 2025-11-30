<#
Runs the Numba warmup script in the project's conda environment.
Usage (PowerShell):
  .\scripts\warmup.ps1

Optional parameters:
  -EnvPath: path to the conda environment (default: user's .conda envs\emergent)
  -ProjectRoot: project root path (default: current working directory)
#>
param(
    [string]$EnvPath = "$env:USERPROFILE\\.conda\\envs\\emergent",
    [string]$ProjectRoot = "$PWD"
)

try {
    $envPath = (Resolve-Path $EnvPath).Path
} catch {
    Write-Error "Conda env path not found: $EnvPath"
    exit 2
}

try {
    $projectRoot = (Resolve-Path $ProjectRoot).Path
} catch {
    Write-Error "Project root not found: $ProjectRoot"
    exit 3
}

Write-Output "Running numba warmup using conda env: $envPath"

# Run the warmup using conda run so we don't require an activated shell
& conda run -p $envPath --no-capture-output python "$projectRoot\tools\numba_warmup.py"

Write-Output "Numba warmup finished."