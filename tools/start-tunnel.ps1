param(
    [string]$HostName = "192.168.102.157",
    [string]$UserName = "kevinnebiolo",
    [int]$LocalPort = 6080,
    [int]$RemotePort = 6080,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$LocalPort/vnc.html"
}

Write-Host "Starting SSH tunnel: localhost:$LocalPort -> $HostName`:$RemotePort"
Write-Host "Press Ctrl+C to stop."

ssh -N -L "${LocalPort}:127.0.0.1:${RemotePort}" "$UserName@$HostName"
