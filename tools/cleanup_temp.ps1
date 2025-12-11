# PowerShell wrapper to run the Python cleanup script
param(
    [string]$Path = 'tmp/pytest_tmp',
    [int]$MaxAgeHours = 48
)
python .\tools\cleanup_temp.py --path $Path --max-age-hours $MaxAgeHours
