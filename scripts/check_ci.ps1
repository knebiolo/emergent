param(
    [string]$Repo = ''
)

# Determine repo from git remote if not provided
if (-not $Repo) {
    try {
        $url = (git config --get remote.origin.url) -replace '\\n',''
    } catch {
        Write-Error "Cannot read git remote. Provide --Repo owner/repo"
        exit 1
    }
    if (-not $url) { Write-Error "No remote.origin.url found. Provide --Repo owner/repo"; exit 1 }
    if ($url -match 'github.com[:/](.+?)(?:\.git)?$') { $Repo = $matches[1] }
}

Write-Host "Using repo: $Repo"

try {
    gh --version > $null 2>&1
} catch {
    Write-Error "GitHub CLI (gh) not found. Install from https://cli.github.com/"
    exit 1
}

Write-Host "Listing recent CI runs for workflow 'ci-tests.yml'..."
gh run list --repo $Repo --workflow ci-tests.yml --limit 5

$id = gh run list --repo $Repo --workflow ci-tests.yml --limit 1 --json id --jq '.[0].id'
if ($id) {
    Write-Host "Opening latest run in browser (id: $id)"
    gh run view $id --repo $Repo --web
} else {
    Write-Warning "No runs found for workflow 'ci-tests.yml' in repo $Repo"
}
