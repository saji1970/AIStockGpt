param(
    [string]$ProjectId = "",
    [string]$Environment = "",
    [string]$Service = ""
)

$ErrorActionPreference = "Stop"

function Test-CommandAvailable {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandAvailable "railway")) {
    throw "Railway CLI is not installed. Install it first: npm install -g @railway/cli"
}

Write-Host "==> Deploying AIStockGpt to Railway" -ForegroundColor Cyan

# Ensure user is authenticated
try {
    $null = railway whoami 2>$null
} catch {
    Write-Host "==> Railway login required. Opening login flow..." -ForegroundColor Yellow
    railway login
}

# Link project/environment/service when provided
if (-not [string]::IsNullOrWhiteSpace($ProjectId)) {
    $linkArgs = @("link", "--project", $ProjectId)
    if (-not [string]::IsNullOrWhiteSpace($Environment)) {
        $linkArgs += @("--environment", $Environment)
    }
    if (-not [string]::IsNullOrWhiteSpace($Service)) {
        $linkArgs += @("--service", $Service)
    }

    Write-Host "==> Linking local directory to Railway project" -ForegroundColor Cyan
    & railway @linkArgs
} else {
    Write-Host "==> Using existing Railway link (run 'railway link' manually if needed)" -ForegroundColor Cyan
}

# Deploy current repository (uses railway.toml -> Dockerfile.backend)
Write-Host "==> Uploading and deploying current source" -ForegroundColor Cyan
railway up --detach

Write-Host "==> Deployment started successfully." -ForegroundColor Green
Write-Host "Run 'railway logs' to monitor startup logs." -ForegroundColor Green
Write-Host "Run 'railway domain' to view or create public domain." -ForegroundColor Green
