# Test frontend assets accessibility
Write-Host "Testing frontend assets..." -ForegroundColor Green

$baseUrl = "https://ai-stock-gpt-frontend-static.storage.googleapis.com"
$assets = @(
    "/index.html",
    "/static/css/main.074497d6.css",
    "/static/js/main.1e34ab80.js",
    "/asset-manifest.json"
)

foreach ($asset in $assets) {
    $url = $baseUrl + $asset
    try {
        $response = Invoke-WebRequest -Uri $url -Method Head
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ $asset - OK" -ForegroundColor Green
        } else {
            Write-Host "❌ $asset - Status: $($response.StatusCode)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ $asset - Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nFrontend URL: $baseUrl/" -ForegroundColor Cyan
Write-Host "If all tests pass, try accessing the URL in your browser with Ctrl+Shift+R to clear cache." -ForegroundColor Yellow
