# Fix static asset paths for Cloud Storage deployment
Write-Host "Fixing static asset paths for Cloud Storage..." -ForegroundColor Green

$indexHtmlPath = "build\index.html"
$baseUrl = "https://ai-stock-gpt-frontend-static.storage.googleapis.com"

if (Test-Path $indexHtmlPath) {
    # Read the current index.html
    $content = Get-Content $indexHtmlPath -Raw
    
    # Replace relative paths with absolute URLs
    $content = $content -replace 'href="/', "href=`"$baseUrl/"
    $content = $content -replace 'src="/', "src=`"$baseUrl/"
    
    # Write the updated content back
    $content | Set-Content $indexHtmlPath -NoNewline
    
    Write-Host "✅ Paths fixed successfully!" -ForegroundColor Green
    Write-Host "Static assets now point to: $baseUrl" -ForegroundColor Yellow
} else {
    Write-Host "❌ Error: build\index.html not found!" -ForegroundColor Red
    Write-Host "Please run 'npm run build' first." -ForegroundColor Yellow
}
