# Deploy React frontend to Cloud Storage for static hosting
Write-Host "Building React frontend..." -ForegroundColor Green

# Build the React app
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Fix static asset paths for Cloud Storage
Write-Host "Fixing static asset paths..." -ForegroundColor Green
$indexHtmlPath = "build\index.html"
$baseUrl = "https://ai-stock-gpt-frontend-static.storage.googleapis.com"

if (Test-Path $indexHtmlPath) {
    $content = Get-Content $indexHtmlPath -Raw
    $content = $content -replace 'href="/', "href=`"$baseUrl/"
    $content = $content -replace 'src="/', "src=`"$baseUrl/"
    $content | Set-Content $indexHtmlPath -NoNewline
    Write-Host "✅ Paths fixed!" -ForegroundColor Green
}

Write-Host "Deploying to Cloud Storage..." -ForegroundColor Green

# Create a bucket for static hosting
$BUCKET_NAME = "ai-stock-gpt-frontend-static"

# Create bucket if it doesn't exist
gsutil mb -l us-central1 gs://$BUCKET_NAME 2>$null

# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME

# Upload all files from build directory
gsutil -m rsync -r -d build/ gs://$BUCKET_NAME/

# Set website configuration
gsutil web set -m index.html -e 404.html gs://$BUCKET_NAME

Write-Host "Frontend deployed to: https://storage.googleapis.com/$BUCKET_NAME/index.html" -ForegroundColor Green
Write-Host "Or use custom domain with Cloud CDN for better performance" -ForegroundColor Yellow
