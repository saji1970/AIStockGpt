# Force refresh frontend by setting cache control headers
Write-Host "Setting cache control headers to force refresh..." -ForegroundColor Green

# Set no-cache headers for all files
gsutil -m setmeta -h "Cache-Control:no-cache, no-store, must-revalidate" gs://ai-stock-gpt-frontend-static/*.html
gsutil -m setmeta -h "Cache-Control:no-cache, no-store, must-revalidate" gs://ai-stock-gpt-frontend-static/static/css/*.css
gsutil -m setmeta -h "Cache-Control:no-cache, no-store, must-revalidate" gs://ai-stock-gpt-frontend-static/static/js/*.js

Write-Host "✅ Cache headers set!" -ForegroundColor Green
Write-Host "Please clear your browser cache (Ctrl+Shift+R) and try again." -ForegroundColor Yellow
Write-Host "Frontend URL: https://ai-stock-gpt-frontend-static.storage.googleapis.com/" -ForegroundColor Cyan
