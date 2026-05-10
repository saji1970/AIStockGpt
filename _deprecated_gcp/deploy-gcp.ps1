# GCP Deployment Script for AI Stock GPT (PowerShell)
# This script deploys the backend and frontend to Google Cloud Platform

param(
    [string]$ProjectId = "stockbroker-28983",  # Your GCP project ID
    [string]$ParallelApiKey = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI",  # Your API key
    [string]$Region = "us-central1"
)

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🚀 AI Stock GPT - GCP Deployment (PowerShell)" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if gcloud is installed
try {
    $null = Get-Command gcloud -ErrorAction Stop
    Write-Host "✅ Google Cloud SDK found" -ForegroundColor Green
} catch {
    Write-Host "❌ Google Cloud SDK is not installed. Please install it first." -ForegroundColor Red
    Write-Host "Visit: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if user is authenticated
try {
    $auth = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if (-not $auth) {
        Write-Host "⚠️  You are not authenticated with gcloud. Please run:" -ForegroundColor Yellow
        Write-Host "gcloud auth login" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✅ Authenticated as: $auth" -ForegroundColor Green
} catch {
    Write-Host "❌ Authentication check failed" -ForegroundColor Red
    exit 1
}

# Set project
Write-Host "📋 Setting GCP project to: $ProjectId" -ForegroundColor Yellow
gcloud config set project $ProjectId

# Enable required APIs
Write-Host "🔧 Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable appengine.googleapis.com

# Build and deploy backend
Write-Host "🏗️  Building and deploying backend..." -ForegroundColor Yellow

# Create a temporary cloudbuild.yaml for backend
$backendBuildConfig = @"
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.backend', '-t', 'gcr.io/$ProjectId/ai-stock-gpt-backend', '.']
images:
  - 'gcr.io/$ProjectId/ai-stock-gpt-backend'
"@

$backendBuildConfig | Out-File -FilePath "cloudbuild-backend.yaml" -Encoding UTF8

# Build backend image
gcloud builds submit --config cloudbuild-backend.yaml .

# Deploy backend to Cloud Run
gcloud run deploy ai-stock-gpt-backend `
    --image "gcr.io/$ProjectId/ai-stock-gpt-backend" `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --port 8080 `
    --memory 1Gi `
    --cpu 1 `
    --max-instances 10 `
    --set-env-vars "PARALLEL_API_KEY=$ParallelApiKey"

# Get backend URL
$BackendUrl = gcloud run services describe ai-stock-gpt-backend --region=$Region --format="value(status.url)"
Write-Host "✅ Backend deployed at: $BackendUrl" -ForegroundColor Green

# Build frontend
Write-Host "🏗️  Building frontend..." -ForegroundColor Yellow
npm install
npm run build

# Deploy frontend to Cloud Storage (static hosting)
Write-Host "📤 Deploying frontend to Cloud Storage..." -ForegroundColor Yellow

# Create bucket if it doesn't exist
$BucketName = "ai-stock-gpt-frontend-static"
gsutil mb -p $ProjectId -c STANDARD -l $Region gs://$BucketName 2>$null

# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://$BucketName

# Upload frontend files
gsutil -m cp -r build/* gs://$BucketName/

# Set website configuration
$websiteConfig = @"
{
    "mainPageSuffix": "index.html",
    "notFoundPage": "index.html"
}
"@

$websiteConfig | Out-File -FilePath "website-config.json" -Encoding UTF8
gsutil web set -m index.html -e 404.html gs://$BucketName

# Get frontend URL
$FrontendUrl = "https://storage.googleapis.com/$BucketName/index.html"
Write-Host "✅ Frontend deployed at: $FrontendUrl" -ForegroundColor Green

# Update frontend API configuration
Write-Host "🔧 Updating frontend API configuration..." -ForegroundColor Yellow
Get-ChildItem -Path "build/static/js" -Filter "*.js" | ForEach-Object {
    (Get-Content $_.FullName) -replace "http://localhost:8000", $BackendUrl | Set-Content $_.FullName
}

# Clean up temporary files
Remove-Item "cloudbuild-backend.yaml" -ErrorAction SilentlyContinue
Remove-Item "website-config.json" -ErrorAction SilentlyContinue

# Display results
Write-Host ""
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Cyan
Write-Host "Backend: $BackendUrl" -ForegroundColor Cyan
Write-Host "API Docs: $BackendUrl/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Test Commands:" -ForegroundColor Yellow
Write-Host "Invoke-WebRequest -Uri '$BackendUrl/status'" -ForegroundColor Gray
Write-Host "Invoke-WebRequest -Uri '$FrontendUrl'" -ForegroundColor Gray
Write-Host ""
Write-Host "🔑 Your API key is configured and ready to use!" -ForegroundColor Green

# Save URLs to file
$urls = @"
Backend URL: $BackendUrl
Frontend URL: $FrontendUrl
API Documentation: $BackendUrl/docs
"@

$urls | Out-File -FilePath "deployment-urls.txt" -Encoding UTF8
Write-Host "📄 URLs saved to deployment-urls.txt" -ForegroundColor Green
