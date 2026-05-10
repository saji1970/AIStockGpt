# GCP Deployment Script for AI Stock GPT
param(
    [string]$ProjectId = "stockbroker-28983",
    [string]$ParallelApiKey = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI",
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

Write-Host "AI Stock GPT - GCP Deployment" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Check gcloud
try {
    $null = Get-Command gcloud -ErrorAction Stop
    Write-Host "Google Cloud SDK found" -ForegroundColor Green
} catch {
    Write-Host "Google Cloud SDK not found. Please install it first." -ForegroundColor Red
    exit 1
}

# Check authentication
$auth = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
if (-not $auth) {
    Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}
Write-Host "Authenticated as: $auth" -ForegroundColor Green

# Set project
Write-Host "Setting project: $ProjectId" -ForegroundColor Yellow
gcloud config set project $ProjectId

# Enable APIs
Write-Host "Enabling APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Deploy backend
Write-Host "Deploying backend..." -ForegroundColor Yellow

# Create cloudbuild config
$backendConfig = "steps:`n  - name: 'gcr.io/cloud-builders/docker'`n    args: ['build', '-f', 'Dockerfile.backend', '-t', 'gcr.io/$ProjectId/ai-stock-gpt-backend', '.']`nimages:`n  - 'gcr.io/$ProjectId/ai-stock-gpt-backend'"
$backendConfig | Out-File -FilePath "cloudbuild-backend.yaml" -Encoding ASCII

# Build and deploy
gcloud builds submit --config cloudbuild-backend.yaml .
gcloud run deploy ai-stock-gpt-backend --image "gcr.io/$ProjectId/ai-stock-gpt-backend" --platform managed --region $Region --allow-unauthenticated --port 8080 --memory 1Gi --cpu 1 --max-instances 10 --set-env-vars "PARALLEL_API_KEY=$ParallelApiKey"

# Get backend URL
$BackendUrl = gcloud run services describe ai-stock-gpt-backend --region=$Region --format="value(status.url)"
Write-Host "Backend: $BackendUrl" -ForegroundColor Green

# Build frontend
Write-Host "Building frontend..." -ForegroundColor Yellow
npm install
npm run build

# Deploy frontend
Write-Host "Deploying frontend..." -ForegroundColor Yellow
$BucketName = "ai-stock-gpt-frontend-static"

# Create bucket if it doesn't exist
try {
    gsutil ls gs://$BucketName 2>$null
    Write-Host "Bucket already exists" -ForegroundColor Gray
} catch {
    gsutil mb -p $ProjectId -c STANDARD -l $Region gs://$BucketName
    Write-Host "Created new bucket" -ForegroundColor Gray
}

# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://$BucketName

# Upload frontend files
gsutil -m cp -r build/* gs://$BucketName/

# Set website configuration
gsutil web set -m index.html -e 404.html gs://$BucketName

$FrontendUrl = "https://storage.googleapis.com/$BucketName/index.html"
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Green

# Cleanup
Remove-Item "cloudbuild-backend.yaml" -ErrorAction SilentlyContinue

# Save URLs
"Backend URL: $BackendUrl`nFrontend URL: $FrontendUrl`nAPI Docs: $BackendUrl/docs" | Out-File -FilePath "deployment-urls.txt" -Encoding ASCII

Write-Host ""
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Cyan
Write-Host "Backend: $BackendUrl" -ForegroundColor Cyan
Write-Host "API Docs: $BackendUrl/docs" -ForegroundColor Cyan
