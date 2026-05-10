#!/bin/bash

# GCP Deployment Script for AI Stock GPT
# This script deploys the backend and frontend to Google Cloud Platform

set -e  # Exit on any error

# Configuration
PROJECT_ID="stockbroker-28983"  # Your GCP project ID
PARALLEL_API_KEY="VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"  # Your API key
REGION="us-central1"

echo "🚀 AI Stock GPT - GCP Deployment"
echo "================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ Google Cloud SDK found"

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  You are not authenticated with gcloud. Please run:"
    echo "gcloud auth login"
    exit 1
fi

AUTH_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
echo "✅ Authenticated as: $AUTH_ACCOUNT"

# Set project
echo "📋 Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable appengine.googleapis.com

# Build and deploy backend
echo "🏗️  Building and deploying backend..."

# Create cloudbuild.yaml for backend
cat > cloudbuild-backend.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.backend', '-t', 'gcr.io/$PROJECT_ID/ai-stock-gpt-backend', '.']
images:
  - 'gcr.io/$PROJECT_ID/ai-stock-gpt-backend'
EOF

# Build backend image
gcloud builds submit --config cloudbuild-backend.yaml .

# Deploy backend to Cloud Run
gcloud run deploy ai-stock-gpt-backend \
    --image "gcr.io/$PROJECT_ID/ai-stock-gpt-backend" \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars "PARALLEL_API_KEY=$PARALLEL_API_KEY"

# Get backend URL
BACKEND_URL=$(gcloud run services describe ai-stock-gpt-backend --region=$REGION --format="value(status.url)")
echo "✅ Backend deployed at: $BACKEND_URL"

# Build frontend
echo "🏗️  Building frontend..."
npm install
npm run build

# Deploy frontend to Cloud Storage (static hosting)
echo "📤 Deploying frontend to Cloud Storage..."

# Create bucket if it doesn't exist
BUCKET_NAME="ai-stock-gpt-frontend-static"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME 2>/dev/null || true

# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME

# Upload frontend files
gsutil -m cp -r build/* gs://$BUCKET_NAME/

# Set website configuration
cat > website-config.json << EOF
{
    "mainPageSuffix": "index.html",
    "notFoundPage": "index.html"
}
EOF

gsutil web set -m index.html -e 404.html gs://$BUCKET_NAME

# Get frontend URL
FRONTEND_URL="https://storage.googleapis.com/$BUCKET_NAME/index.html"
echo "✅ Frontend deployed at: $FRONTEND_URL"

# Update frontend API configuration
echo "🔧 Updating frontend API configuration..."
sed -i "s|http://localhost:8000|$BACKEND_URL|g" build/static/js/*.js

# Clean up temporary files
rm -f cloudbuild-backend.yaml website-config.json

# Display results
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "Frontend: $FRONTEND_URL"
echo "Backend: $BACKEND_URL"
echo "API Docs: $BACKEND_URL/docs"
echo ""
echo "📋 Test Commands:"
echo "curl $BACKEND_URL/status"
echo "curl $FRONTEND_URL"
echo ""
echo "🔑 Your API key is configured and ready to use!"

# Save URLs to file
cat > deployment-urls.txt << EOF
Backend URL: $BACKEND_URL
Frontend URL: $FRONTEND_URL
API Documentation: $BACKEND_URL/docs
EOF

echo "📄 URLs saved to deployment-urls.txt"
