#!/bin/bash

# Quick GCP Deployment Script for AI Stock GPT
# Simplified version for fast deployment

set -e

# Configuration - UPDATE THESE VALUES
PROJECT_ID="your-gcp-project-id"  # Replace with your actual project ID
PARALLEL_API_KEY="VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"  # Your API key
REGION="us-central1"

echo "🚀 Quick GCP Deployment for AI Stock GPT"
echo "========================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  You are not authenticated with gcloud. Please run:"
    echo "gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy backend
echo "🏗️  Building and deploying backend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/ai-stock-gpt-backend --file Dockerfile.backend .

gcloud run deploy ai-stock-gpt-backend \
    --image gcr.io/$PROJECT_ID/ai-stock-gpt-backend \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars PARALLEL_API_KEY="$PARALLEL_API_KEY"

# Get backend URL
BACKEND_URL=$(gcloud run services describe ai-stock-gpt-backend --region=$REGION --format="value(status.url)")
echo "✅ Backend deployed at: $BACKEND_URL"

# Build and deploy frontend
echo "🏗️  Building and deploying frontend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/ai-stock-gpt-frontend --file Dockerfile.frontend .

gcloud run deploy ai-stock-gpt-frontend \
    --image gcr.io/$PROJECT_ID/ai-stock-gpt-frontend \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 80 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 5 \
    --set-env-vars REACT_APP_API_URL="$BACKEND_URL"

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe ai-stock-gpt-frontend --region=$REGION --format="value(status.url)")
echo "✅ Frontend deployed at: $FRONTEND_URL"

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
echo "curl $FRONTEND_URL/health"
echo ""
echo "🔑 Your API key is configured and ready to use!"
