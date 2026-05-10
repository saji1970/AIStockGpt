# AI Stock GPT - GCP Deployment Guide

This guide will help you deploy the AI Stock GPT application to Google Cloud Platform using Cloud Run.

## 🚀 Prerequisites

### 1. Google Cloud Account
- Create a Google Cloud account at [cloud.google.com](https://cloud.google.com)
- Set up billing for your project

### 2. Install Google Cloud SDK
```bash
# Download and install from: https://cloud.google.com/sdk/docs/install
# Or use package manager:

# macOS
brew install google-cloud-sdk

# Ubuntu/Debian
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-sdk

# Windows
# Download from: https://cloud.google.com/sdk/docs/install#windows
```

### 3. Install Docker (for local testing)
```bash
# Download from: https://www.docker.com/products/docker-desktop
```

## 🔧 Initial Setup

### 1. Authenticate with Google Cloud
```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Create a New Project (or use existing)
```bash
# Create new project
gcloud projects create ai-stock-gpt-[YOUR-UNIQUE-ID]

# Or use existing project
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

### 3. Enable Required APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 📝 Configuration

### 1. Update Project ID
Edit `deploy-gcp.sh` and replace `your-gcp-project-id` with your actual project ID:
```bash
PROJECT_ID="your-actual-project-id"
```

### 2. Set Environment Variables
```bash
# Set your Parallel AI API key
export PARALLEL_API_KEY="VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"

# Set your GCP project ID
export GCP_PROJECT_ID="your-actual-project-id"
```

## 🐳 Local Testing (Optional)

Before deploying to GCP, you can test locally with Docker:

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run individually
docker build -f Dockerfile.backend -t ai-stock-gpt-backend .
docker build -f Dockerfile.frontend -t ai-stock-gpt-frontend .

docker run -p 8080:8080 -e PARALLEL_API_KEY="$PARALLEL_API_KEY" ai-stock-gpt-backend
docker run -p 80:80 ai-stock-gpt-frontend
```

## 🚀 Deployment

### Option 1: Automated Deployment
```bash
# Make the script executable
chmod +x deploy-gcp.sh

# Run the deployment script
./deploy-gcp.sh
```

### Option 2: Manual Deployment

#### Deploy Backend
```bash
# Build and push backend image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/ai-stock-gpt-backend --file Dockerfile.backend .

# Deploy to Cloud Run
gcloud run deploy ai-stock-gpt-backend \
    --image gcr.io/$GCP_PROJECT_ID/ai-stock-gpt-backend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars PARALLEL_API_KEY="$PARALLEL_API_KEY"
```

#### Deploy Frontend
```bash
# Get backend URL
BACKEND_URL=$(gcloud run services describe ai-stock-gpt-backend --region=us-central1 --format="value(status.url)")

# Build and push frontend image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/ai-stock-gpt-frontend --file Dockerfile.frontend .

# Deploy to Cloud Run
gcloud run deploy ai-stock-gpt-frontend \
    --image gcr.io/$GCP_PROJECT_ID/ai-stock-gpt-frontend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 80 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 5 \
    --set-env-vars REACT_APP_API_URL="$BACKEND_URL"
```

## 🌐 Custom Domain Setup (Optional)

### 1. Map Custom Domain
```bash
gcloud run domain-mappings create \
    --service ai-stock-gpt-frontend \
    --domain your-domain.com \
    --region us-central1
```

### 2. Update DNS Records
Add a CNAME record pointing to the Cloud Run service URL.

## 📊 Monitoring and Management

### View Logs
```bash
# Backend logs
gcloud logging tail --project=$GCP_PROJECT_ID --filter="resource.type=cloud_run_revision AND resource.labels.service_name=ai-stock-gpt-backend"

# Frontend logs
gcloud logging tail --project=$GCP_PROJECT_ID --filter="resource.type=cloud_run_revision AND resource.labels.service_name=ai-stock-gpt-frontend"
```

### Scale Services
```bash
# Scale backend
gcloud run services update ai-stock-gpt-backend \
    --region us-central1 \
    --max-instances 20

# Scale frontend
gcloud run services update ai-stock-gpt-frontend \
    --region us-central1 \
    --max-instances 10
```

### Update Environment Variables
```bash
# Update backend environment
gcloud run services update ai-stock-gpt-backend \
    --region us-central1 \
    --update-env-vars PARALLEL_API_KEY="new-api-key"
```

## 💰 Cost Optimization

### 1. Set Resource Limits
```bash
# Reduce memory and CPU for cost savings
gcloud run services update ai-stock-gpt-backend \
    --region us-central1 \
    --memory 512Mi \
    --cpu 0.5 \
    --max-instances 5
```

### 2. Enable Autoscaling
```bash
# Set minimum instances to 0 for cost savings
gcloud run services update ai-stock-gpt-backend \
    --region us-central1 \
    --min-instances 0 \
    --max-instances 10
```

## 🔒 Security

### 1. Enable Authentication (Optional)
```bash
# Remove --allow-unauthenticated to require authentication
gcloud run services update ai-stock-gpt-backend \
    --region us-central1 \
    --no-allow-unauthenticated
```

### 2. Set Up IAM
```bash
# Grant specific permissions
gcloud run services add-iam-policy-binding ai-stock-gpt-backend \
    --region us-central1 \
    --member="user:your-email@domain.com" \
    --role="roles/run.invoker"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Check build logs
gcloud builds log [BUILD_ID]

# Verify Dockerfile syntax
docker build -f Dockerfile.backend . --no-cache
```

#### 2. Service Not Starting
```bash
# Check service logs
gcloud run services logs read ai-stock-gpt-backend --region us-central1

# Verify environment variables
gcloud run services describe ai-stock-gpt-backend --region us-central1
```

#### 3. API Connection Issues
```bash
# Test backend health
curl https://your-backend-url/status

# Check CORS settings
curl -H "Origin: https://your-frontend-url" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS https://your-backend-url/chat
```

## 📈 Performance Optimization

### 1. Enable CDN
```bash
# Enable Cloud CDN for frontend
gcloud compute backend-services create ai-stock-gpt-frontend-cdn \
    --global \
    --load-balancing-scheme=EXTERNAL
```

### 2. Optimize Images
```bash
# Use multi-stage builds for smaller images
# Already implemented in Dockerfiles
```

## 🔄 Continuous Deployment

### Set Up Cloud Build Triggers
```bash
# Create trigger for GitHub repository
gcloud builds triggers create github \
    --repo-name=ai-stock-gpt \
    --repo-owner=your-username \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

## 📞 Support

For issues with:
- **GCP Services**: Check [Google Cloud Documentation](https://cloud.google.com/docs)
- **Application**: Check the logs and troubleshooting section above
- **Billing**: Visit [Google Cloud Billing](https://console.cloud.google.com/billing)

---

**Happy Deploying! 🚀**
