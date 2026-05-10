# AI Stock GPT - GCP Deployment Guide

This guide will walk you through deploying the AI Stock GPT application to Google Cloud Platform (GCP).

## 🚀 Quick Deployment

### Prerequisites

1. **Google Cloud SDK** installed and configured
2. **Node.js** (for frontend build)
3. **Docker** (optional, for local testing)
4. **GCP Project** with billing enabled

### One-Click Deployment

#### For Windows (PowerShell):
```powershell
.\deploy-gcp.ps1
```

#### For Linux/Mac (Bash):
```bash
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

## 📋 Manual Deployment Steps

### 1. Setup GCP Project

```bash
# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable appengine.googleapis.com
```

### 2. Deploy Backend

The backend is deployed to **Cloud Run** for scalability and cost-effectiveness.

```bash
# Build and deploy backend
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-stock-gpt-backend --file Dockerfile.backend .

gcloud run deploy ai-stock-gpt-backend \
    --image gcr.io/YOUR_PROJECT_ID/ai-stock-gpt-backend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars "PARALLEL_API_KEY=YOUR_API_KEY"
```

### 3. Deploy Frontend

The frontend is deployed to **Cloud Storage** for static hosting.

```bash
# Build frontend
npm install
npm run build

# Create storage bucket
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l us-central1 gs://ai-stock-gpt-frontend-static

# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://ai-stock-gpt-frontend-static

# Upload files
gsutil -m cp -r build/* gs://ai-stock-gpt-frontend-static/

# Configure website
gsutil web set -m index.html -e 404.html gs://ai-stock-gpt-frontend-static
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PARALLEL_API_KEY` | API key for external services | Required |
| `PORT` | Backend port | 8080 |

### Project Structure

```
AIStockGpt/
├── backend/                 # FastAPI backend
│   └── main.py             # Main backend application
├── src/                    # React frontend
├── build/                  # Built frontend files
├── Dockerfile.backend      # Backend container
├── Dockerfile.frontend     # Frontend container
├── app.yaml               # App Engine config
├── requirements.txt       # Python dependencies
├── package.json          # Node.js dependencies
├── deploy-gcp.sh         # Bash deployment script
└── deploy-gcp.ps1        # PowerShell deployment script
```

## 🌐 Service Architecture

### Backend (Cloud Run)
- **Service**: `ai-stock-gpt-backend`
- **Runtime**: Python 3.11
- **Framework**: FastAPI
- **Memory**: 1GB
- **CPU**: 1 vCPU
- **Scaling**: 1-10 instances

### Frontend (Cloud Storage)
- **Bucket**: `ai-stock-gpt-frontend-static`
- **Type**: Static website hosting
- **Access**: Public read
- **Domain**: `https://storage.googleapis.com/ai-stock-gpt-frontend-static/`

## 📊 Monitoring & Logs

### View Logs
```bash
# Backend logs
gcloud logging tail --project=YOUR_PROJECT_ID --filter="resource.type=cloud_run_revision"

# Build logs
gcloud builds log BUILD_ID
```

### Monitor Performance
```bash
# Check service status
gcloud run services describe ai-stock-gpt-backend --region=us-central1

# View metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"
```

## 🔒 Security

### IAM Permissions
- **Cloud Run**: Public access for API endpoints
- **Cloud Storage**: Public read for frontend files
- **Cloud Build**: Service account with build permissions

### API Security
- CORS configured for frontend domain
- Environment variables for sensitive data
- Input validation with Pydantic models

## 💰 Cost Optimization

### Backend (Cloud Run)
- **Pricing**: Pay per request + compute time
- **Optimization**: 
  - Set min instances to 0 for dev
  - Use max instances to limit costs
  - Configure CPU and memory appropriately

### Frontend (Cloud Storage)
- **Pricing**: Storage + bandwidth
- **Optimization**:
  - Enable compression
  - Use CDN for global access
  - Set appropriate lifecycle policies

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   gcloud builds log BUILD_ID
   
   # Test locally
   docker build -f Dockerfile.backend .
   ```

2. **Service Not Starting**
   ```bash
   # Check service logs
   gcloud run services logs read ai-stock-gpt-backend --region=us-central1
   
   # Verify environment variables
   gcloud run services describe ai-stock-gpt-backend --region=us-central1
   ```

3. **Frontend Not Loading**
   ```bash
   # Check bucket permissions
   gsutil iam get gs://ai-stock-gpt-frontend-static
   
   # Verify files uploaded
   gsutil ls gs://ai-stock-gpt-frontend-static/
   ```

### Performance Issues

1. **Slow Response Times**
   - Increase memory allocation
   - Optimize model loading
   - Use connection pooling

2. **High Costs**
   - Reduce max instances
   - Set appropriate scaling policies
   - Monitor usage patterns

## 🔄 Updates & Maintenance

### Update Backend
```bash
# Rebuild and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-stock-gpt-backend --file Dockerfile.backend .
gcloud run deploy ai-stock-gpt-backend --image gcr.io/YOUR_PROJECT_ID/ai-stock-gpt-backend --region=us-central1
```

### Update Frontend
```bash
# Rebuild and upload
npm run build
gsutil -m cp -r build/* gs://ai-stock-gpt-frontend-static/
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review GCP documentation
3. Check service logs for errors
4. Verify configuration files

## 🎯 Next Steps

After successful deployment:
1. Test all API endpoints
2. Configure custom domain (optional)
3. Set up monitoring alerts
4. Configure backup strategies
5. Implement CI/CD pipeline

---

**Note**: This deployment uses GCP's serverless services for cost-effectiveness and scalability. The backend automatically scales based on demand, and the frontend is served from Cloud Storage for optimal performance.
