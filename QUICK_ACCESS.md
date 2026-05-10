# 🚀 AI Stock GPT - Quick Access Guide

## ✅ **Working URLs**

### **Frontend (React App)**
- **Primary URL**: https://ai-stock-gpt-frontend-static.storage.googleapis.com/
- **Alternative URL**: https://storage.googleapis.com/ai-stock-gpt-frontend-static/index.html

### **Backend API**
- **Base URL**: https://ai-stock-gpt-backend-1012090067429.us-central1.run.app
- **Status Check**: https://ai-stock-gpt-backend-1012090067429.us-central1.run.app/

## 🔧 **API Endpoints**

### **Available Endpoints**
- `GET /` - Status check
- `POST /chat` - Chat with AI
- `GET /predict/{symbol}` - Stock prediction
- `GET /technical/{symbol}` - Technical analysis
- `GET /market-analysis` - Market news and sentiment
- `POST /run-pipeline` - Run Parallel AI pipeline

### **Example Usage**
```bash
# Test backend
curl https://ai-stock-gpt-backend-1012090067429.us-central1.run.app/

# Get stock prediction for AAPL
curl https://ai-stock-gpt-backend-1012090067429.us-central1.run.app/predict/AAPL

# Chat with AI
curl -X POST https://ai-stock-gpt-backend-1012090067429.us-central1.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current price of AAPL?"}'
```

## 🐛 **Troubleshooting**

### **If you get "NoSuchKey" error:**
- Make sure you're using the correct URL: `https://ai-stock-gpt-frontend-static.storage.googleapis.com/`
- The file is `index.html`, not `index.htm`

### **If frontend doesn't load (403/404 errors):**
- **Clear browser cache**: Press `Ctrl+Shift+R` (hard refresh)
- **Force refresh assets**: Run `.\force-refresh.ps1`
- **Test assets**: Run `.\test-frontend.ps1`
- Try the alternative URL
- Check if the backend is responding

### **If backend doesn't respond:**
- Check the service status: `gcloud run services list --region us-central1`
- View logs: `gcloud logging read "resource.type=cloud_run_revision"`

## 📱 **How to Use**

1. **Open the frontend URL** in your browser
2. **Type your question** in the chat interface
3. **Ask about stocks** like:
   - "What is the current price of AAPL?"
   - "Predict the stock price for TSLA"
   - "Show technical analysis for GOOGL"
   - "What's the market sentiment today?"

## 🔄 **Update Frontend**
```powershell
# Rebuild and redeploy (with automatic path fixing)
.\deploy-static.ps1

# Or manually:
npm run build
# Fix paths automatically
.\fix-paths.ps1
gsutil -m rsync -r -d build/ gs://ai-stock-gpt-frontend-static/
```

## 🔄 **Update Backend**
```powershell
# Rebuild and redeploy
gcloud builds submit --config cloudbuild.yaml .
gcloud run deploy ai-stock-gpt-backend --image gcr.io/stockbroker-28983/ai-stock-gpt-backend:latest
```

---

**Status**: ✅ **FULLY OPERATIONAL**  
**Last Updated**: August 31, 2025
