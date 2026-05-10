# Test Deployment Script for AI Stock GPT
# This script tests the deployed application

param(
    [string]$BackendUrl = "",
    [string]$FrontendUrl = ""
)

Write-Host "🧪 Testing AI Stock GPT Deployment" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# If URLs not provided, try to read from file
if (-not $BackendUrl -or -not $FrontendUrl) {
    if (Test-Path "deployment-urls.txt") {
        $urls = Get-Content "deployment-urls.txt"
        foreach ($line in $urls) {
            if ($line -match "Backend URL: (.+)") {
                $BackendUrl = $matches[1]
            }
            if ($line -match "Frontend URL: (.+)") {
                $FrontendUrl = $matches[1]
            }
        }
    }
}

if (-not $BackendUrl -or -not $FrontendUrl) {
    Write-Host "❌ Please provide BackendUrl and FrontendUrl parameters" -ForegroundColor Red
    Write-Host "Usage: .\test-deployment.ps1 -BackendUrl 'https://...' -FrontendUrl 'https://...'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Backend URL: $BackendUrl" -ForegroundColor Cyan
Write-Host "Frontend URL: $FrontendUrl" -ForegroundColor Cyan
Write-Host ""

# Test Backend Health
Write-Host "🔍 Testing Backend Health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BackendUrl/status" -Method GET -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is healthy" -ForegroundColor Green
        $status = $response.Content | ConvertFrom-Json
        Write-Host "   API Status: $($status.api_status)" -ForegroundColor Gray
        Write-Host "   NLP Processor: $($status.nlp_processor)" -ForegroundColor Gray
        Write-Host "   Cached Models: $($status.cached_models)" -ForegroundColor Gray
    } else {
        Write-Host "❌ Backend health check failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Backend health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test Backend Root Endpoint
Write-Host "🔍 Testing Backend Root Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BackendUrl/" -Method GET -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend root endpoint is working" -ForegroundColor Green
        $data = $response.Content | ConvertFrom-Json
        Write-Host "   Message: $($data.message)" -ForegroundColor Gray
        Write-Host "   Version: $($data.version)" -ForegroundColor Gray
    } else {
        Write-Host "❌ Backend root endpoint failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Backend root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test Chat Endpoint
Write-Host "🔍 Testing Chat Endpoint..." -ForegroundColor Yellow
try {
    $chatBody = @{
        message = "What can you do?"
        timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "$BackendUrl/chat" -Method POST -Body $chatBody -ContentType "application/json" -TimeoutSec 60
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Chat endpoint is working" -ForegroundColor Green
        $chatResponse = $response.Content | ConvertFrom-Json
        Write-Host "   Response received: $($chatResponse.message.Length) characters" -ForegroundColor Gray
        Write-Host "   Confidence: $($chatResponse.confidence)" -ForegroundColor Gray
    } else {
        Write-Host "❌ Chat endpoint failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Chat endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test Stock Prediction Endpoint
Write-Host "🔍 Testing Stock Prediction Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BackendUrl/predict/AAPL" -Method GET -TimeoutSec 60
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Stock prediction endpoint is working" -ForegroundColor Green
        $prediction = $response.Content | ConvertFrom-Json
        if ($prediction.stockData) {
            Write-Host "   Symbol: $($prediction.stockData.symbol)" -ForegroundColor Gray
            Write-Host "   Current Price: $($prediction.stockData.currentPrice)" -ForegroundColor Gray
            Write-Host "   Predicted Price: $($prediction.stockData.predictedPrice)" -ForegroundColor Gray
        }
    } else {
        Write-Host "❌ Stock prediction endpoint failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Stock prediction endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test Frontend
Write-Host "🔍 Testing Frontend..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$FrontendUrl" -Method GET -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Frontend is accessible" -ForegroundColor Green
        Write-Host "   Content Type: $($response.Headers.'Content-Type')" -ForegroundColor Gray
        Write-Host "   Content Length: $($response.Content.Length) characters" -ForegroundColor Gray
    } else {
        Write-Host "❌ Frontend failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Frontend test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test API Documentation
Write-Host "🔍 Testing API Documentation..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BackendUrl/docs" -Method GET -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API documentation is accessible" -ForegroundColor Green
        Write-Host "   Swagger UI available at: $BackendUrl/docs" -ForegroundColor Gray
    } else {
        Write-Host "❌ API documentation failed with status: $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ API documentation test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Deployment Testing Complete!" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor Yellow
Write-Host "   Backend: $BackendUrl" -ForegroundColor Cyan
Write-Host "   Frontend: $FrontendUrl" -ForegroundColor Cyan
Write-Host "   API Docs: $BackendUrl/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Your AI Stock GPT application is ready to use!" -ForegroundColor Green
