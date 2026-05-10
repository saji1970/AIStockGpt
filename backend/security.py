"""
Security module for AI Stock GPT
Handles rate limiting, CORS, input validation, and security middleware
"""

import os
import re
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator
import logging

logger = logging.getLogger(__name__)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)

# Security configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://aistockgpt.com",
    "https://www.aistockgpt.com",
    os.getenv("RAILWAY_PUBLIC_DOMAIN", ""),
]
# Filter out empty strings
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]

# Input validation patterns
STOCK_SYMBOL_PATTERN = r'^[A-Z]{1,5}$'
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
PASSWORD_PATTERN = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

class SecurityConfig:
    """Security configuration class"""
    
    def __init__(self):
        self.allowed_origins = ALLOWED_ORIGINS
        self.stock_symbol_pattern = re.compile(STOCK_SYMBOL_PATTERN)
        self.email_pattern = re.compile(EMAIL_PATTERN)
        self.password_pattern = re.compile(PASSWORD_PATTERN)
    
    def validate_stock_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        return bool(self.stock_symbol_pattern.match(symbol.upper()))
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False
        return bool(self.email_pattern.match(email.lower()))
    
    def validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if not password or not isinstance(password, str):
            return False
        return bool(self.password_pattern.match(password))
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        # Limit length
        return sanitized[:1000]
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or not isinstance(api_key, str):
            return False
        # Basic validation - you might want to check against a database
        return len(api_key) >= 20 and api_key.startswith('sk-')

# Global security config
security_config = SecurityConfig()

# Input validation models
class StockSymbolRequest(BaseModel):
    """Stock symbol request model with validation"""
    symbol: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not security_config.validate_stock_symbol(v):
            raise ValueError('Invalid stock symbol format')
        return v.upper()

class ChatMessageRequest(BaseModel):
    """Chat message request model with validation"""
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        return security_config.sanitize_input(v)

class UserRegistrationRequest(BaseModel):
    """User registration request model with validation"""
    email: str
    password: str
    first_name: str
    last_name: str
    username: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if not security_config.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if not security_config.validate_password(v):
            raise ValueError('Password must be at least 8 characters with uppercase, lowercase, number, and special character')
        return v
    
    @validator('first_name', 'last_name')
    def validate_names(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 50:
            raise ValueError('Name too long (max 50 characters)')
        return security_config.sanitize_input(v.strip())
    
    @validator('username')
    def validate_username(cls, v):
        if v:
            if len(v) < 3 or len(v) > 20:
                raise ValueError('Username must be between 3 and 20 characters')
            if not re.match(r'^[a-zA-Z0-9_]+$', v):
                raise ValueError('Username can only contain letters, numbers, and underscores')
        return v

# Security middleware
class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup security middleware"""
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Add rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Add request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all requests for security monitoring"""
            start_time = request.scope.get("start_time", 0)
            
            # Log request details
            logger.info(
                "Request received",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown")
            )
            
            response = await call_next(request)
            
            # Log response details
            process_time = request.scope.get("start_time", 0) - start_time
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time
            )
            
            return response
        
        # Add security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            """Add security headers to responses"""
            response = await call_next(request)
            
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
            
            return response

# Rate limiting decorators
def rate_limit_public(requests_per_minute: int = 60):
    """Rate limit for public endpoints"""
    return limiter.limit(f"{requests_per_minute}/minute")

def rate_limit_authenticated(requests_per_minute: int = 120):
    """Rate limit for authenticated endpoints"""
    return limiter.limit(f"{requests_per_minute}/minute")

def rate_limit_sensitive(requests_per_minute: int = 10):
    """Rate limit for sensitive operations"""
    return limiter.limit(f"{requests_per_minute}/minute")

# Security utilities
def validate_api_request(request: Request) -> bool:
    """Validate API request for security"""
    try:
        # Check for required headers
        user_agent = request.headers.get("user-agent")
        if not user_agent or user_agent.lower() in ["curl", "wget", "python"]:
            # Allow but log
            logger.warning("Request from command line tool", user_agent=user_agent)
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        return True
    
    except Exception as e:
        logger.error(f"API request validation failed: {e}")
        return False

def check_suspicious_activity(request: Request) -> bool:
    """Check for suspicious activity patterns"""
    try:
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"sqlmap",
            r"nikto",
            r"nmap",
            r"hydra",
            r"medusa",
            r"dirbuster",
            r"gobuster"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent.lower()):
                logger.warning("Suspicious activity detected", 
                             client_ip=client_ip, 
                             user_agent=user_agent,
                             pattern=pattern)
                return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error checking suspicious activity: {e}")
        return False

# Error handlers
async def security_exception_handler(request: Request, exc: Exception):
    """Handle security-related exceptions"""
    logger.error(f"Security exception: {exc}", 
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else "unknown")
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": "Access denied for security reasons"}
    )

# Input sanitization utilities
def sanitize_stock_symbol(symbol: str) -> str:
    """Sanitize stock symbol input"""
    if not symbol:
        return ""
    
    # Remove non-alphabetic characters and limit length
    sanitized = re.sub(r'[^A-Za-z]', '', symbol.upper())
    return sanitized[:5]

def sanitize_numeric_input(value: str) -> Optional[float]:
    """Sanitize numeric input"""
    try:
        if not value:
            return None
        
        # Remove non-numeric characters except decimal point
        sanitized = re.sub(r'[^0-9.]', '', value)
        return float(sanitized)
    
    except (ValueError, TypeError):
        return None

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range for API requests"""
    try:
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if end date is after start date
        if end <= start:
            return False
        
        # Check if range is reasonable (max 2 years)
        if (end - start) > timedelta(days=730):
            return False
        
        return True
    
    except (ValueError, TypeError):
        return False
