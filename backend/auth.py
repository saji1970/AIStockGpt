"""
Authentication module for AI Stock GPT
Handles user authentication, JWT tokens, and security
"""

import os
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging
from .database import db_manager

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"


def _admin_email_allowlist() -> set:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()

class UserCreate(BaseModel):
    """User registration model"""
    email: str
    password: str
    first_name: str
    last_name: str
    username: Optional[str] = None

class UserLogin(BaseModel):
    """User login model"""
    email: str
    password: str

class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    email: str
    first_name: str
    last_name: str
    username: Optional[str]
    is_active: bool
    is_admin: bool = False
    created_at: datetime

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class ChangePasswordRequest(BaseModel):
    """Change password request model"""
    current_password: str
    new_password: str

class ForgotPasswordRequest(BaseModel):
    """Forgot password request model"""
    email: str

class ResetPasswordRequest(BaseModel):
    """Reset password request model"""
    token: str
    new_password: str

class AuthManager:
    """Manages authentication operations"""
    
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.error(f"JWT error: {e}")
            return None
    
    def register_user(self, user_data: UserCreate) -> UserResponse:
        """Register a new user"""
        try:
            # Check if email already exists
            existing_user = db_manager.get_user_by_email(user_data.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

            # Determine username
            username = user_data.username or user_data.email.split('@')[0]

            # Check if username already exists
            existing_username = db_manager.get_user_by_username(username)
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Username '{username}' is already taken"
                )

            # Hash password
            hashed_password = self.hash_password(user_data.password)

            # Create user data
            user_dict = {
                "email": user_data.email,
                "hashed_password": hashed_password,
                "first_name": user_data.first_name,
                "last_name": user_data.last_name,
                "username": username,
                "is_active": True
            }
            
            # Generate user ID (you might want to use UUID)
            user_id = f"user_{int(datetime.utcnow().timestamp())}"
            
            # Save to database
            if db_manager.create_user(user_id, user_dict):
                return UserResponse(
                    user_id=user_id,
                    email=user_data.email,
                    first_name=user_data.first_name,
                    last_name=user_data.last_name,
                    username=user_dict["username"],
                    is_active=True,
                    created_at=datetime.utcnow()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with email and password"""
        try:
            user = db_manager.get_user_by_email(email.strip().lower())
            if not user:
                return None
            
            if not self.verify_password(password, user.get("hashed_password", "")):
                return None
            
            if not user.get("is_active", False):
                return None

            # Bootstrap admin from ADMIN_EMAILS env
            email = (user.get("email") or "").lower()
            if email in _admin_email_allowlist() and not user.get("is_admin"):
                db_manager.update_user(user["user_id"], {"is_admin": True})
                user["is_admin"] = True
            
            return user
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def login_user(self, user_data: UserLogin) -> Token:
        """Login a user and return tokens"""
        user = self.authenticate_user(
            (user_data.email or "").strip().lower(),
            user_data.password,
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user["user_id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        refresh_token = self.create_refresh_token(
            data={"sub": user["user_id"]}
        )
        
        # Update last login
        db_manager.update_user(user["user_id"], {"last_login": datetime.utcnow()})

        user = db_manager.get_user(user["user_id"]) or user
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def refresh_access_token(self, refresh_token: str) -> Token:
        """Refresh an access token using a refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user_id, "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change password for an authenticated user."""
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if not self.verify_password(current_password, user.get("hashed_password", "")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        from .security import security_config
        if not security_config.validate_password(new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters with uppercase, lowercase, number, and special character"
            )

        if current_password == new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )

        new_hashed = self.hash_password(new_password)
        success = db_manager.update_user(user_id, {"hashed_password": new_hashed})
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        return True

    def create_password_reset(self, email: str) -> Optional[str]:
        """Create a password reset token. Returns None if user not found."""
        user = db_manager.get_user_by_email(email)
        if not user:
            return None

        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=30)

        success = db_manager.create_password_reset_token(
            user_id=user["user_id"],
            token=token,
            expires_at=expires_at,
        )
        return token if success else None

    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using a valid reset token."""
        from .security import security_config
        if not security_config.validate_password(new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character"
            )

        token_data = db_manager.get_valid_reset_token(token)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        new_hashed = self.hash_password(new_password)
        success = db_manager.update_user(token_data["user_id"], {"hashed_password": new_hashed})
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset password"
            )

        db_manager.mark_reset_token_used(token)
        return True

# Global auth manager instance
auth_manager = AuthManager()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = db_manager.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return db_manager.sanitize_user(user)

async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current active user"""
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return db_manager.sanitize_user(current_user)


async def get_current_admin(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Require an authenticated admin user."""
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
