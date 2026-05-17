"""
Admin API routes for AI Stock GPT.
User management, system analytics, env key status, and training pipeline proxy.
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from .auth import AuthManager, UserCreate, get_current_admin
from .database import db_manager

logger = logging.getLogger(__name__)

# /api/admin avoids collision with React route /admin when UI is served from the same host
router = APIRouter(prefix="/api/admin", tags=["admin"])
auth_manager = AuthManager()

TRAIN_PIPE_URL = os.getenv("TRAIN_PIPE_URL", "http://127.0.0.1:8090").rstrip("/")
TRAIN_PIPE_SECRET = os.getenv("TRAIN_PIPE_SECRET", "")


# ── Request models ────────────────────────────────────────────

class AdminUserCreate(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    username: Optional[str] = None
    is_admin: bool = False
    is_active: bool = True


class AdminUserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None


class TrainingTriggerRequest(BaseModel):
    mode: Optional[str] = Field(None, description="quick or full")
    symbols: Optional[str] = Field(None, description="Comma-separated tickers")
    message: Optional[str] = None
    remote: str = "origin"
    branch: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────

def _mask_secret(value: Optional[str]) -> Dict[str, Any]:
    if not value or not str(value).strip():
        return {"configured": False, "preview": None}
    s = str(value).strip()
    if len(s) <= 8:
        preview = "***"
    else:
        preview = f"{s[:4]}...{s[-4:]}"
    return {"configured": True, "preview": preview}


def _env_keys_status() -> Dict[str, Any]:
    keys = {
        "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "alphavantage_apikey": os.getenv("alphavantage_apikey") or os.getenv("ALPHA_VANTAGE_KEY"),
        "FRED_API_KEY": os.getenv("FRED_API_KEY"),
        "SENDGRID_API_KEY": os.getenv("SENDGRID_API_KEY"),
        "HF_API_TOKEN": os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN"),
        "TRAIN_PIPE_SECRET": TRAIN_PIPE_SECRET,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
    }
    return {name: _mask_secret(val) for name, val in keys.items()}


def _train_pipe_request(method: str, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    url = f"{TRAIN_PIPE_URL}{path}"
    headers = {}
    if TRAIN_PIPE_SECRET:
        headers["X-Train-Pipe-Secret"] = TRAIN_PIPE_SECRET
    try:
        resp = requests.request(method, url, params=params, headers=headers, timeout=30)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=body)
        return body
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Training pipeline is not running. "
                f"Start it on your local machine with: python train_pipe.py"
            ),
        )


# ── Dashboard ─────────────────────────────────────────────────

@router.get("/dashboard")
async def admin_dashboard(_admin: Dict = Depends(get_current_admin)):
    analytics = db_manager.get_system_analytics()
    users = db_manager.list_users(offset=0, limit=1)
    return {
        "analytics": analytics,
        "total_users": users.get("total", analytics.get("total_users", 0)),
        "env_keys": _env_keys_status(),
        "train_pipe_url": TRAIN_PIPE_URL,
        "train_pipe_configured": bool(TRAIN_PIPE_SECRET) or TRAIN_PIPE_URL.startswith("http://127.0.0.1"),
    }


@router.get("/env-keys")
async def admin_env_keys(_admin: Dict = Depends(get_current_admin)):
    return {"keys": _env_keys_status()}


# ── Users ─────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    include_inactive: bool = Query(True),
    _admin: Dict = Depends(get_current_admin),
):
    return db_manager.list_users(offset=offset, limit=limit, search=search, include_inactive=include_inactive)


@router.get("/users/{user_id}")
async def get_user(user_id: str, _admin: Dict = Depends(get_current_admin)):
    detail = db_manager.get_user_admin_detail(user_id)
    if not detail:
        raise HTTPException(status_code=404, detail="User not found")
    return detail


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(body: AdminUserCreate, admin: Dict = Depends(get_current_admin)):
    existing = db_manager.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    username = body.username or body.email.split("@")[0]
    if db_manager.get_user_by_username(username):
        raise HTTPException(status_code=400, detail=f"Username '{username}' is taken")
    user_id = f"user_{int(datetime.utcnow().timestamp())}"
    ok = db_manager.create_user(
        user_id,
        {
            "email": body.email,
            "hashed_password": auth_manager.hash_password(body.password),
            "first_name": body.first_name,
            "last_name": body.last_name,
            "username": username,
            "is_active": body.is_active,
            "is_admin": body.is_admin,
        },
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create user")
    detail = db_manager.get_user_admin_detail(user_id)
    return {"message": "User created", "user": detail}


@router.patch("/users/{user_id}")
async def update_user(user_id: str, body: AdminUserUpdate, admin: Dict = Depends(get_current_admin)):
    if user_id == admin.get("id") and body.is_admin is False:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin role")
    if user_id == admin.get("id") and body.is_active is False:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    user = db_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    updates: Dict[str, Any] = {}
    for field in ("first_name", "last_name", "username", "is_active", "is_admin"):
        val = getattr(body, field, None)
        if val is not None:
            updates[field] = val
    if body.password:
        updates["hashed_password"] = auth_manager.hash_password(body.password)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    if not db_manager.update_user(user_id, updates):
        raise HTTPException(status_code=500, detail="Update failed")
    return {"message": "User updated", "user": db_manager.get_user_admin_detail(user_id)}


@router.post("/users/{user_id}/block")
async def block_user(user_id: str, admin: Dict = Depends(get_current_admin)):
    if user_id == admin.get("id"):
        raise HTTPException(status_code=400, detail="Cannot block yourself")
    if not db_manager.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    db_manager.update_user(user_id, {"is_active": False})
    return {"message": "User blocked", "user_id": user_id}


@router.post("/users/{user_id}/unblock")
async def unblock_user(user_id: str, _admin: Dict = Depends(get_current_admin)):
    if not db_manager.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    db_manager.update_user(user_id, {"is_active": True})
    return {"message": "User unblocked", "user_id": user_id}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: Dict = Depends(get_current_admin)):
    if user_id == admin.get("id"):
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not db_manager.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted", "user_id": user_id}


# ── Training pipeline (proxy to train_pipe.py) ────────────────

@router.get("/training/status")
async def training_status(_admin: Dict = Depends(get_current_admin)):
    return _train_pipe_request("GET", "/status")


@router.get("/training/status/detail")
async def training_status_detail(_admin: Dict = Depends(get_current_admin)):
    return _train_pipe_request("GET", "/status/detail")


@router.post("/training/train")
async def training_start(
    body: TrainingTriggerRequest,
    _admin: Dict = Depends(get_current_admin),
):
    params: Dict[str, Any] = {}
    if body.mode:
        params["mode"] = body.mode
    if body.symbols:
        params["symbols"] = body.symbols
    return _train_pipe_request("POST", "/train", params=params or None)


@router.post("/training/commit")
async def training_commit(
    message: Optional[str] = Query(None),
    _admin: Dict = Depends(get_current_admin),
):
    params = {"message": message} if message else None
    return _train_pipe_request("POST", "/commit", params=params)


@router.post("/training/push")
async def training_push(
    remote: str = Query("origin"),
    branch: Optional[str] = Query(None),
    _admin: Dict = Depends(get_current_admin),
):
    params: Dict[str, Any] = {"remote": remote}
    if branch:
        params["branch"] = branch
    return _train_pipe_request("POST", "/push", params=params)


@router.post("/training/train-commit-push")
async def training_full_pipeline(
    body: TrainingTriggerRequest,
    _admin: Dict = Depends(get_current_admin),
):
    params: Dict[str, Any] = {"remote": body.remote}
    if body.mode:
        params["mode"] = body.mode
    if body.symbols:
        params["symbols"] = body.symbols
    if body.message:
        params["message"] = body.message
    if body.branch:
        params["branch"] = body.branch
    return _train_pipe_request("POST", "/train-commit-push", params=params)
