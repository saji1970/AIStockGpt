"""
In-memory session store for multi-turn @learn conversations.

Sessions expire after 15 minutes of inactivity.  Each session tracks the
state-machine step and the data collected so far.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

_TTL_SECONDS = 900  # 15 minutes
_sessions: Dict[str, Dict[str, Any]] = {}


def _cleanup_expired() -> None:
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v["last_active"] > _TTL_SECONDS]
    for k in expired:
        del _sessions[k]


def create_session(key: Optional[str] = None) -> tuple:
    """Create a new learn session.  Returns (key, session_dict)."""
    _cleanup_expired()
    if not key:
        key = uuid.uuid4().hex[:12]
    _sessions[key] = {
        "step": "ask_symbol",
        "symbols": [],
        "direction_error": None,     # bullish_went_bearish | bearish_went_bullish
        "date_range": None,          # (start_str, end_str)
        "user_context": None,
        "retrospect_text": None,
        "last_active": time.time(),
        "created_at": time.time(),
    }
    return key, _sessions[key]


def get_session(key: str) -> Optional[Dict[str, Any]]:
    _cleanup_expired()
    sess = _sessions.get(key)
    if sess is not None:
        sess["last_active"] = time.time()
    return sess


def update_session(key: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sess = _sessions.get(key)
    if sess is None:
        return None
    sess.update(updates)
    sess["last_active"] = time.time()
    return sess


def delete_session(key: str) -> None:
    _sessions.pop(key, None)


def session_exists(key: str) -> bool:
    _cleanup_expired()
    return key in _sessions
