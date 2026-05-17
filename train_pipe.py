#!/usr/bin/env python3
"""
AI Stock GPT - Training Pipeline Server.

Lightweight HTTP server that triggers model training, git commit, and push.
Runs independently of the main API on port 8090.

Usage:
    python train_pipe.py                    # Start the server
    python train_pipe.py --port 9000        # Custom port

Trigger via curl:
    curl -X POST http://localhost:8090/train                         # Full training (all 168 symbols)
    curl -X POST http://localhost:8090/train?mode=quick              # Quick (14 symbols)
    curl -X POST http://localhost:8090/train?symbols=AAPL,MSFT,TSLA  # Specific symbols
    curl    http://localhost:8090/status                              # Check training status
    curl -X POST http://localhost:8090/commit                        # Git commit models
    curl -X POST http://localhost:8090/push                          # Git push
    curl -X POST http://localhost:8090/train-commit-push?mode=quick  # All-in-one
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
import uvicorn

from backend.ml.feature_pipeline import FeaturePipeline
from backend.ml.xgboost_model import XGBoostPredictor
from backend.data.alphavantage_collector import AlphaVantageCollector

from train_models import (
    ALL_SYMBOLS, QUICK_SYMBOLS, train_symbols, print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────

_state: Dict[str, Any] = {
    "status": "idle",          # idle | training | done | failed
    "started_at": None,
    "finished_at": None,
    "mode": None,
    "symbols": [],
    "progress": 0,             # symbols completed so far
    "total": 0,
    "current_symbol": None,
    "results": None,
    "error": None,
    "elapsed": None,
}
_lock = threading.Lock()

# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(
    title="AI Stock GPT - Training Pipeline",
    version="1.0.0",
)

GIT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PIPE_SECRET = os.getenv("TRAIN_PIPE_SECRET", "")


@app.middleware("http")
async def _train_pipe_auth_middleware(request: Request, call_next):
    """Optional shared secret for mutating endpoints (local admin UI proxies here)."""
    if request.method in ("GET", "HEAD", "OPTIONS") and request.url.path in ("/", "/status", "/status/detail"):
        return await call_next(request)
    if not TRAIN_PIPE_SECRET:
        return await call_next(request)
    provided = request.headers.get("X-Train-Pipe-Secret") or request.headers.get("x-train-pipe-secret")
    if provided != TRAIN_PIPE_SECRET:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-Train-Pipe-Secret"})
    return await call_next(request)


def _reset_state(mode: str, symbols: List[str]) -> None:
    with _lock:
        _state.update({
            "status": "training",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "mode": mode,
            "symbols": symbols,
            "progress": 0,
            "total": len(symbols),
            "current_symbol": None,
            "results": None,
            "error": None,
            "elapsed": None,
        })


def _training_thread(symbols: List[str], mode: str, tune: bool) -> None:
    """Run training in a background thread."""
    try:
        av_collector = AlphaVantageCollector()
        pipeline = FeaturePipeline(av_collector=av_collector)
        predictor = XGBoostPredictor(models_dir="models")

        total_start = time.time()

        # Wrap train_symbols with progress tracking
        results: Dict[str, Any] = {}
        for i, symbol in enumerate(symbols, 1):
            with _lock:
                _state["progress"] = i - 1
                _state["current_symbol"] = symbol

            logger.info(f"[{i}/{len(symbols)}] Training {symbol}...")
            start = time.time()
            try:
                features = pipeline.build_features(symbol, lookback_days=756)
                metrics = predictor.train(symbol, features)
                elapsed = time.time() - start
                results[symbol] = {
                    "status": "success",
                    "accuracy": metrics["accuracy"],
                    "calibrated_accuracy": metrics.get("calibrated_accuracy", metrics["accuracy"]),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "n_features": metrics.get("n_features", 0),
                    "ensemble": metrics.get("ensemble", False),
                    "elapsed": round(elapsed, 1),
                }
                logger.info(f"  {symbol} done  acc={metrics['accuracy']:.2%}  f1={metrics['f1']:.2%}")
            except Exception as e:
                elapsed = time.time() - start
                results[symbol] = {"status": "failed", "error": str(e), "elapsed": round(elapsed, 1)}
                logger.error(f"  {symbol} FAILED: {e}")

        total_elapsed = time.time() - total_start

        # Save training report
        _save_report(results, total_elapsed)

        succeeded = sum(1 for r in results.values() if r["status"] == "success")
        failed = sum(1 for r in results.values() if r["status"] == "failed")

        with _lock:
            _state.update({
                "status": "done",
                "finished_at": datetime.now().isoformat(),
                "progress": len(symbols),
                "current_symbol": None,
                "results": {
                    "succeeded": succeeded,
                    "failed": failed,
                    "total": len(symbols),
                    "per_symbol": results,
                },
                "elapsed": round(total_elapsed, 1),
            })
        logger.info(f"Training complete: {succeeded}/{len(symbols)} succeeded in {total_elapsed:.0f}s")

    except Exception as e:
        logger.error(f"Training thread error: {e}")
        with _lock:
            _state.update({
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "error": str(e),
            })


def _save_report(results: Dict, total_elapsed: float) -> None:
    """Save training report JSON (same format as train_models.py)."""
    import numpy as np
    import shutil

    os.makedirs("models", exist_ok=True)
    report_path = os.path.join("models", "training_report.json")
    prev_path = os.path.join("models", "training_report_prev.json")

    if os.path.exists(report_path):
        shutil.copy2(report_path, prev_path)

    succeeded = [s for s, r in results.items() if r["status"] == "success"]
    accs = [results[s]["accuracy"] for s in succeeded] if succeeded else []
    cal_accs = [results[s].get("calibrated_accuracy", results[s]["accuracy"]) for s in succeeded] if succeeded else []
    f1s = [results[s]["f1"] for s in succeeded] if succeeded else []

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_symbols": len(results),
        "succeeded": len(succeeded),
        "failed": len(results) - len(succeeded),
        "total_time": total_elapsed,
        "results": results,
    }
    if succeeded:
        report["avg_accuracy"] = float(np.mean(accs))
        report["avg_calibrated_accuracy"] = float(np.mean(cal_accs))
        report["avg_f1"] = float(np.mean(f1s))
        report["median_accuracy"] = float(np.median(accs))

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")


def _resolve_symbols(mode: Optional[str], symbols_csv: Optional[str]) -> (str, List[str]):
    """Resolve the training mode and symbol list."""
    if symbols_csv:
        syms = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
        return "custom", syms
    if mode == "quick":
        return "quick", list(QUICK_SYMBOLS)
    return "full", list(ALL_SYMBOLS)


def _git_run(*args: str) -> Dict[str, Any]:
    """Run a git command and return result dict."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=GIT_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)}


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "AI Stock GPT Training Pipeline",
        "endpoints": {
            "POST /train": "Trigger model training (params: mode=quick|full, symbols=AAPL,MSFT)",
            "GET  /status": "Check training status and progress",
            "POST /commit": "Git commit trained models (param: message=...)",
            "POST /push": "Git push to remote",
            "POST /train-commit-push": "All-in-one: train, commit, push",
        },
    }


@app.post("/train")
def train(
    mode: Optional[str] = Query(None, description="quick or full (default: full)"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    tune: bool = Query(False, description="Run Optuna tuning"),
):
    """Start model training in the background."""
    with _lock:
        if _state["status"] == "training":
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Training already in progress",
                    "started_at": _state["started_at"],
                    "progress": f"{_state['progress']}/{_state['total']}",
                    "current_symbol": _state["current_symbol"],
                },
            )

    resolved_mode, symbol_list = _resolve_symbols(mode, symbols)
    _reset_state(resolved_mode, symbol_list)

    thread = threading.Thread(
        target=_training_thread,
        args=(symbol_list, resolved_mode, tune),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Training started",
        "mode": resolved_mode,
        "total_symbols": len(symbol_list),
        "symbols": symbol_list[:20] if len(symbol_list) > 20 else symbol_list,
        "check_progress": "GET /status",
    }


@app.get("/status")
def status():
    """Return current training status."""
    with _lock:
        resp = dict(_state)
    # Strip per-symbol detail from top-level to keep response small
    if resp.get("results") and resp["results"].get("per_symbol"):
        per_sym = resp["results"]["per_symbol"]
        resp["results"] = {
            k: v for k, v in resp["results"].items() if k != "per_symbol"
        }
        resp["results"]["symbols_succeeded"] = [
            s for s, r in per_sym.items() if r["status"] == "success"
        ]
        resp["results"]["symbols_failed"] = [
            s for s, r in per_sym.items() if r["status"] == "failed"
        ]
    return resp


@app.get("/status/detail")
def status_detail():
    """Return full training status including per-symbol results."""
    with _lock:
        return dict(_state)


@app.post("/commit")
def commit(
    message: Optional[str] = Query(None, description="Commit message"),
):
    """Git add and commit trained model files."""
    if not message:
        # Auto-generate commit message
        report_path = os.path.join("models", "training_report.json")
        n_models = "?"
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    rpt = json.load(f)
                n_models = rpt.get("succeeded", "?")
            except Exception:
                pass
        message = f"Retrain {n_models} models via training pipeline"

    # Stage model files
    add_result = _git_run("add", "models/")
    if not add_result["ok"]:
        return JSONResponse(status_code=500, content={"error": "git add failed", "detail": add_result})

    # Commit
    commit_result = _git_run("commit", "-m", message)
    if not commit_result["ok"]:
        # Check if "nothing to commit"
        if "nothing to commit" in commit_result["stdout"] or "nothing to commit" in commit_result["stderr"]:
            return {"message": "Nothing to commit - models are up to date", "ok": True}
        return JSONResponse(status_code=500, content={"error": "git commit failed", "detail": commit_result})

    return {
        "message": "Models committed",
        "ok": True,
        "commit_message": message,
        "git_output": commit_result["stdout"],
    }


@app.post("/push")
def push(
    remote: str = Query("origin", description="Git remote name"),
    branch: Optional[str] = Query(None, description="Branch name (default: current)"),
):
    """Git push to remote."""
    cmd = ["push", remote]
    if branch:
        cmd.append(branch)

    result = _git_run(*cmd)
    if not result["ok"]:
        return JSONResponse(status_code=500, content={"error": "git push failed", "detail": result})

    return {"message": "Pushed to remote", "ok": True, "git_output": result["stdout"] or result["stderr"]}


@app.post("/train-commit-push")
def train_commit_push(
    mode: Optional[str] = Query(None, description="quick or full"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    message: Optional[str] = Query(None, description="Commit message"),
    remote: str = Query("origin", description="Git remote"),
    branch: Optional[str] = Query(None, description="Branch name"),
):
    """All-in-one: train models, commit, push. Training runs in background;
    commit+push happen automatically when training finishes."""
    with _lock:
        if _state["status"] == "training":
            return JSONResponse(
                status_code=409,
                content={"error": "Training already in progress"},
            )

    resolved_mode, symbol_list = _resolve_symbols(mode, symbols)
    _reset_state(resolved_mode, symbol_list)

    def _full_pipeline():
        # Train
        _training_thread(symbol_list, resolved_mode, tune=False)

        with _lock:
            train_ok = _state["status"] == "done"

        if not train_ok:
            logger.error("Training failed - skipping commit & push")
            return

        # Commit
        succeeded = 0
        with _lock:
            if _state.get("results"):
                succeeded = _state["results"].get("succeeded", 0)

        commit_msg = message or f"Retrain {succeeded} models via training pipeline"
        add_r = _git_run("add", "models/")
        if add_r["ok"]:
            commit_r = _git_run("commit", "-m", commit_msg)
            if commit_r["ok"]:
                logger.info(f"Committed: {commit_msg}")
            else:
                logger.warning(f"Commit skipped: {commit_r['stderr']}")

        # Push
        push_cmd = ["push", remote]
        if branch:
            push_cmd.append(branch)
        push_r = _git_run(*push_cmd)
        if push_r["ok"]:
            logger.info("Pushed to remote")
        else:
            logger.warning(f"Push failed: {push_r['stderr']}")

    thread = threading.Thread(target=_full_pipeline, daemon=True)
    thread.start()

    return {
        "message": "Training pipeline started (will auto-commit & push on completion)",
        "mode": resolved_mode,
        "total_symbols": len(symbol_list),
        "check_progress": "GET /status",
    }


# ── Main ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AI Stock GPT Training Pipeline Server")
    parser.add_argument("--port", type=int, default=8090, help="Server port (default: 8090)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting Training Pipeline on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
