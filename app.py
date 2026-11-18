"""app.py -- FastAPI server for the DIP histogram-equalization workbench.

Run with:  python app.py          (serves http://localhost:8000)
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import os
import platform
import re
import secrets
import shutil
import sqlite3
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                              Response)
from fastapi.staticfiles import StaticFiles

import dip_core as dc
import plotting
import simulation

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
SAMPLE = BASE / "samples" / "day.jpg"
DATA_DIR = BASE / "data"

# Vercel serverless functions have a read-only source tree; only /tmp can be
# written.  Copy the bundled accounts database into /tmp on a cold start and
# keep the session secret stable (bundled file or DIP_SESSION_SECRET env var).
ON_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
if ON_VERCEL:
    RUNTIME_DIR = Path(tempfile.gettempdir()) / "dip-studio"
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = RUNTIME_DIR / "users.db"
    if not DB_PATH.exists() and (DATA_DIR / "users.db").exists():
        shutil.copy2(DATA_DIR / "users.db", DB_PATH)
    SECRET_PATH = DATA_DIR / "session.key"
else:
    DB_PATH = DATA_DIR / "users.db"
    SECRET_PATH = DATA_DIR / "session.key"

SESSION_TTL = 8 * 60 * 60  # 8 hours
# A signed copy of the account travels in its own cookie.  Vercel keeps the
# SQLite file in /tmp, which is discarded when an instance restarts; the
# passport lets that same browser log in and rebuild its account on any
# later instance.  Logout clears only the session, never the passport.
ACCOUNT_COOKIE = "dip_account"
ACCOUNT_TTL = 365 * 24 * 60 * 60  # 1 year
CONDITIONS = ["day", "midday", "evening", "night"]
DEFAULT_PARAMS = {
    "width": 640,
    "height": 480,
    "clip_limit": 2.0,
    "tile": 8,
    "stretch_lo": 1,
    "stretch_hi": 99,
    "seed": 11,
    "gains": {"midday": 2.6, "evening": 0.20, "night": 0.012},
}

app = FastAPI(title="DIP Studio", version="1.0.0",
              description="Histogram equalization workbench")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Small in-memory run cache: the sample scene and repeated parameter runs
# reload instantly, and figures are rendered lazily on first view.  Keyed by
# the exact inputs and parameters.
_RUNS: "OrderedDict[str, dict]" = OrderedDict()
_RUNS_MAX = 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def jpeg_url(img: np.ndarray, quality: int = 84) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def thumb_url(img: np.ndarray, width: int = 512) -> str:
    """Small JPEG for the before/after slider (keeps the JSON payload light)."""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA)
    return jpeg_url(img, 84)


def png_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def clean_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, (np.floating, float)):
            f = float(v)
            out[k] = None if (f != f or f in (float("inf"), float("-inf"))) else round(f, 4)
        else:
            out[k] = int(v)
    return out


def parse_params(raw: str) -> dict:
    p = dict(DEFAULT_PARAMS)
    p["gains"] = dict(DEFAULT_PARAMS["gains"])
    try:
        incoming = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise ValueError("params must be valid JSON")
    if not isinstance(incoming, dict):
        raise ValueError("params must be a JSON object")
    for key in ("width", "height", "tile", "stretch_lo", "stretch_hi", "seed"):
        if key in incoming:
            p[key] = int(incoming[key])
    for key in ("clip_limit",):
        if key in incoming:
            p[key] = float(incoming[key])
    gains = incoming.get("gains") or {}
    for cond, value in gains.items():
        if cond in p["gains"]:
            p["gains"][cond] = float(value)

    p["width"] = max(160, min(1600, p["width"]))
    p["height"] = max(120, min(1200, p["height"]))
    p["tile"] = max(2, min(64, p["tile"]))
    p["clip_limit"] = max(0.1, min(20.0, p["clip_limit"]))
    lo, hi = sorted((int(p["stretch_lo"]), int(p["stretch_hi"])))
    if not (0 <= lo < hi <= 100):
        lo, hi = 1, 99
    p["stretch_lo"], p["stretch_hi"] = lo, hi
    for cond in p["gains"]:
        p["gains"][cond] = max(0.0, min(50.0, p["gains"][cond]))
    return p


# ---------------------------------------------------------------------------
# accounts and sessions -- standard library only (SQLite + PBKDF2 + HMAC)
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


if ON_VERCEL:
    env_secret = os.environ.get("DIP_SESSION_SECRET", "").strip()
    if env_secret:
        SESSION_SECRET = env_secret.encode("utf-8")
    elif SECRET_PATH.exists():
        SESSION_SECRET = SECRET_PATH.read_text(encoding="ascii").strip().encode("ascii")
    else:
        SESSION_SECRET = secrets.token_bytes(32)
else:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_PATH.exists():
        SECRET_PATH.write_text(secrets.token_hex(32), encoding="ascii")
    SESSION_SECRET = SECRET_PATH.read_text(encoding="ascii").strip().encode("ascii")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, "
        "password_hash TEXT NOT NULL, "
        "created_at INTEGER NOT NULL)")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if "email" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    return conn


_db().close()   # apply schema upgrades at startup, not just on first login


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)
    return f"pbkdf2_sha256$210000${_b64e(salt)}${_b64e(digest)}"


def verify_password(stored: str, password: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     _b64d(salt), int(iterations))
        return hmac.compare_digest(actual, _b64d(digest))
    except Exception:
        return False


COMMON_PASSWORDS = {
    "password", "password123", "123456789", "qwerty123", "letmein123",
    "admin12345", "welcome123", "iloveyou1", "abc123456", "111111111",
    "password1!", "dipstudio1", "manjunath123",
}


def password_errors(password: str) -> list[str]:
    errors: list[str] = []
    if len(password) < 12:
        errors.append("at least 12 characters")
    if len(password) > 128:
        errors.append("at most 128 characters")
    if not re.search(r"[a-z]", password):
        errors.append("one lowercase letter")
    if not re.search(r"[A-Z]", password):
        errors.append("one uppercase letter")
    if not re.search(r"\d", password):
        errors.append("one number")
    if not re.search(r"[^A-Za-z0-9\s]", password):
        errors.append("one special character (!@#$...)")
    if password.lower() in COMMON_PASSWORDS:
        errors.append("a less predictable phrase")
    return errors


def username_errors(username: str) -> list[str]:
    if not 3 <= len(username) <= 32:
        return ["3 to 32 characters"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        return ["letters, numbers, dot, dash or underscore only"]
    return []


def email_errors(email: str) -> list[str]:
    """Validate an address without sending mail: format only, no deliverability."""
    email = email.strip()
    if not email:
        return ["required"]
    if len(email) > 254:
        return ["at most 254 characters"]
    if email.count("@") != 1:
        return ["exactly one @ symbol"]
    local, domain = email.rsplit("@", 1)
    if not local or len(local) > 64:
        return ["a name before @"]
    if not re.fullmatch(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.\-]+", local):
        return ["valid characters before @"]
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return ["no leading, trailing or consecutive dots before @"]
    labels = domain.split(".")
    if len(labels) < 2 or not re.fullmatch(r"[A-Za-z]{2,}", labels[-1]):
        return ["a domain with a TLD, e.g. example.com"]
    for label in labels:
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?", label):
            return ["a valid domain name"]
    return []


def _sign(data: dict) -> str:
    body = _b64e(json.dumps(data, separators=(",", ":"),
                            sort_keys=True).encode("utf-8"))
    signature = _b64e(hmac.new(SESSION_SECRET, body.encode("ascii"),
                                hashlib.sha256).digest())
    return f"{body}.{signature}"


def _unsign(token: str) -> dict | None:
    try:
        body, signature = token.split(".", 1)
        expected = _b64e(hmac.new(SESSION_SECRET, body.encode("ascii"),
                                  hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(_b64d(body))
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return data
    except Exception:
        return None


def make_session(username: str) -> str:
    return _sign({"u": username, "exp": int(time.time()) + SESSION_TTL})


def read_session(request: Request) -> str | None:
    data = _unsign(request.cookies.get("dip_session", ""))
    return str(data.get("u", "")) if data else None


def account_passport(request: Request) -> dict | None:
    """Return the signed account carried by this browser, if present."""
    data = _unsign(request.cookies.get(ACCOUNT_COOKIE, ""))
    if not data or not data.get("u") or not data.get("p"):
        return None
    return data


def restore_account(passport: dict) -> None:
    """Rebuild a lost /tmp account row from the browser passport."""
    username = str(passport.get("u") or "").strip()
    password_hash = str(passport.get("p") or "")
    if not username or not password_hash:
        return
    with _db() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE username = ?",
                            (username,)).fetchone():
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, passport.get("e"), password_hash,
                 int(passport.get("c") or time.time())))


_LOGIN_ATTEMPTS: dict[str, list[float]] = {}


def _rate_limited(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _LOGIN_ATTEMPTS.get(ip, []) if now - t < 60]
    _LOGIN_ATTEMPTS[ip] = recent
    return len(recent) >= 10


def _record_attempt(ip: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(ip, []).append(time.time())


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
PUBLIC_PATHS = {
    "/", "/login", "/register", "/favicon.ico",
    "/api/login", "/api/register", "/api/health", "/health",
}


@app.middleware("http")
async def auth_guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static/") or path in PUBLIC_PATHS:
        return await call_next(request)
    user = read_session(request)
    if not user:
        if path.startswith("/api/"):
            return JSONResponse(status_code=401,
                                content={"ok": False,
                                         "error": "Please log in to continue."})
        return RedirectResponse(f"/login?next={quote(path)}", status_code=303)
    request.state.user = user
    return await call_next(request)


@app.get("/")
async def landing():
    return FileResponse(STATIC / "landing.html")


@app.get("/login")
@app.get("/register")
async def auth_page():
    return FileResponse(STATIC / "auth.html")


@app.get("/app")
async def workbench():
    return FileResponse(STATIC / "index.html")


@app.get("/api/me")
async def me(request: Request):
    username = getattr(request.state, "user", None)
    email = None
    if username:
        with _db() as conn:
            row = conn.execute("SELECT email FROM users WHERE username = ?",
                               (username,)).fetchone()
        email = row["email"] if row else None
        if email is None:
            passport = account_passport(request)
            if passport and passport.get("u") == username:
                email = passport.get("e")
    return {"ok": True, "username": username, "email": email}


@app.post("/api/register")
async def register(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "Invalid request."})
    username = str(body.get("username") or "").strip()
    email = str(body.get("email") or "").strip()
    password = str(body.get("password") or "")
    problems = username_errors(username)
    if problems:
        return JSONResponse(status_code=400, content={
            "ok": False, "error": "Username: " + ", ".join(problems)})
    problems = email_errors(email)
    if problems:
        return JSONResponse(status_code=400, content={
            "ok": False, "error": "Email: " + ", ".join(problems) + "."})
    problems = password_errors(password)
    if problems:
        return JSONResponse(status_code=400, content={
            "ok": False, "error": "Weak password — needs " + ", ".join(problems) + "."})
    email = email.lower()
    password_hash = hash_password(password)
    passport = account_passport(request)
    if passport and passport.get("u") == username:
        return JSONResponse(status_code=409, content={
            "ok": False, "error": "That username is already taken."})
    with _db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?",
                        (username,)).fetchone():
            return JSONResponse(status_code=409, content={
                "ok": False, "error": "That username is already taken."})
        if conn.execute("SELECT 1 FROM users WHERE email = ? COLLATE NOCASE",
                        (email,)).fetchone():
            return JSONResponse(status_code=409, content={
                "ok": False, "error": "That email address is already registered."})
        created = int(time.time())
        conn.execute("INSERT INTO users (username, email, password_hash, created_at) "
                     "VALUES (?, ?, ?, ?)",
                     (username, email, password_hash, created))
    response = JSONResponse({"ok": True, "username": username,
                             "email": email, "redirect": "/app"})
    response.set_cookie("dip_session", make_session(username),
                        max_age=SESSION_TTL, httponly=True, samesite="lax",
                        path="/", secure=ON_VERCEL)
    response.set_cookie(ACCOUNT_COOKIE, _sign({
        "u": username, "e": email, "p": password_hash, "c": created,
        "exp": int(time.time()) + ACCOUNT_TTL}),
                        max_age=ACCOUNT_TTL, httponly=True, samesite="lax",
                        path="/", secure=ON_VERCEL)
    return response


@app.post("/api/login")
async def login(request: Request):
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        return JSONResponse(status_code=429, content={
            "ok": False,
            "error": "Too many attempts. Please try again in a minute."})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "Invalid request."})
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")
    with _db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?",
                           (username,)).fetchone()
    if row is None or not verify_password(row["password_hash"], password):
        # The /tmp database may have been discarded on a new instance.
        # A browser that registered here still carries a signed passport;
        # use it to rebuild the account instead of rejecting the login.
        passport = account_passport(request)
        if (passport and passport.get("u") == username and
                verify_password(str(passport.get("p") or ""), password)):
            restore_account(passport)
        else:
            _record_attempt(ip)
            return JSONResponse(status_code=401, content={
                "ok": False, "error": "Incorrect username or password."})
    _LOGIN_ATTEMPTS.pop(ip, None)
    response = JSONResponse({"ok": True, "username": username,
                             "redirect": "/app"})
    response.set_cookie("dip_session", make_session(username),
                        max_age=SESSION_TTL, httponly=True, samesite="lax",
                        path="/", secure=ON_VERCEL)
    if not request.cookies.get(ACCOUNT_COOKIE):
        # Accounts created before this fallback had no passport.  Seed one
        # from the current database row on their next successful login.
        with _db() as conn:
            existing = conn.execute(
                "SELECT password_hash, email, created_at FROM users WHERE username = ?",
                (username,)).fetchone()
        if existing:
            response.set_cookie(ACCOUNT_COOKIE, _sign({
                "u": username, "e": existing["email"],
                "p": existing["password_hash"], "c": existing["created_at"],
                "exp": int(time.time()) + ACCOUNT_TTL}),
                                max_age=ACCOUNT_TTL, httponly=True,
                                samesite="lax", path="/", secure=ON_VERCEL)
    return response


@app.post("/api/logout")
async def logout():
    response = JSONResponse({"ok": True, "redirect": "/"})
    response.delete_cookie("dip_session", path="/")
    return response


@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "conditions": CONDITIONS,
        "sample": SAMPLE.exists(),
    }


@app.post("/api/run")
async def run(request: Request):
    started = time.perf_counter()
    try:
        form = await request.form()
        uploads = [v for k, v in form.multi_items() if k == "files"]
        mode = str(form.get("mode") or "simulate").lower()
        params = parse_params(str(form.get("params") or "{}"))
        try:
            labels = json.loads(str(form.get("labels") or "[]"))
        except json.JSONDecodeError:
            labels = []
        if not isinstance(labels, list):
            labels = []

        size = (params["width"], params["height"])
        source_names: list[str] = []
        rgb_map: dict[str, np.ndarray] = {}
        detection: dict | None = None

        # ---- read uploads once, compute a cache key -----------------------
        raw: list[tuple[str, bytes]] = []
        for up in uploads[:4]:
            if not hasattr(up, "read"):
                continue
            data = await up.read()
            name = getattr(up, "filename", None) or "upload"
            if data:
                raw.append((name, data))

        key = hashlib.sha256()
        key.update(json.dumps(params, sort_keys=True, default=str).encode())
        key.update(mode.encode())
        key.update(json.dumps(labels).encode())
        if raw:
            for name, data in raw:
                key.update(name.encode())
                key.update(hashlib.sha256(data).digest())
        else:
            st = SAMPLE.stat() if SAMPLE.exists() else None
            key.update(f"sample:{st.st_size if st else 0}:{st.st_mtime_ns if st else 0}".encode())
        run_id = key.hexdigest()[:24]

        if run_id in _RUNS:
            _RUNS.move_to_end(run_id)
            entry = _RUNS[run_id]["response"]
            cached = dict(entry)
            cached["cached"] = True
            cached["pipeline_ms"] = entry["elapsed_ms"]
            cached["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return cached

        # ---- decode and build the four conditions -------------------------
        if raw:
            images = []
            for name, data in raw:
                try:
                    images.append((name, dc.decode_and_preprocess(data, size)))
                except ValueError as exc:
                    raise ValueError(f"{name}: {exc}") from exc
            if not images:
                raise ValueError("no readable image was uploaded")
            source_names = [n for n, _ in images]

            if mode == "direct":
                assigned: dict[str, np.ndarray] = {}
                for i, (name, rgb) in enumerate(images):
                    label = str(labels[i]).lower() if i < len(labels) else CONDITIONS[i]
                    if label not in CONDITIONS:
                        label = CONDITIONS[i]
                    if label in assigned:
                        raise ValueError(f"two images were assigned to '{label}'")
                    assigned[label] = rgb
                rgb_map = {c: assigned[c] for c in CONDITIONS if c in assigned}
                source_names = [f"{n} ({c})" for n, c in
                                zip(source_names, [c for c in CONDITIONS if c in assigned])]
            else:
                # One uploaded photograph: detect its illumination and build
                # the other three conditions from that detected reference.
                base = images[0][1]
                rgb_map, detection = simulation.conditions_from_photo(
                    base, params["gains"], seed=params["seed"])
                source_names = [f"{images[0][0]} (detected {detection['label']})"]
        else:
            if not SAMPLE.exists():
                raise ValueError("no sample image found; upload an image to begin")
            base = dc.load_and_preprocess(str(SAMPLE), size)
            rgb_map, detection = simulation.conditions_from_photo(
                base, params["gains"], seed=params["seed"])
            source_names = ["sample: forest stream (day.jpg)"]

        results: dict[str, dict] = {}
        thumbs: dict[str, dict] = {}
        overview_data: dict[str, dict] = {}

        for cond in CONDITIONS:
            if cond not in rgb_map:
                continue
            t0 = time.perf_counter()
            d = dc.analyze_condition(
                rgb_map[cond], clip_limit=params["clip_limit"],
                tile=params["tile"], lo=params["stretch_lo"],
                hi=params["stretch_hi"])
            d["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            results[cond] = {
                "metrics": {k: clean_metrics(v) for k, v in d["metrics"].items()},
                "max_diff_cv": int(d["max_diff_cv"]),
                "levels": {
                    "original": int(d["metrics"]["original"]["levels_used"]),
                    "he": int(d["metrics"]["he"]["levels_used"]),
                },
                "tails": {
                    "dark_frac": round(float((d["gray"] <= 40).mean()), 6),
                    "bright_frac": round(float((d["gray"] >= 250).mean()), 6),
                },
                "elapsed_ms": d["elapsed_ms"],
            }
            overview_data[cond] = d
            thumbs[cond] = {
                "gray": thumb_url(d["gray"]),
                "he": thumb_url(d["he"]),
            }

        if not results:
            raise ValueError("nothing to analyze")

        # Render figures eagerly and return them inline.  On Vercel a later
        # /api/figure request can land on a different serverless instance with
        # an empty in-memory run cache; embedding the images in the analysis
        # response makes the figures independent of instance affinity.
        metrics_map = {c: results[c]["metrics"] for c in results}
        figure_specs = {
            "overview:": lambda: plotting.overview_figure(overview_data),
            "metrics:": lambda: plotting.metrics_figure(metrics_map),
            "techniques:": lambda: plotting.techniques_figure(metrics_map),
        }
        for cond in results:
            figure_specs[f"panel:{cond}"] = (
                lambda cond=cond: plotting.condition_panel(cond, overview_data[cond]))
        inline_figures: dict[str, str] = {}
        for key, render in figure_specs.items():
            try:
                inline_figures[key] = render()
            except Exception:  # leave the lazy endpoint as a fallback
                pass

        elapsed = round((time.perf_counter() - started) * 1000, 1)
        response = {
            "ok": True,
            "cached": False,
            "run_id": run_id,
            "conditions": list(results.keys()),
            "thumbs": thumbs,
            "figure_urls": {
                "panel": {c: f"/api/figure/{run_id}/panel/{c}" for c in results},
                "overview": f"/api/figure/{run_id}/overview",
                "metrics": f"/api/figure/{run_id}/metrics",
                "techniques": f"/api/figure/{run_id}/techniques",
            },
            "figures": inline_figures,
            "results": results,
            "params": params,
            "source": {"names": source_names, "mode": mode,
                       "used_sample": not bool(raw)},
            "detection": detection,
            "elapsed_ms": elapsed,
        }
        figure_cache = {}
        for key, data_url in inline_figures.items():
            header, payload = data_url.split(",", 1)
            figure_cache[key] = (base64.b64decode(payload),
                                 header.split(";")[0].split(":")[1])
        _RUNS[run_id] = {
            "response": response,
            "data": overview_data,
            "metrics": metrics_map,
            "figures": figure_cache,
        }
        while len(_RUNS) > _RUNS_MAX:
            _RUNS.popitem(last=False)
        return response
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - unexpected server error
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/figure/{run_id}/{kind}")
@app.get("/api/figure/{run_id}/{kind}/{cond}")
def figure(run_id: str, kind: str, cond: str | None = None):
    """Render (and cache) one figure of a run on first request."""
    art = _RUNS.get(run_id)
    if art is None:
        return JSONResponse(status_code=404,
                            content={"ok": False,
                                     "error": "run expired — please run the analysis again"})
    key = f"{kind}:{cond or ''}"
    cached = art["figures"].get(key)
    if cached is None:
        try:
            if kind == "panel" and cond in art["data"]:
                data_url = plotting.condition_panel(cond, art["data"][cond])
            elif kind == "overview" and cond is None:
                data_url = plotting.overview_figure(art["data"])
            elif kind == "metrics" and cond is None:
                data_url = plotting.metrics_figure(art["metrics"])
            elif kind == "techniques" and cond is None:
                data_url = plotting.techniques_figure(art["metrics"])
            else:
                return JSONResponse(status_code=404,
                                    content={"ok": False, "error": "unknown figure"})
        except Exception as exc:  # pragma: no cover
            return JSONResponse(status_code=500,
                                content={"ok": False, "error": str(exc)})
        header, payload = data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        cached = (base64.b64decode(payload), mime)
        art["figures"][key] = cached
    body, mime = cached
    return Response(body, media_type=mime,
                    headers={"Cache-Control": "public, max-age=3600"})


def _data_b64(data_url: str) -> bytes:
    """Decode a Matplotlib/JPEG data URL into raw bytes."""
    return base64.b64decode(data_url.split(",", 1)[1])


def _metrics_csv(metrics: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["condition", "technique", "metric", "value"])
    for cond, methods in metrics.items():
        for technique, values in methods.items():
            for key in sorted(values):
                writer.writerow([cond, technique, key, values[key]])
    return buf.getvalue()


def _bundle_inputs() -> tuple[dict, dict, dict | None]:
    """Use the latest cached run, or analyse the bundled sample from scratch."""
    if _RUNS:
        entry = next(reversed(_RUNS.values()))
        return entry["data"], entry["metrics"], entry
    base = dc.load_and_preprocess(
        str(SAMPLE), (DEFAULT_PARAMS["width"], DEFAULT_PARAMS["height"]))
    data: dict[str, dict] = {}
    for i, cond in enumerate(CONDITIONS):
        rgb = simulation.simulate_illumination(
            base, cond, DEFAULT_PARAMS["gains"].get(cond),
            seed=DEFAULT_PARAMS["seed"] + i)
        data[cond] = dc.analyze_condition(
            rgb, clip_limit=DEFAULT_PARAMS["clip_limit"],
            tile=DEFAULT_PARAMS["tile"], lo=DEFAULT_PARAMS["stretch_lo"],
            hi=DEFAULT_PARAMS["stretch_hi"])
    return data, {c: data[c]["metrics"] for c in data}, None


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
