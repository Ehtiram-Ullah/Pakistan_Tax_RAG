"""Account-scoped discovery cache and short-lived model failure cooldowns."""
import hashlib
import logging
import os
from threading import RLock
from time import monotonic

MAX_MODELS = 16
CATALOG_TTL = 900
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-flash-latest", "gemini-flash-lite-latest",
]
_lock = RLock()
_catalog = {}
_cooldowns = {}
_winners = {}
logger = logging.getLogger(__name__)


def account_id():
    return hashlib.sha256(os.getenv("GEMINI_API_KEY", "").encode()).hexdigest()


def text_model(name):
    return name.startswith("gemini-") and not any(
        token in name for token in (
            "image", "tts", "audio", "live", "embedding", "robotics",
            "computer-use", "transcribe", "omni", "nano-banana", "customtools",
        )
    )


def discover_models(client):
    """Fetch at most one 100-model catalog page, once per 15 minutes."""
    key = account_id()
    with _lock:
        cached = _catalog.get(key)
        if cached and cached[0] > monotonic():
            return cached[1], True, cached[2]
        try:
            page = client.models.list(config={
                "page_size": 100,
                "http_options": {"timeout": 10000},
            }).page
            names = []
            for model in page:
                name = (model.name or "").removeprefix("models/")
                if text_model(name) and "generateContent" in (model.supported_actions or []):
                    names.append(name)
            # Prefer low-latency Flash models; leave Pro as a later fallback.
            names = sorted(set(names), key=lambda name: ("flash" not in name, "preview" in name, name))
            _catalog[key] = (monotonic() + CATALOG_TTL, names, True)
            return names, False, True
        except Exception as error:
            logger.warning("Model discovery failed: %s", type(error).__name__)
            names = cached[1] if cached else FALLBACK_MODELS
            _catalog[key] = (monotonic() + 60, names, False)
            return names, False, False


def model_plan(client, preferred):
    discovered, cached, discovery_ok = discover_models(client)
    names = list(dict.fromkeys(preferred + discovered))
    key = account_id()
    with _lock:
        winner = _winners.get(key)
        if winner in names:
            names.remove(winner)
            names.insert(0, winner)
        ready, skipped = [], []
        for name in names:
            cooldown = _cooldowns.get((key, name))
            if cooldown and cooldown[0] > monotonic():
                skipped.append({"type": "model_skipped", "model": name, "reason": cooldown[1] + " · temporary cooldown"})
            else:
                ready.append(name)
        return ready[:MAX_MODELS], skipped, cached, discovery_ok


def record_failure(model, error, reason):
    code = str(getattr(error, "code", ""))
    seconds = {"404": 900, "429": 60, "500": 30, "502": 30, "503": 30, "504": 30}.get(code, 0)
    if seconds:
        with _lock:
            _cooldowns[(account_id(), model)] = (monotonic() + seconds, reason)


def record_success(model):
    with _lock:
        _winners[account_id()] = model
        _cooldowns.pop((account_id(), model), None)
