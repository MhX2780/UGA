"""
General Agent configuration.
"""
import os
import sys
from pathlib import Path


def _get_persistent_base_dir() -> Path:
    """
    Returns a directory to store persistent app data (API key, memory,
    execution log, usage stats) that works correctly whether running as a
    normal Python script OR as a PyInstaller-frozen executable.

    Why not just Path(__file__).parent? Under PyInstaller — especially
    --onefile mode — the running program is unpacked into a TEMPORARY
    directory each time it starts (sys._MEIPASS), which is deleted again on
    exit. Using that location for persistent data (the saved API key,
    conversation memory, etc.) would silently lose everything between runs,
    which defeats the entire point of this app's persistent-memory feature.

    Instead: if AGENT_DATA_DIR is set, honor it explicitly. Otherwise, use a
    dedicated folder next to the actual executable/script (sys.executable
    when frozen, since that path IS persistent — it's the .exe/binary
    itself, not a temp extraction — or the script's own directory
    otherwise). This keeps the "everything lives next to the program"
    simplicity of the original design while being safe under PyInstaller.
    """
    env_override = os.environ.get("AGENT_DATA_DIR")
    if env_override:
        return Path(env_override)

    if getattr(sys, "frozen", False):
        # Running as a PyInstaller-frozen executable: sys.executable points
        # at the actual persistent .exe/binary location (not the temporary
        # extraction dir), so store data next to that.
        return Path(sys.executable).parent

    # Normal `python cli.py` execution — behave as before, next to the source.
    return Path(__file__).parent


# ---------- Base paths ----------
BASE_DIR = _get_persistent_base_dir()
MEMORY_DIR = BASE_DIR / "memory_store"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Local file used to persist the API key (instead of environment variables)
API_KEY_FILE = BASE_DIR / ".gemini_api_key"


def load_saved_api_key() -> str:
    """Reads the saved key from the local file if it exists, else returns ''."""
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_api_key(key: str):
    """Saves the key to a local file with owner-only read/write permissions."""
    API_KEY_FILE.write_text(key.strip(), encoding="utf-8")
    try:
        os.chmod(API_KEY_FILE, 0o600)  # restrict access to the current OS user only
    except OSError:
        pass  # some systems (e.g. Windows) don't support chmod the same way; safe to ignore


GEMINI_API_KEY = load_saved_api_key()

# ---------- User-editable settings (persisted, overrides the defaults below) ----------
SETTINGS_FILE = BASE_DIR / "settings.json"


def load_settings() -> dict:
    """
    Loads user-editable settings from settings.json (created/edited by
    /settings in the CLI), falling back to {} if it doesn't exist yet or is
    corrupted. Settings here override the hardcoded defaults further below
    in this file — e.g. a saved "model_chain" list replaces MODEL_CHAIN, a
    saved "multi_agent" dict overrides MULTI_AGENT_ROLES/MULTI_AGENT_ENABLED.
    This lets /settings change behavior without editing this file directly,
    and survives restarts.
    """
    if SETTINGS_FILE.exists():
        try:
            import json
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_settings(settings: dict):
    """Persists the given settings dict to settings.json."""
    import json
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


_saved_settings = load_settings()

# ---------- Memory paths ----------
LONG_TERM_MEMORY_FILE = MEMORY_DIR / "long_term_memory.jsonl"   # persistent facts/preferences
SESSION_LOG_FILE = MEMORY_DIR / "session_log.jsonl"             # raw log of all conversations
STATS_FILE = MEMORY_DIR / "usage_stats.json"                    # per-model usage statistics
EXECUTION_LOG_FILE = MEMORY_DIR / "execution_log.jsonl"         # tool-call action/result history

# Max number of recent execution-log entries injected into the system prompt
# as context. Keeps a new/switched-to model aware of what was just done
# without re-sending the entire raw tool-call history (which can be large
# and noisy — e.g. full file contents in a read_file result).
EXECUTION_LOG_CONTEXT_ENTRIES = 15

# ---------- Model chain (priority order for automatic switching) ----------
# The first entry is preferred; on failure or quota exhaustion the router
# moves to the next one.
#
# IMPORTANT — verified July 2026: as of April 1, 2026, Google removed ALL
# Pro-tier models (gemini-2.5-pro, gemini-3-pro, gemini-3.1-pro) from the free
# tier — they are now paid-only. Only Flash and Flash-Lite models retain a
# free tier (with reduced daily quotas). Pro models are kept in the chain
# below as fallbacks for paid accounts, but a free-tier account will simply
# have them skipped automatically (via the zero-quota detection in
# model_router.py) rather than ever succeeding on them.
#
# NOTE: this list now includes every model currently listed under Google AI
# Studio's free tier (see gemini_models_list.txt), not just general-purpose
# text/chat models. That means it also includes image-generation models
# (e.g. "Nano Banana" variants), TTS models, robotics/computer-use models,
# and Deep Research / Antigravity agent previews. These do NOT behave like
# plain chat models — e.g. TTS models expect/return audio, image models
# return images, etc. — so if the agent sends a normal text prompt to one of
# these and gets back something it can't use, the router should treat that
# as a failure and move on to the next entry, same as a quota error. Flash /
# Flash-Lite text models are kept first in priority order for that reason.
#
# This list is also just the DEFAULT — it's fully user-editable at runtime
# via /settings in the CLI, since which models exist and which are free
# changes often enough that hardcoding a single "correct" list isn't
# reliable. /settings lets you add/remove/reorder any model name the Gemini
# API accepts, and assign specific models to specific ROLES (see
# MULTI_AGENT_ROLES below) for the multi-agent feature.
_DEFAULT_MODEL_CHAIN = [
    # --- General-purpose text/chat models (preferred first) ---
    {"name": "gemini-3.6-flash", "max_requests_per_session": 200},
    {"name": "gemini-3.5-flash", "max_requests_per_session": 200},
    {"name": "gemini-flash-latest", "max_requests_per_session": 200},
    {"name": "gemini-3-flash-preview", "max_requests_per_session": 200},
    {"name": "gemini-2.5-flash", "max_requests_per_session": 200},
    {"name": "gemini-2.0-flash", "max_requests_per_session": 200},
    {"name": "gemini-2.0-flash-001", "max_requests_per_session": 200},
    {"name": "gemini-omni-flash-preview", "max_requests_per_session": 200},
    {"name": "gemini-3.5-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-3.1-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-3.1-flash-lite-preview", "max_requests_per_session": 300},
    {"name": "gemini-flash-lite-latest", "max_requests_per_session": 300},
    {"name": "gemini-2.5-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-2.0-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-2.0-flash-lite-001", "max_requests_per_session": 300},
    {"name": "gemma-4-26b-a4b-it", "max_requests_per_session": 300},
    {"name": "gemma-4-31b-it", "max_requests_per_session": 300},

    # --- Pro-tier text models (paid-only since Apr 1, 2026 — skipped
    # automatically on free-tier accounts via zero-quota detection) ---
    {"name": "gemini-pro-latest", "max_requests_per_session": 100},
    {"name": "gemini-2.5-pro", "max_requests_per_session": 100},
    {"name": "gemini-3-pro-preview", "max_requests_per_session": 100},
    {"name": "gemini-3.1-pro-preview", "max_requests_per_session": 100},
    {"name": "gemini-3.1-pro-preview-customtools", "max_requests_per_session": 100},

    # --- Image generation models ("Nano Banana" family) — return images,
    # not text; only useful if the agent has an image-handling code path ---
    {"name": "gemini-2.5-flash-image", "max_requests_per_session": 100},
    {"name": "gemini-3-pro-image-preview", "max_requests_per_session": 100},
    {"name": "gemini-3-pro-image", "max_requests_per_session": 100},
    {"name": "nano-banana-pro-preview", "max_requests_per_session": 100},
    {"name": "gemini-3.1-flash-image-preview", "max_requests_per_session": 100},
    {"name": "gemini-3.1-flash-image", "max_requests_per_session": 100},
    {"name": "gemini-3.1-flash-lite-image", "max_requests_per_session": 100},

    # --- TTS / audio models — return audio, not text ---
    {"name": "gemini-2.5-flash-preview-tts", "max_requests_per_session": 100},
    {"name": "gemini-2.5-pro-preview-tts", "max_requests_per_session": 100},
    {"name": "gemini-3.1-flash-tts-preview", "max_requests_per_session": 100},

    # --- Music generation models (Lyria) ---
    {"name": "lyria-3-clip-preview", "max_requests_per_session": 100},
    {"name": "lyria-3-pro-preview", "max_requests_per_session": 100},

    # --- Robotics / computer-use / agentic preview models ---
    {"name": "gemini-robotics-er-1.5-preview", "max_requests_per_session": 100},
    {"name": "gemini-robotics-er-1.6-preview", "max_requests_per_session": 100},
    {"name": "gemini-robotics-er-2-preview", "max_requests_per_session": 100},
    {"name": "gemini-2.5-computer-use-preview-10-2025", "max_requests_per_session": 100},
    {"name": "antigravity-preview-05-2026", "max_requests_per_session": 100},

    # --- Deep Research previews ---
    {"name": "deep-research-max-preview-04-2026", "max_requests_per_session": 50},
    {"name": "deep-research-preview-04-2026", "max_requests_per_session": 50},
    {"name": "deep-research-pro-preview-12-2025", "max_requests_per_session": 50},
]

# The actual chain used at runtime: the saved override from /settings if one
# exists, otherwise the built-in default above.
MODEL_CHAIN = _saved_settings.get("model_chain", _DEFAULT_MODEL_CHAIN)

_DEFAULT_MULTI_AGENT_ROLES = {
    "classifier": "gemini-2.5-flash-lite",  # fast/cheap: decides simple-vs-complex
    "planner": "gemini-3.6-flash",          # breaks a complex task into steps
    "executor": "gemini-3.5-flash",         # runs tools, writes code
    "reviewer": "gemini-2.5-flash-lite",    # checks the final result
}

MULTI_AGENT_ROLES = _saved_settings.get("multi_agent_roles", _DEFAULT_MULTI_AGENT_ROLES)
MULTI_AGENT_ENABLED = _saved_settings.get("multi_agent_enabled", False)


def get_current_settings_snapshot() -> dict:
    """
    Returns the full current settings as a plain dict — used by /settings in
    the CLI to display current values and by save_settings() to persist
    changes. Always includes every known setting key (falling back to
    defaults for anything not yet customized), so the saved file is
    self-documenting rather than a sparse diff.
    """
    return {
        "model_chain": MODEL_CHAIN,
        "multi_agent_roles": MULTI_AGENT_ROLES,
        "multi_agent_enabled": MULTI_AGENT_ENABLED,
    }

# Number of retry attempts on the same model before moving to the next one
RETRIES_PER_MODEL = 2

# Base wait time in seconds between retries (exponential backoff base)
RETRY_BACKOFF_BASE = 2

# Max number of messages kept as in-memory "context" from short-term memory
MAX_HISTORY_MESSAGES = 30

# Workspace directory — the only place the Agent is allowed to create/edit files (security)
WORKSPACE_DIR = Path(os.environ.get("AGENT_WORKSPACE", str(Path.cwd() / "workspace")))
WORKSPACE_DIR.mkdir(exist_ok=True)
