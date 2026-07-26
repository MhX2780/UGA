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
# Verified against Google's official Gemini API pricing page
# (https://ai.google.dev/gemini-api/docs/pricing) as of July 2026:
#   - gemini-2.0-flash and gemini-2.0-flash-lite were shut down June 1, 2026
#     and now return errors for all requests — excluded here.
#   - "gemini-3.5-flash-lite" does not exist as a model (there is no
#     Flash-Lite variant in the 3.5 generation yet) — removed; the closest
#     equivalent is gemini-3.1-flash-lite.
#   - gemini-3.1-pro-preview has NO free tier at all (paid-only per the
#     pricing page) — placed last, only useful once billing is enabled.
#   - Every other model below is confirmed "Free of charge" for standard
#     input/output on the free tier as of this writing. Free-tier quotas can
#     still change without notice, which is exactly what the automatic
#     model-switching in this agent is designed to handle gracefully.
MODEL_CHAIN = [
    {
        "name": "gemini-3.5-flash",
        "max_requests_per_session": 200,
    },
    {
        "name": "gemini-3-flash-preview",
        "max_requests_per_session": 200,
    },
    {
        "name": "gemini-3.1-flash-lite",
        "max_requests_per_session": 300,
    },
    {
        "name": "gemini-2.5-flash",
        "max_requests_per_session": 200,
    },
    {
        "name": "gemini-2.5-flash-lite",
        "max_requests_per_session": 300,
    },
    {
        "name": "gemini-2.5-pro",
        "max_requests_per_session": 100,
    },
    {
        # Paid-only (no free tier) — last resort, only works if billing is
        # enabled on the account. Kept as the final fallback for best quality
        # on paid accounts rather than failing outright once free options
        # are exhausted.
        "name": "gemini-3.1-pro-preview",
        "max_requests_per_session": None,
    },
]

# Number of retry attempts on the same model before moving to the next one
RETRIES_PER_MODEL = 2

# Base wait time in seconds between retries (exponential backoff base)
RETRY_BACKOFF_BASE = 2

# Max number of messages kept as in-memory "context" from short-term memory
MAX_HISTORY_MESSAGES = 30

# Workspace directory — the only place the Agent is allowed to create/edit files (security)
WORKSPACE_DIR = Path(os.environ.get("AGENT_WORKSPACE", str(Path.cwd() / "workspace")))
WORKSPACE_DIR.mkdir(exist_ok=True)
