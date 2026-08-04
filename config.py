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

# ---------- Puter.js auth token (optional — see providers.py) ----------
# Puter.js gives free access to Claude/GPT/Gemini/DeepSeek/etc. under its
# "User-Pays" model: each user authenticates their OWN Puter account, and
# usage is billed to them, not the app developer. Puter.js itself is a
# browser-only SDK, but Puter documents an "auth token" (from
# puter.com/dashboard#account -> Create token) that works directly against
# their OpenAI-compatible REST endpoint from any environment — including
# plain Python, no browser needed. That's what's stored here.
PUTER_TOKEN_FILE = BASE_DIR / ".puter_token"


def load_puter_token() -> str:
    """Reads the saved Puter auth token from the local file if it exists,
    else returns ''."""
    if PUTER_TOKEN_FILE.exists():
        return PUTER_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_puter_token(token: str):
    """Saves the Puter auth token to a local file with owner-only
    read/write permissions, same as the Gemini key."""
    PUTER_TOKEN_FILE.write_text(token.strip(), encoding="utf-8")
    try:
        os.chmod(PUTER_TOKEN_FILE, 0o600)
    except OSError:
        pass


def delete_puter_token():
    if PUTER_TOKEN_FILE.exists():
        PUTER_TOKEN_FILE.unlink()

# ---------- Multiple Gemini API keys (Google AI Studio) ----------
# Beyond the single "primary" key above (kept for backward compatibility —
# older saved installs and simple single-key setups keep working exactly as
# before), the agent can hold a POOL of additional Gemini API keys and
# rotate to the next one specifically when a key hits its DAILY quota
# (RPD) — which, unlike a per-minute rate limit, waiting out is never
# useful for (see model_router.py's RPD/RPM distinction). This is the
# actual fix for "the whole agent goes idle until tomorrow" once every
# model in MODEL_CHAIN has also exhausted its daily quota on the current
# key: instead of stopping there, it moves to the next configured key and
# retries the SAME model chain from the top on that key.
API_KEYS_FILE = BASE_DIR / ".gemini_api_keys.jsonl"


def load_api_key_pool() -> list:
    """
    Returns the full list of configured Gemini API keys, in priority order
    (the primary key from .gemini_api_key first if set, then any additional
    keys from .gemini_api_keys.jsonl, keys deduplicated but order-preserved).
    """
    keys = []
    primary = load_saved_api_key()
    if primary:
        keys.append(primary)
    if API_KEYS_FILE.exists():
        for line in API_KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and line not in keys:
                keys.append(line)
    return keys


def add_api_key_to_pool(key: str):
    """
    Adds an additional Gemini API key to the pool. If no primary key is set
    yet (.gemini_api_key doesn't exist), this becomes the primary key
    instead of going into the pool file — keeping the common single-key case
    simple (nothing in .gemini_api_keys.jsonl at all until a SECOND key is
    actually added).
    """
    key = key.strip()
    if not key:
        return
    if not load_saved_api_key():
        save_api_key(key)
        return
    existing = load_api_key_pool()
    if key in existing:
        return
    with open(API_KEYS_FILE, "a", encoding="utf-8") as f:
        f.write(key + "\n")
    try:
        os.chmod(API_KEYS_FILE, 0o600)
    except OSError:
        pass


def remove_api_key_from_pool(key_suffix: str) -> bool:
    """
    Removes a key from the pool by matching its last N characters (so the
    user can identify a specific key without needing to paste the whole
    secret back, e.g. remove_api_key_from_pool("Xy12") to remove whichever
    key ends in "Xy12"). Returns True if a key was found and removed, False
    otherwise. Cannot remove the primary key this way (use /resetkey for
    that) — only additional pool keys.
    """
    if not API_KEYS_FILE.exists():
        return False
    lines = [l.strip() for l in API_KEYS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    matching = [l for l in lines if l.endswith(key_suffix)]
    if not matching:
        return False
    remaining = [l for l in lines if l not in matching]
    API_KEYS_FILE.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
    return True


def mask_api_key(key: str) -> str:
    """Returns a masked display form of a key, e.g. 'AIza...Xy12', for
    showing in lists without printing the full secret to the terminal."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"

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
# IMPORTANT DESIGN NOTE: this chain is TEXT/CHAT MODELS ONLY. The router
# (model_router.py) sends a plain text prompt and reads response.text — it
# has no code path for images, audio, or action/robotics outputs. Mixing in
# non-text models (image generation, TTS, music, robotics/computer-use,
# Deep Research agent previews) here would not "gracefully fail over" the
# way a quota error does — it would return a response with no usable .text,
# which the rest of the app isn't built to handle. Those model families are
# kept below in separate, clearly-labeled lists (IMAGE_MODEL_CHAIN, and
# reference-only lists for the others) instead, so they're documented and
# available to route to specifically (e.g. Image_Create in tools.py already
# uses IMAGE_MODEL_CHAIN's first entry) without corrupting the main
# conversational failover chain.
#
# Verified against Google's official Gemini API changelog/model docs and
# Firebase AI Logic model-availability notices, as of end of July 2026:
#   - Gemini 3.6 Flash and Gemini 3.5 Flash-Lite reached GENERAL AVAILABILITY
#     on July 21, 2026 — these are now the recommended stable defaults.
#   - gemini-2.0-flash, gemini-2.0-flash-001, gemini-2.0-flash-lite, and
#     gemini-2.0-flash-lite-001 were ALL RETIRED June 1, 2026 and return
#     errors for every request — removed from the chain entirely (not just
#     commented out) since there's no reason to ever attempt them.
#   - gemini-3-pro-preview was retired March 9, 2026 in favor of
#     gemini-3.1-pro-preview.
#   - As of April 1, 2026, ALL Pro-tier text models are paid-only (no free
#     tier). Kept as fallbacks for paid accounts; a free-tier account will
#     simply have them skipped automatically via the zero-quota detection
#     in model_router.py.
#   - Heads-up for later maintenance: Google has announced Gemini 2.5 Pro and
#     2.5 Flash will be retired in October 2026 — if you're reading this
#     after that date, those two entries likely need removing too.
#   - "gemini-flash-latest" / "gemini-flash-lite-latest" / "gemini-pro-latest"
#     are Google's "floating" aliases that always point at the current
#     recommended release — kept as safety-net fallback entries so the chain
#     stays useful even if a pinned version name below gets retired before
#     this file is updated again.
#   - gemma-* entries are a separate open-weight model family (not part of
#     the Gemini line) with historically generous but sometimes volatile
#     free-tier quotas — kept as a last-resort fallback before paid Pro
#     models, not as a primary choice.
#
# This list is also just the DEFAULT — it's fully user-editable at runtime
# via /settings in the CLI (including /settings models, which queries your
# actual API key for the live list Google currently serves it), since which
# models exist and which are free changes often enough that a hardcoded
# "correct" list alone isn't reliable long-term.
_DEFAULT_MODEL_CHAIN = [
    {"name": "gemini-3.6-flash", "max_requests_per_session": 200},
    {"name": "gemini-3.5-flash", "max_requests_per_session": 200},
    {"name": "gemini-3.5-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-3.1-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-2.5-flash", "max_requests_per_session": 200},
    {"name": "gemini-2.5-flash-lite", "max_requests_per_session": 300},
    {"name": "gemini-flash-latest", "max_requests_per_session": 200},
    {"name": "gemini-flash-lite-latest", "max_requests_per_session": 300},
    {"name": "gemma-4-31b-it", "max_requests_per_session": 300},
    {"name": "gemma-4-26b-a4b-it", "max_requests_per_session": 300},
    {
        # Paid-only since April 1, 2026 — kept as a fallback for paid
        # accounts; skipped automatically on free-tier accounts.
        "name": "gemini-2.5-pro",
        "max_requests_per_session": 100,
    },
    {"name": "gemini-pro-latest", "max_requests_per_session": 100},
    {
        # Paid-only, preview.
        "name": "gemini-3.1-pro-preview",
        "max_requests_per_session": None,
    },
]

# ---------- Image generation models ("Nano Banana" family) ----------
# These return IMAGE data, not text — used by tools.Image_Create, never by
# the main conversational MODEL_CHAIN above. Kept as its own ordered
# fallback list for the same "try the next one on failure" reasoning.
IMAGE_MODEL_CHAIN = [
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "nano-banana-pro-preview",
    # Preview-stage siblings of the stable entries above — kept last since
    # "-preview" model IDs are less stable/guaranteed than GA releases.
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
]

# ---------- Reference-only: other non-text model families ----------
# These are NOT wired into any automatic chain in this codebase — there's no
# tool here that produces/consumes audio, music, robot action sequences, or
# uses the standalone Deep Research/Antigravity agent frameworks. Listed
# here purely as an up-to-date reference in case a future tool needs them,
# so their exact current model-ID strings are documented in one place
# rather than needing to be re-discovered.
_REFERENCE_TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
]
# Newer preview-stage text/chat model IDs, not yet promoted into
# _DEFAULT_MODEL_CHAIN above (kept as reference until they reach GA, at
# which point they'd move up into the main chain like 3.5/3.6-flash did).
_REFERENCE_PREVIEW_TEXT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview-customtools",  # gemini-3.1-pro-preview tuned for custom tool use
    "gemini-omni-flash-preview",
]
_REFERENCE_MUSIC_MODELS = [
    "lyria-3-clip-preview",
    "lyria-3-pro-preview",
]
_REFERENCE_ROBOTICS_COMPUTER_USE_MODELS = [
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-2-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "antigravity-preview-05-2026",
]
_REFERENCE_DEEP_RESEARCH_MODELS = [
    "deep-research-max-preview-04-2026",
    "deep-research-preview-04-2026",
    "deep-research-pro-preview-12-2025",
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

# ---------- Puter.js chat integration ----------
# If ON, Puter.js models are allowed to be used for plain-text replies in the
# main chat, not just via /puterJS or /free for manual model picking.
# IMPORTANT: tool calling (file edits, running commands, etc.) is NOT wired
# up for Puter — the whole tools.ALL_TOOLS pipeline in agent.py only talks to
# Gemini's function-calling API. When a Puter model answers in chat it does
# so WITHOUT access to any tools. Tool support for Puter may be added later.
PUTER_CHAT_ENABLED = _saved_settings.get("puter_chat_enabled", False)

# If ON, any call routed to the "puter" provider is restricted to models whose
# id ends in "free" (or contains "free"), regardless of which Puter model was
# otherwise selected. This keeps usage on Puter's no-cost tier only.
PUTER_FREE_ONLY = _saved_settings.get("puter_free_only", False)

# ---------- Puter.js tool calling (BETA) ----------
# If ON, chat turns routed through Puter (i.e. PUTER_CHAT_ENABLED=True) are
# given the full tools.ALL_TOOLS toolbox, translated to the OpenAI-style
# `tools=[{"type": "function", "function": {...}}]` schema, and the agent
# will execute any tool_calls the model returns — the same file/git/command
# tools available to Gemini, now also reachable via Puter models.
#
# BETA — off by default. Two real sources of risk, both confirmed against
# Puter's own docs before this was added:
#   1) Puter's REST endpoint used here (api.puter.com/puterai/openai/v1/,
#      hit via the standard `openai` SDK) is a different code path than the
#      puter.ai.chat() JS client shown in Puter's official tool-calling
#      examples. The *documented* request/response shape (tools param,
#      tool_calls in the response, role:"tool" reply format) matches OpenAI's
#      own convention closely, and Puter's REST endpoint is advertised as
#      OpenAI-compatible — but this hasn't been exercised against every
#      model Puter re-exposes, only spot-checked.
#   2) Not every model available through Puter actually supports function
#      calling well even when the request is formatted correctly — smaller
#      or free-tier models in particular may ignore the tools list, or
#      "hallucinate" a tool call that doesn't match any real tool. Gemini
#      models remain the well-tested, fully-supported path for tool-calling;
#      this setting extends that capability to Puter models on a best-effort
#      basis, not a guaranteed one.
#
# Toggle with /settings puter tools on|off in the CLI.
PUTER_TOOL_CALLING_ENABLED = _saved_settings.get("puter_tool_calling_enabled", False)


# ---------- Puter.js image tools (BETA, opt-in fallback) ----------
# If ON, when Image_Fetch or Image_Create fails via Gemini, the tool's
# error message tells the model that a Puter.js fallback is available (but
# does NOT call it automatically) — the model is expected to ask the user
# for permission first, then call Image_Fetch_Puter / Image_Create_Puter
# explicitly once the user agrees. See tools.py's docstrings on those
# functions for the full rationale.
#
# Off by default, and only offered as a fallback at all when a Puter token
# is actually configured (config.load_puter_token()) — there's no point
# suggesting a fallback the user hasn't set up.
#
# BETA scope notes, confirmed against Puter's own documentation:
#   - Image_Fetch_Puter (vision/image-understanding) uses Puter's
#     documented, REST-compatible chat.completions endpoint with an image
#     content part — the same code path already verified for text and
#     tool-calling in this app. Reasonably safe to rely on.
#   - Image_Create_Puter (image GENERATION) is a different story: Puter's
#     own documented image-generation function, puter.ai.txt2img(), is
#     ONLY demonstrated in Puter's docs via the browser JavaScript SDK
#     (HTMLImageElement return value, puter.print, etc.) — no official
#     example anywhere shows an image-generation REST endpoint analogous
#     to OpenAI's /v1/images/generations reachable from Python/the openai
#     SDK. Image_Create_Puter below attempts the OpenAI SDK's standard
#     images.generate() call against Puter's base_url as a best-effort,
#     since Puter advertises OpenAI-compatibility broadly — but this
#     specific capability has NOT been confirmed to exist server-side by
#     any documentation found. It may simply 404. Treat this one as
#     genuinely experimental, more so than every other Puter integration
#     in this file.
PUTER_IMAGE_TOOLS_ENABLED = _saved_settings.get("puter_image_tools_enabled", False)

# Puter model used for Image_Fetch_Puter (vision-capable). Kept separate
# from the text MODEL_CHAIN/MULTI_AGENT_ROLES since it must specifically
# support image inputs — not every model Puter re-exposes does.
PUTER_VISION_MODEL = _saved_settings.get("puter_vision_model", "infron:deepseek/deepseek-v4-flash:free")

# Puter model used for Image_Create_Puter (image generation) — see the
# BETA note above; this path is unverified server-side.
PUTER_IMAGE_GEN_MODEL = _saved_settings.get("puter_image_gen_model", "infron:deepseek/deepseek-v4-flash:free")

# Default free Puter model used by /free-puter-models-only when auto-selecting.
# Should be a model with "free" in its id (Puter's current naming convention
# for no-cost tier models, e.g. "infron:deepseek/deepseek-v4-flash:free").
PUTER_FREE_CHAT_MODEL = _saved_settings.get("puter_free_chat_model", "infron:deepseek/deepseek-v4-flash:free")


# ---------------- Deep Thinking (Gemini) ----------------
# If ON, requests to Gemini models go out with an explicit thinking
# configuration attached (google.genai.types.ThinkingConfig), asking the
# model to spend extra internal reasoning tokens before answering. Only
# Gemini's "thinking" family (2.5/3.x line) actually honors this — passing
# it to a model that doesn't support thinking is harmless (the SDK/backend
# ignores unknown-to-it config), but it's still gated behind this toggle so
# it's not silently on by default (thinking tokens cost more and are slower).
#
# thinking_budget: -1 lets the model decide how much to think (Google's
# "dynamic thinking"); a positive integer caps the thinking-token budget;
# 0 disables thinking outright. include_thoughts: if True, the model's
# thought summaries are included in the response (surfaced to the user as
# a distinct "thinking" section) rather than only used internally.
DEEP_THINKING_ENABLED = _saved_settings.get("deep_thinking_enabled", False)
DEEP_THINKING_BUDGET = _saved_settings.get("deep_thinking_budget", -1)  # -1 = dynamic/auto
DEEP_THINKING_INCLUDE_THOUGHTS = _saved_settings.get("deep_thinking_include_thoughts", True)

# ---------------- Deep Thinking (Puter.js) ----------------
# Puter's OpenAI-compatible endpoint re-exposes reasoning-capable models
# (e.g. o1/o3, deepseek-reasoner, and Claude's extended-thinking models)
# which accept an OpenAI-style `reasoning_effort` field ("low"/"medium"/
# "high") to control how much hidden reasoning the model performs before
# answering. This is a SEPARATE toggle from the Gemini one above since it's
# a different provider/parameter shape — not every Puter model supports
# reasoning_effort, but sending it to one that doesn't is generally ignored
# rather than an error, so this is left to the caller/model rather than
# hard-validated here.
PUTER_DEEP_THINKING_ENABLED = _saved_settings.get("puter_deep_thinking_enabled", False)
PUTER_DEEP_THINKING_EFFORT = _saved_settings.get("puter_deep_thinking_effort", "high")  # low|medium|high

# ---------------- System access (windows / processes) ----------------
# Gates two sensitive tools: Available_Active_Windows (lists the user's
# currently open windows and can screenshot each one) and
# List_System_Processes (task-manager-style listing of ALL system
# processes with PID/CPU/memory, not just ones this agent started).
#
# Off by default. When off, both tools return an error telling the model to
# ask the user, in English, for explicit permission before turning this on
# (e.g. "Do you allow the AI to access your open windows and system
# processes?") — see the docstrings on those two tools in tools.py, and the
# matching instruction in agent.py's system prompt. The user grants access
# themselves via /settings system access on; the model cannot flip this
# setting on its own.
SYSTEM_ACCESS_ENABLED = _saved_settings.get("system_access_enabled", False)

# ---------------- Deep Research (Gemini / Google AI Studio) ----------------
# Deep Research is a SEPARATE Gemini model family (see
# _REFERENCE_DEEP_RESEARCH_MODELS above) — it is not a flag on the normal
# chat models, it's its own model you call directly, purpose-built for
# multi-step autonomous web research culminating in a long-form report.
# It is NOT part of MODEL_CHAIN (same reasoning as IMAGE_MODEL_CHAIN: a
# fundamentally different call shape that the normal failover loop isn't
# built to handle) — instead it's invoked explicitly via the /deepresearch
# command (see cli.py) through model_router.run_deep_research().
DEEP_RESEARCH_MODEL = _saved_settings.get("deep_research_model", "deep-research-pro-preview-12-2025")


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
        "puter_chat_enabled": PUTER_CHAT_ENABLED,
        "puter_free_only": PUTER_FREE_ONLY,
        "puter_tool_calling_enabled": PUTER_TOOL_CALLING_ENABLED,
        "puter_image_tools_enabled": PUTER_IMAGE_TOOLS_ENABLED,
        "puter_vision_model": PUTER_VISION_MODEL,
        "puter_image_gen_model": PUTER_IMAGE_GEN_MODEL,
        "puter_free_chat_model": PUTER_FREE_CHAT_MODEL,
        "deep_thinking_enabled": DEEP_THINKING_ENABLED,
        "deep_thinking_budget": DEEP_THINKING_BUDGET,
        "deep_thinking_include_thoughts": DEEP_THINKING_INCLUDE_THOUGHTS,
        "puter_deep_thinking_enabled": PUTER_DEEP_THINKING_ENABLED,
        "puter_deep_thinking_effort": PUTER_DEEP_THINKING_EFFORT,
        "deep_research_model": DEEP_RESEARCH_MODEL,
        "system_access_enabled": SYSTEM_ACCESS_ENABLED,
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
