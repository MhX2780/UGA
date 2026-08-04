#!/usr/bin/env python3
"""
Command-line interface (CLI) for the Agent — similar to Gemini CLI.

Usage:
    python3 cli.py

The first time you run it, it will ask for your GEMINI_API_KEY and save it
locally (see config.API_KEY_FILE) so you won't be asked again.

Special in-session commands:
    /help              show help
    /clear             clear the screen
    /remember <k>=<v>  save a persistent fact to memory
    /memory            show all saved long-term memory
    /forget <key>      delete a key from long-term memory
    /undo              revert the last file change the Agent made
    /tree              show the workspace as a directory tree
    /stats             show model usage/switching report
    /workspace         show the current workspace path
    /resetkey          delete the saved API key
    /exit or /quit     quit
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Tab-completion via readline is opt-in (see _setup_slash_completion) because
# on some platforms — Termux on Android in particular — even just importing
# readline hooks into input() globally and has been observed to corrupt
# typed keystrokes (duplicated/interleaved characters), independent of
# whether completion is actually configured or used. So we only import it at
# all when explicitly requested.
_READLINE_AVAILABLE = False
if os.environ.get("AGENT_ENABLE_TAB_COMPLETE") == "1":
    try:
        import readline
        _READLINE_AVAILABLE = True
    except ImportError:
        _READLINE_AVAILABLE = False

import config
import tools
import model_router
import providers
import taskbar_progress as tb
from agent import GeminiAgent
from colors import C, draw_box, select_menu
from markdown_render import render_markdown

# The google-genai SDK logs a harmless warning whenever a response mixes text
# with a function_call part (e.g. the model wrote a reply AND used a tool in
# the same turn — very common for this agent). It's expected behavior here,
# not something to alert the user about, so we quiet just that logger rather
# than all logging (keeps real errors visible).
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


SLASH_COMMANDS = [
    "/help", "/clear", "/remember", "/memory", "/forget", "/undo", "/tree",
    "/ps", "/log", "/clearlog", "/image", "/force_review",
    "/multi-agent", "/settings", "/model",
    "/stats", "/workspace", "/resetkey", "/keys", "/puterJS", "/free",
    "/free-puter-models-only", "/deepresearch", "/exit", "/quit",
]

COMMAND_HINTS = {
    "/help": "show available commands",
    "/clear": "clear the screen",
    "/remember": "k=v — save a persistent fact",
    "/memory": "show everything saved in long-term memory",
    "/forget": "<key> — delete a fact from memory",
    "/undo": "revert the last file change",
    "/tree": "show the workspace as a directory tree",
    "/ps": "list background processes (dev servers, etc.)",
    "/log": "show recent actions taken this session",
    "/clearlog": "clear the execution log",
    "/image": "attach one or more images to your next message",
    "/force_review": "force the Agent to read and understand every file",
    "/multi-agent": "toggle multi-agent mode (plan/execute/review team) on or off",
    "/settings": "view or change model chain, roles, and multi-agent settings",
    "/model": "switch this session between Gemini and a Puter.js model [BETA]",
    "/stats": "model usage report and switches",
    "/workspace": "show the workspace path",
    "/resetkey": "delete the saved API key",
    "/keys": "manage multiple Gemini API keys (list/add/remove)",
    "/puterJS": "connect Puter.js for free access to 500+ AI models",
    "/free": "list Puter.js models ending in \"free\" and assign one to a role",
    "/free-puter-models-only": "use ONLY Puter.js free models (no Gemini needed)",
    "/deepresearch": "<question> — run Google AI Studio's Deep Research agent",
    "/exit": "quit the program",
    "/quit": "quit the program",
}


def _clean_user_input(raw: str) -> str:
    """
    Strips invisible Unicode formatting/control characters (RTL mark U+200F,
    LTR mark U+200E, zero-width space/joiner/non-joiner, BOM) from the start
    and end of the input, on top of a normal .strip().

    Why this matters: on RTL-aware terminals/keyboards (common when typing
    Arabic), an invisible RTL mark is frequently auto-inserted right before
    or after a typed character like '/' — completely invisible to the user,
    so "/" LOOKS identical whether or not it happened. Without stripping
    this, user_input would actually be "\u200f/" instead of "/", which
    fails every startswith("/")/== "/" check in this file (slash-suggestion
    hint, /exit, /settings, etc. would all silently stop matching) even
    though the user typed exactly what they intended to.
    """
    invisible_chars = "\u200e\u200f\u200b\u200c\u200d\ufeff"
    return raw.strip().strip(invisible_chars).strip()


def _setup_slash_completion():
    """
    Tab-completion via readline is intentionally DISABLED by default (see the
    import guard near the top of this file for why — readline has been
    observed to corrupt typed input on some platforms, notably Termux on
    Android, even without any custom completer configured). Slash commands
    remain discoverable without it: typing a lone '/' or an unrecognized
    '/xxx' both show the suggestions box (see the main loop). Set
    AGENT_ENABLE_TAB_COMPLETE=1 to opt back in on platforms where readline is
    known to behave.
    """
    if not _READLINE_AVAILABLE:
        return

    try:
        def completer(text, state):
            buffer = readline.get_line_buffer()
            if not buffer.startswith("/"):
                return None
            matches = [c for c in SLASH_COMMANDS if c.startswith(buffer)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def print_slash_suggestions():
    """Prints a compact list of available slash commands, shown when the
    user types just '/' — acts as an inline 'suggestions' hint since plain
    terminal input() can't show a live dropdown."""
    lines = [
        f"{C.CYAN}{cmd:<14}{C.RESET} {C.DIM}{COMMAND_HINTS.get(cmd, '')}{C.RESET}"
        for cmd in SLASH_COMMANDS
    ]
    print(draw_box("Suggestions", lines, color=C.TEAL))


BANNER_LINES = [
    f"{C.BOLD}{C.VIOLET}Gemini Agent{C.RESET} {C.DIM}· CLI{C.RESET}",
    f"{C.DIM}Persistent memory · automatic model switching · live file activity{C.RESET}",
]


def print_banner():
    print()
    print(draw_box("🤖", BANNER_LINES, color=C.VIOLET))
    print()


def print_help():
    rows = [
        ("/help", "show this help"),
        ("/clear", "clear the screen"),
        ("/remember k=v", "save a persistent fact, e.g. /remember name=Ahmed"),
        ("/memory", "show everything saved in long-term memory"),
        ("/forget <key>", "delete a specific key from memory"),
        ("/undo", "revert the last file change the Agent made"),
        ("/tree", "show the workspace as a directory tree"),
        ("/ps", "list background processes (dev servers)"),
        ("/log", "show recent actions taken this session"),
        ("/clearlog", "clear the execution log"),
        ("/image", "attach one or more images to your next message"),
        ("/force_review", "force reading/understanding every project file"),
        ("/multi-agent", "toggle the plan/execute/review team on or off"),
        ("/settings", "view or change models, roles, multi-agent settings"),
        ("/model", "switch this session between Gemini and a Puter.js model [BETA]"),
        ("/stats", "model usage report and automatic switches"),
        ("/workspace", "show the workspace path"),
        ("/resetkey", "delete the saved API key"),
        ("/keys", "manage multiple Gemini API keys (list/add/remove)"),
        ("/puterJS", "connect Puter.js for free access to 500+ AI models"),
        ("/free", "list Puter.js models ending in \"free\" and assign one to a role"),
        ("/free-puter-models-only", "use ONLY Puter.js free models — no Gemini API key needed"),
        ("/deepresearch <q>", "run Google AI Studio's Deep Research agent on a question"),
        ("/exit, /quit", "quit the program"),
    ]
    lines = [f"{C.CYAN}{cmd:<16}{C.RESET} {desc}" for cmd, desc in rows]
    lines.append("")
    lines.append("Any other message goes straight to the Agent — it can create/read/edit/")
    lines.append("search files and run shell commands inside the workspace.")
    print(draw_box("Commands", lines, color=C.TEAL))


class LiveStatusLine:
    """
    Shows a single "what the Agent is doing" line before its reply starts
    arriving (e.g. an animated "⠋ Thinking..." spinner, or "Running command:
    npm install"), updating in place via \\r + padding.

    History note: an earlier version used a background thread to animate the
    spinner with NO lock protecting the actual stdout write, which meant the
    animation thread and the main thread's own status updates could
    literally interleave their writes mid-string on some terminals — that's
    what caused "Thinking..." to sometimes appear duplicated instead of
    updating in place. The fix here is not "no thread" but "every write goes
    through the same lock", so the background animation thread and any
    set_activity() call from the main thread can never write at the same
    time, regardless of terminal quirks.

    Usage:
        status = LiveStatusLine()
        status.start()                              # begins animating "Thinking..."
        status.set_activity("Running command: X")   # replaces the line, still animated
        status.stop()                                # stops animation, blanks the line
    """
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAME_INTERVAL = 0.12  # seconds between animation frames

    def __init__(self):
        self._max_written_width = 0
        self._stopped = True
        self._label = "Thinking..."
        self._lock = None
        self._stop_event = None
        self._thread = None

    def start(self):
        import threading
        self._max_written_width = 0
        self._stopped = False
        self._label = "Thinking..."
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def set_activity(self, label: str):
        """Changes what the animated line shows next, e.g. 'Running command:
        npm install'. No-op if the line has already been stopped."""
        if self._stopped or self._lock is None:
            return
        with self._lock:
            self._label = label

    def _animate(self):
        import time
        frame_i = 0
        while not self._stop_event.is_set():
            with self._lock:
                frame = self.FRAMES[frame_i % len(self.FRAMES)]
                self._write_locked(f"{C.VIOLET}{frame}{C.RESET} {self._label}")
            frame_i += 1
            time.sleep(self.FRAME_INTERVAL)

    def _write_locked(self, text: str):
        """Writes text to the line in place. MUST be called with self._lock
        already held — this is what guarantees the animation thread and any
        direct caller of set_activity() never interleave a write."""
        from colors import _visible_len
        width = _visible_len(text)
        pad = max(0, self._max_written_width - width)
        self._max_written_width = max(self._max_written_width, width)
        sys.stdout.write(f"\r{text}{' ' * pad}")
        sys.stdout.flush()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Blank the line under the same lock, so there's no window where the
        # animation thread (even mid-shutdown) and this final clear could race.
        if self._lock:
            with self._lock:
                sys.stdout.write(f"\r{' ' * (self._max_written_width + 2)}\r")
                sys.stdout.flush()


def print_reply_streaming(chunk_iterator, status: "LiveStatusLine"):
    """
    Prints the agent's reply as it streams in, rendering Markdown
    progressively. Chunks are buffered line-by-line: a line is only rendered
    and printed once it's complete (ends in '\\n'), since Markdown constructs
    like **bold** or fenced code blocks need to be seen whole to render
    correctly. `status` is the shared status line (spinner + file activity /
    retry / switch messages, all updating in place on one line — see
    LiveStatusLine) which is stopped once real reply text starts arriving.
    """
    prefix_printed = False

    def ensure_prefix():
        nonlocal prefix_printed
        if not prefix_printed:
            status.stop()
            print(f"{C.GREEN}{C.BOLD}🤖 Agent{C.RESET} {C.GREEN}›{C.RESET} ", end="", flush=True)
            prefix_printed = True

    full_text_parts = []
    line_buffer = ""
    first_line = True

    try:
        tb.set_indeterminate()
        for chunk in chunk_iterator:
            ensure_prefix()
            full_text_parts.append(chunk)
            line_buffer += chunk
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                if not first_line:
                    print()
                print(render_markdown(line), end="", flush=True)
                first_line = False
    finally:
        status.stop()
        tb.clear()

    ensure_prefix()  # in case the reply was empty / only tool calls happened

    if line_buffer:
        if not first_line:
            print()
        print(render_markdown(line_buffer), end="", flush=True)

    print("\n")
    tb.clear()
    return "".join(full_text_parts)


def print_multi_agent_turn(event_iterator, status: "LiveStatusLine"):
    """
    Consumes a stream of multi_agent.MultiAgentEvent objects (from
    GeminiAgent.run_multi_agent_turn) and renders the team's progress live:

        Plan 1 of 3

        Plan 1:
         Created main.py
         Deleted old.py
        Plan 2:
         Running command...
         git clone ...
        Plan 3:
         Code check
         Zip workspace

    followed by the reviewer's final streamed summary. If the request was
    classified as "simple", this just prints the plain single-agent reply
    (no plan box) — multi-agent overhead is only shown when it actually ran.

    Any exception raised mid-stream (e.g. every model in a role's fallback
    chain failing) is caught here so a single failed step never crashes the
    whole CLI session — it's reported as a clear error and whatever
    steps/text were already shown remain on screen.
    """
    current_step_number = None
    step_action_lines: dict = {}  # step_number -> list of action summary strings
    total_steps_seen = None
    reply_prefix_printed = False

    def ensure_reply_prefix():
        nonlocal reply_prefix_printed
        if not reply_prefix_printed:
            status.stop()
            print(f"{C.GREEN}{C.BOLD}🤖 Agent{C.RESET} {C.GREEN}›{C.RESET} ", end="", flush=True)
            reply_prefix_printed = True

    full_text_parts = []
    line_buffer = ""
    first_text_line = True

    try:
        for event in event_iterator:
            if event.kind == "classified":
                if event.data["complexity"] == "simple":
                    status.set_activity("Thinking...")
                continue

            if event.kind == "plan_ready":
                status.stop()
                steps = event.data["steps"]
                total_steps_seen = len(steps)
                print(f"\n{C.BOLD}{C.VIOLET}Plan 1 of {total_steps_seen}{C.RESET}\n")
                tb.set_step_progress(1, total_steps_seen)
                status.start()
                continue

            if event.kind == "step_start":
                status.stop()
                n = event.data["step_number"]
                total = event.data["total_steps"]
                current_step_number = n
                step_action_lines[n] = []
                print(f"{C.BOLD}{C.CYAN}Plan {n}:{C.RESET}")
                tb.set_step_progress(n, total)
                status = LiveStatusLine()
                tools.set_activity_callback(make_activity_printer(status))
                status.start()
                continue

            if event.kind == "step_action":
                n = event.data["step_number"]
                summary = event.data["action_summary"]
                step_action_lines.setdefault(n, []).append(summary)
                status.stop()
                print(f" {summary}")
                status = LiveStatusLine()
                tools.set_activity_callback(make_activity_printer(status))
                status.start()
                continue

            if event.kind == "step_done":
                n = event.data["step_number"]
                if not step_action_lines.get(n):
                    status.stop()
                    print(f" {C.DIM}(no actions taken for this step){C.RESET}")
                    status = LiveStatusLine()
                    tools.set_activity_callback(make_activity_printer(status))
                    status.start()
                continue

            if event.kind == "text_chunk":
                ensure_reply_prefix()
                chunk = event.data["text"]
                full_text_parts.append(chunk)
                line_buffer += chunk
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    if not first_text_line:
                        print()
                    print(render_markdown(line), end="", flush=True)
                    first_text_line = False
                continue
    except Exception as e:
        status.stop()
        tb.set_error()
        print(f"\n{C.RED}❌ Multi-agent turn failed: {e}{C.RESET}")
        print(f"{C.DIM}(Any steps completed above were still carried out and are not undone.){C.RESET}\n")
        return "".join(full_text_parts)
    finally:
        status.stop()
        tb.set_progress(100)
        tb.clear()

    ensure_reply_prefix()
    if line_buffer:
        if not first_text_line:
            print()
        print(render_markdown(line_buffer), end="", flush=True)

    print("\n")
    tb.clear()
    return "".join(full_text_parts)


# ---------------- live file-activity status (shown on the spinner line) ----------------
_ACTIVITY_LABELS = {
    "creating": "Creating file",
    "created": "Created file",
    "editing": "Editing file",
    "edited": "Edited file",
    "deleting": "Deleting file",
    "deleted": "Deleted file",
    "running": "Running command",
}

# Stages that mean "a tool just finished" — rather than naming the just-
# completed action again (which can look frozen/stale while the model reads
# the result and decides what to do next), the status line switches back to
# "Thinking..." to accurately reflect that the model is now processing.
_BACK_TO_THINKING_STAGES = {"created", "edited", "deleted", "ran"}


def make_activity_printer(status: "LiveStatusLine"):
    """
    Returns a callback suitable for tools.set_activity_callback that updates
    the shared status line with what the Agent is doing, e.g. "Creating file
    main.py" or "Running command: npm install".

    Only "in progress" stages show the specific action (Creating/Editing/
    Deleting/Running file/command); once a tool finishes, the line switches
    back to "Thinking..." instead of showing "Created file: X" or "Ran
    command: X" — those completed-action labels tended to sit on screen
    looking stale while the model was actually busy processing the result
    and deciding its next step, which "Thinking..." communicates more
    accurately.
    """
    RUNNING_STAGES = {"creating", "editing", "deleting", "running"}

    def _summarize(text: str, limit: int = 60) -> str:
        # Collapse multi-line commands/paths (e.g. a python -c "..." with
        # embedded newlines) to a single readable line instead of dumping
        # the whole thing, which is what caused walls of near-duplicate
        # output when a long command was retried.
        first_line = text.strip().split("\n", 1)[0]
        if len(text.strip().split("\n")) > 1:
            return (first_line[:limit] + "...") if len(first_line) > limit else first_line + " ..."
        return first_line if len(first_line) <= limit else first_line[:limit] + "..."

    def _printer(stage: str, path: str):
        if stage in _BACK_TO_THINKING_STAGES:
            status.set_activity("Thinking...")
            return
        if stage not in RUNNING_STAGES:
            return
        label = _ACTIVITY_LABELS.get(stage, stage)
        display = _summarize(path)
        status.set_activity(f"{label}: {display}")

    return _printer


_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _prompt_for_images() -> list:
    """
    Prompts the user for one or more image file paths (comma-separated,
    absolute or relative to the current working directory — NOT restricted
    to the Agent's sandboxed workspace, since these are files the user is
    handing to the model directly from their own filesystem, similar to
    attaching a file in a chat app). Validates each path exists and is a
    supported image format, skipping and warning about any that aren't,
    and returns the list of valid absolute paths.
    """
    raw = input(
        f"{C.BOLD}{C.MAGENTA}Image path(s){C.RESET} {C.DIM}(comma-separated for multiple){C.RESET} "
        f"{C.MAGENTA}›{C.RESET} "
    ).strip()
    if not raw:
        return []

    valid_paths = []
    for raw_path in raw.split(","):
        raw_path = raw_path.strip().strip('"').strip("'")
        if not raw_path:
            continue
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            print(f"{C.YELLOW}⚠️  Skipping (not found): {raw_path}{C.RESET}")
            continue
        if not p.is_file():
            print(f"{C.YELLOW}⚠️  Skipping (not a file): {raw_path}{C.RESET}")
            continue
        if p.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
            print(f"{C.YELLOW}⚠️  Skipping (unsupported format '{p.suffix}'): {raw_path}{C.RESET}")
            continue
        valid_paths.append(str(p))

    if valid_paths:
        print(f"{C.GREEN}✅ Attached {len(valid_paths)} image(s).{C.RESET}")
    return valid_paths


def _persist_current_settings():
    """Saves the live in-memory config (model chain, multi-agent roles/state)
    to settings.json so changes made via /settings or /multi-agent survive
    a restart."""
    config.save_settings(config.get_current_settings_snapshot())


def handle_puterjs_command():
    """
    Connects Puter.js — free access to 500+ AI models (Claude, GPT, Gemini,
    DeepSeek, and more) via a "User-Pays" model where usage is billed to
    whichever Puter account signs in, not to this app.

    Presents an arrow-key selectable menu (see colors.select_menu) with two
    options, exactly as requested:
      1) Sign in via browser — opens Puter's own sign-in/dashboard page;
         the user creates a token there and pastes it back (Puter.js itself
         is browser-only and doesn't hand back a token to a Python process
         automatically, so this is "assisted paste" rather than a fully
         automated callback).
      2) Paste an auth token directly — for a user who already has a token
         from a previous session, skips opening a browser entirely.
    """
    choice = select_menu(
        "🧩 Connect Puter.js",
        ["Sign in via browser", "Paste an auth token directly"],
        [
            "Opens puter.com to sign in, then you copy your token back here",
            "Already have a token from puter.com/dashboard#account? Skip the browser",
        ],
    )
    if choice is None:
        print(f"{C.DIM}Cancelled.{C.RESET}")
        return

    if choice == 0:
        dashboard_url = "https://puter.com/dashboard#account"
        print(f"{C.DIM}Opening {dashboard_url} in your browser...{C.RESET}")
        try:
            import webbrowser
            opened = webbrowser.open(dashboard_url)
        except Exception:
            opened = False
        if not opened:
            print(f"{C.YELLOW}⚠️  Couldn't open a browser automatically. "
                  f"Visit this URL manually:{C.RESET}\n  {dashboard_url}")
        print(f"{C.DIM}On that page: sign in, then click 'Create token' under Account.{C.RESET}")

    token = input(f"{C.BOLD}Paste your Puter auth token{C.RESET} {C.MAGENTA}›{C.RESET} ").strip()
    if not token:
        print(f"{C.DIM}No token entered — cancelled.{C.RESET}")
        return

    providers.save_provider_api_key("puter", token)
    print(f"{C.GREEN}✅ Puter.js token saved.{C.RESET}")

    print(f"{C.DIM}Verifying and fetching your available models...{C.RESET}")
    try:
        models = providers.puter_list_models()
    except Exception as e:
        print(f"{C.YELLOW}⚠️  Token saved, but couldn't verify it yet: {e}{C.RESET}")
        return

    preview = models[:15]
    lines = [f"  {C.CYAN}{m}{C.RESET}" for m in preview]
    if len(models) > len(preview):
        lines.append(f"  {C.DIM}...and {len(models) - len(preview)} more{C.RESET}")
    lines.append("")
    lines.append(f"{C.DIM}Assign one to a multi-agent role with:{C.RESET}")
    lines.append(f"  {C.CYAN}/settings role <role> <model-name>{C.RESET}")
    print(draw_box(f"Puter.js connected — {len(models)} model(s) available", lines, color=C.GREEN))


def handle_free_command(agent):
    """
    /free — lists only the Puter.js models whose id ends in "free" (e.g.
    some providers expose a no-cost/limited tier via a name ending in
    ":free" or "-free" through Puter), and lets the user pick one with an
    arrow-key menu to immediately assign it to a multi-agent role.

    Requires a Puter token to already be configured (via /puterJS or
    /settings provider puter <token>) — this command only filters/selects
    from Puter's model list, it doesn't add a new provider.
    """
    if not providers.has_provider_key("puter"):
        print(f"{C.YELLOW}⚠️  No Puter token configured yet — use /puterJS first.{C.RESET}")
        return

    print(f"{C.DIM}Fetching free-tier Puter models...{C.RESET}")
    try:
        free_models = providers.puter_list_free_models()
    except Exception as e:
        print(f"{C.RED}❌ Could not fetch Puter models: {e}{C.RESET}")
        return

    if not free_models:
        print(f"{C.YELLOW}⚠️  No models ending in \"free\" were found in your Puter model list.{C.RESET}")
        return

    lines = [f"  {C.CYAN}{m}{C.RESET}" for m in free_models]
    print(draw_box(f"Free Puter models ({len(free_models)})", lines, color=C.GREEN))

    choice = select_menu(
        "🆓 Assign a free model to a role",
        free_models,
        None,
    )
    if choice is None:
        print(f"{C.DIM}Cancelled.{C.RESET}")
        return

    model_name = free_models[choice]
    role_names = list(config.MULTI_AGENT_ROLES.keys())
    role_choice = select_menu(
        f"Assign '{model_name}' to which role?",
        role_names,
        None,
    )
    if role_choice is None:
        print(f"{C.DIM}Cancelled.{C.RESET}")
        return

    role = role_names[role_choice]
    config.MULTI_AGENT_ROLES[role] = model_name
    _persist_current_settings()
    print(f"{C.GREEN}✅ Role '{role}' assigned to free model '{model_name}'.{C.RESET}")


def handle_free_puter_models_only(args: str = "") -> Optional[str]:
    """
    /free-puter-models-only [token] [off]

    One-command shortcut to put the CLI into "Puter free-only" mode:
      - Saves the Puter auth token (pasted or passed as argument)
      - Enables PUTER_CHAT_ENABLED (Puter models in main chat)
      - Enables PUTER_FREE_ONLY (block non-free models)
      - Enables PUTER_TOOL_CALLING_ENABLED (tools work via Puter)
      - Auto-selects a free model for the session

    No Gemini API key is needed — everything runs through Puter's free tier.

    Usage:
      /free-puter-models-only                  — paste a token interactively
      /free-puter-models-only <token>           — use the token directly
      /free-puter-models-only off               — disable, go back to Gemini

    Returns the chosen free model name (to set current_puter_model) or None.
    """
    # --- Handle "off" subcommand ---
    if args.strip().lower() == "off":
        config.PUTER_CHAT_ENABLED = False
        config.PUTER_FREE_ONLY = False
        config.PUTER_TOOL_CALLING_ENABLED = False
        _persist_current_settings()
        print(f"{C.GREEN}✅ Free Puter mode disabled. Back to Gemini.{C.RESET}")
        return None

    # --- Determine the token ---
    token = args.strip() if args.strip() and args.strip().lower() != "off" else ""

    if not token:
        # Try using already-saved token
        token = config.load_puter_token()
        if not token:
            # Ask user to paste one
            try:
                token = input(
                    f"{C.CYAN}Paste your Puter auth token{C.RESET} "
                    f"{C.DIM}(from puter.com/dashboard#account → Create token){C.RESET} "
                    f"{C.MAGENTA}›{C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C.DIM}Cancelled.{C.RESET}")
                return None

    if not token:
        print(f"{C.YELLOW}⚠️  No token provided. Cannot enable free Puter mode.{C.RESET}")
        return None

    # --- Save the token ---
    config.save_puter_token(token)

    # --- Flip all the Puter switches ON ---
    config.PUTER_CHAT_ENABLED = True
    config.PUTER_FREE_ONLY = True
    config.PUTER_TOOL_CALLING_ENABLED = True
    _persist_current_settings()

    # --- Fetch available free models ---
    print(f"{C.DIM}Fetching free Puter models...{C.RESET}")
    try:
        free_models = providers.puter_list_free_models()
    except Exception as e:
        print(f"{C.RED}❌ Could not fetch Puter models: {e}{C.RESET}")
        print(f"{C.DIM}Token saved and settings enabled — use /model puter <id> to manually pick one.{C.RESET}")
        # Still return the default free model so something is active
        return config.PUTER_FREE_CHAT_MODEL

    if not free_models:
        print(f"{C.YELLOW}⚠️  No free-tier models found. Settings are enabled but you'll need to pick a model manually.{C.RESET}")
        return config.PUTER_FREE_CHAT_MODEL

    # --- Auto-select the best free model ---
    # Prefer models that are known to support tool calling well.
    preferred_patterns = ["deepseek", "qwen", "gemma", "claude", "gpt"]
    selected_model = free_models[0]  # fallback
    for pattern in preferred_patterns:
        for m in free_models:
            if pattern in m.lower():
                selected_model = m
                break
        if selected_model != free_models[0]:
            break

    # Let the user confirm or pick a different one
    lines = [
        f"{C.GREEN}✅ Puter token saved & verified{C.RESET}",
        f"{C.GREEN}✅ Free-only mode: ON{C.RESET}",
        f"{C.GREEN}✅ Tool calling: ON{C.RESET}",
        f"{C.GREEN}✅ No Gemini API key needed{C.RESET}",
        "",
        f"{C.DIM}Auto-selected model: {C.BOLD}{selected_model}{C.RESET}{C.DIM}{C.RESET}",
        f"{C.DIM}Use /model gemini to switch back to Gemini at any time.{C.RESET}",
        f"{C.DIM}Use /free-puter-models-only off to disable.{C.RESET}",
    ]
    print(draw_box("🆓 Free Puter Mode Active", lines, color=C.GREEN))

    return selected_model


def handle_deepresearch_command(agent, query: str):
    """
    /deepresearch <question> — runs Google AI Studio's Deep Research model
    (config.DEEP_RESEARCH_MODEL) on the given question and prints the
    resulting long-form, cited report.

    This is a genuinely different capability from the normal chat models:
    Deep Research autonomously plans a multi-step web-research strategy,
    issues its own searches, and synthesizes a report — it is not just "the
    normal model with a research prompt". Because it can take noticeably
    longer than a normal chat turn (no meaningful partial-output streaming
    for this model type), no live status/spinner is shown beyond a single
    "researching..." message — see model_router.run_deep_research()'s
    docstring for why this deliberately bypasses the normal ModelRouter
    failover chain.
    """
    if not query:
        print(f"{C.YELLOW}⚠️  Usage: /deepresearch <question or topic>{C.RESET}")
        print(f"{C.DIM}   Example: /deepresearch What are the latest advances in solid-state batteries?{C.RESET}")
        print(f"{C.DIM}   Model used: {config.DEEP_RESEARCH_MODEL}  "
              f"(change with /settings deepresearch model <model-id>){C.RESET}")
        return

    api_key = config.GEMINI_API_KEY or config.load_saved_api_key()
    if not api_key:
        print(f"{C.RED}❌ No Gemini API key configured — Deep Research requires Google AI Studio access.{C.RESET}")
        return

    print(f"{C.MAGENTA}🔎 Running Deep Research ({config.DEEP_RESEARCH_MODEL})... "
          f"this can take a while for thorough topics.{C.RESET}")
    try:
        report_text = model_router.run_deep_research(api_key, query)
    except Exception as e:
        print(f"{C.RED}❌ Deep Research failed: {e}{C.RESET}")
        print(f"{C.DIM}   Note: Deep Research models may not be enabled for every API key/tier — "
              f"check availability in Google AI Studio, or try a different model with "
              f"/settings deepresearch model <model-id>.{C.RESET}")
        return

    print(draw_box("🔎 Deep Research Report", [render_markdown(report_text)], color=C.MAGENTA))


def handle_model_command(agent, args: str, current_puter_model: Optional[str]) -> Optional[str]:
    """
    /model — session-only quick switch between Gemini (the default chain)
    and a single Puter.js model [BETA], for the rest of this conversation
    (or until changed again / the app restarts — this is NOT persisted to
    disk like /settings changes, since it's meant as a "just for now"
    override rather than a standing preference).

    Usage:
        /model                    — show the current active model
        /model gemini             — switch back to Gemini (the default)
        /model puter               — pick a Puter model interactively from
                                      an arrow-key menu (requires a Puter
                                      token; requires /settings puter tools
                                      on and /settings puter chat on to
                                      actually take effect, same as the
                                      puter_model parameter everywhere else
                                      in this app — checked and warned
                                      about here if not yet on)
        /model puter <model-id>   — switch directly to a named Puter model
                                      without the picker menu

    Returns the new current_puter_model value (None means "use Gemini") —
    the caller (main()'s loop) stores this and passes it into
    _send_and_print() for every subsequent message until /model changes it
    again.
    """
    arg_parts = args.strip().split(maxsplit=1)
    sub = arg_parts[0].lower() if arg_parts else ""

    if not sub:
        if current_puter_model:
            print(f"{C.CYAN}Active model: {C.BOLD}{current_puter_model}{C.RESET} {C.DIM}(via Puter.js, BETA){C.RESET}")
        else:
            print(f"{C.CYAN}Active model: {C.BOLD}{agent.router.current_model_name}{C.RESET} {C.DIM}(Gemini){C.RESET}")
        print(f"{C.DIM}Use /model gemini or /model puter [model-id] to switch.{C.RESET}")
        return current_puter_model

    if sub == "gemini":
        if current_puter_model is None:
            print(f"{C.DIM}Already using Gemini.{C.RESET}")
        else:
            print(f"{C.GREEN}✅ Switched back to Gemini for this session.{C.RESET}")
        return None

    if sub == "puter":
        if not providers.has_provider_key("puter"):
            print(f"{C.YELLOW}⚠️  No Puter token configured yet — use /puterJS first.{C.RESET}")
            return current_puter_model
        if not (config.PUTER_CHAT_ENABLED and config.PUTER_TOOL_CALLING_ENABLED):
            print(f"{C.YELLOW}⚠️  /model puter also needs both of these turned on to actually take effect:{C.RESET}")
            print(f"{C.YELLOW}   /settings puter chat on{C.RESET}")
            print(f"{C.YELLOW}   /settings puter tools on   {C.DIM}(BETA){C.RESET}")
            print(f"{C.DIM}   You can still pick a model now — it just won't be used until those are on.{C.RESET}")

        explicit_model = arg_parts[1].strip() if len(arg_parts) > 1 else None
        if explicit_model:
            model_name = explicit_model
        else:
            print(f"{C.DIM}Fetching your Puter models...{C.RESET}")
            try:
                available_models = providers.puter_list_models()
            except Exception as e:
                print(f"{C.RED}❌ Could not fetch Puter models: {e}{C.RESET}")
                return current_puter_model
            if not available_models:
                print(f"{C.YELLOW}⚠️  No models found in your Puter account.{C.RESET}")
                return current_puter_model
            choice = select_menu(f"🧪 Pick a Puter.js model for this session [BETA]", available_models, None)
            if choice is None:
                print(f"{C.DIM}Cancelled.{C.RESET}")
                return current_puter_model
            model_name = available_models[choice]

        print(f"{C.MAGENTA}🧪 [BETA]{C.RESET} {C.GREEN}Switched to Puter.js model "
              f"'{model_name}' for this session.{C.RESET}")
        print(f"{C.DIM}   May not support tool calling/vision reliably — see /settings puter for details.{C.RESET}")
        print(f"{C.DIM}   Use /model gemini to switch back.{C.RESET}")
        return model_name

    print(f"{C.YELLOW}⚠️  Usage: /model | /model gemini | /model puter [model-id]{C.RESET}")
    return current_puter_model


def print_keys_menu(agent):
    """
    Shows every configured Gemini API key (masked — never the full secret),
    which one is currently active, and how to add/remove additional keys.
    """
    pool = config.load_api_key_pool()
    lines = []
    if not pool:
        lines.append(f"{C.DIM}No API keys configured.{C.RESET}")
    else:
        for i, key in enumerate(pool):
            marker = "→" if i == agent.router.current_key_index else " "
            active = f" {C.GREEN}(active){C.RESET}" if i == agent.router.current_key_index else ""
            lines.append(f"  {marker} #{i + 1}  {config.mask_api_key(key)}{active}")

    lines.append("")
    lines.append(f"{C.DIM}Why multiple keys? Each Gemini API key has its own daily request{C.RESET}")
    lines.append(f"{C.DIM}quota (RPD). When one key's quota is exhausted for a model, the{C.RESET}")
    lines.append(f"{C.DIM}agent automatically rotates to the next configured key instead of{C.RESET}")
    lines.append(f"{C.DIM}waiting until tomorrow.{C.RESET}")
    lines.append("")
    lines.append(f"{C.CYAN}/keys add <key>{C.RESET}          add another Gemini API key to the pool")
    lines.append(f"{C.CYAN}/keys remove <last4chars>{C.RESET}  remove a key by its last few characters")

    print(draw_box("API Keys", lines, color=C.TEAL))


def handle_keys_subcommand(agent, rest: str):
    """Handles '/keys add <key>' and '/keys remove <suffix>'."""
    parts = rest.strip().split(maxsplit=1)
    if not parts:
        print_keys_menu(agent)
        return

    sub = parts[0].lower()

    if sub == "add":
        if len(parts) < 2 or not parts[1].strip():
            print(f"{C.YELLOW}⚠️  Usage: /keys add <your-gemini-api-key>{C.RESET}")
            return
        new_key = parts[1].strip()
        existing_pool = config.load_api_key_pool()
        if new_key in existing_pool:
            print(f"{C.DIM}ℹ️  That key is already in the pool.{C.RESET}")
            return
        config.add_api_key_to_pool(new_key)
        # Refresh the live router's pool so the new key is usable immediately,
        # without needing to restart the CLI.
        agent.router.key_pool = config.load_api_key_pool()
        print(f"{C.GREEN}✅ Added key {config.mask_api_key(new_key)} to the pool "
              f"({len(agent.router.key_pool)} key(s) total).{C.RESET}")
        return

    if sub == "remove":
        if len(parts) < 2 or not parts[1].strip():
            print(f"{C.YELLOW}⚠️  Usage: /keys remove <last-characters-of-key>{C.RESET}")
            return
        suffix = parts[1].strip()
        removed = config.remove_api_key_from_pool(suffix)
        if removed:
            agent.router.key_pool = config.load_api_key_pool()
            if agent.router.current_key_index >= len(agent.router.key_pool):
                agent.router.current_key_index = 0
            print(f"{C.GREEN}✅ Removed a key ending in '{suffix}'. "
                  f"{len(agent.router.key_pool)} key(s) remain.{C.RESET}")
        else:
            print(f"{C.YELLOW}⚠️  No pool key found ending in '{suffix}'. "
                  f"(The primary key can't be removed this way — use /resetkey.){C.RESET}")
        return

    print(f"{C.YELLOW}⚠️  Unknown /keys subcommand '{sub}'. Try /keys for the menu.{C.RESET}")


def print_settings_menu(agent):
    """
    Shows the current settings (model chain, multi-agent roles/enabled
    state) and how to change them. Actual editing happens via focused
    sub-commands rather than a single freeform /settings prompt, to avoid
    ambiguous input parsing:
      /settings models          — list every model available to this API key
      /settings role <role> <model>  — assign a model to a multi-agent role
      /settings chain <model1,model2,...> — replace the model failover chain
    """
    lines = [
        f"{C.BOLD}Model chain{C.RESET} (failover order):",
    ]
    for i, m in enumerate(config.MODEL_CHAIN):
        marker = "→" if i == agent.router.current_index else " "
        lines.append(f"  {marker} {m['name']}  {C.DIM}(cap: {m.get('max_requests_per_session') or 'none'}){C.RESET}")

    lines.append("")
    lines.append(f"{C.BOLD}Multi-agent roles{C.RESET}:")
    for role, model in config.MULTI_AGENT_ROLES.items():
        lines.append(f"  {C.CYAN}{role:<12}{C.RESET} {model}")
    lines.append(f"  Multi-agent mode: {'ON' if config.MULTI_AGENT_ENABLED else 'OFF'} (toggle with /multi-agent)")

    lines.append("")
    lines.append(f"{C.BOLD}Puter.js{C.RESET}:")
    lines.append(f"  Use in chat: {'ON' if config.PUTER_CHAT_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings puter chat on|off){C.RESET}")
    lines.append(f"  Free models only: {'ON' if config.PUTER_FREE_ONLY else 'OFF'} "
                  f"{C.DIM}(toggle with /settings puter free on|off){C.RESET}")
    lines.append(f"  Tool calling {C.MAGENTA}[BETA]{C.RESET}: {'ON' if config.PUTER_TOOL_CALLING_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings puter tools on|off){C.RESET}")
    if config.PUTER_TOOL_CALLING_ENABLED:
        lines.append(f"    {C.YELLOW}⚠️  Beta feature — may not work on all Puter models. Some models{C.RESET}")
        lines.append(f"    {C.YELLOW}   may ignore tools or return malformed tool calls.{C.RESET}")
    lines.append(f"  Image fallback {C.MAGENTA}[BETA]{C.RESET}: {'ON' if config.PUTER_IMAGE_TOOLS_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings puter images on|off){C.RESET}")
    if config.PUTER_IMAGE_TOOLS_ENABLED:
        lines.append(f"    {C.YELLOW}⚠️  Beta feature — offered only after Gemini's Image_Fetch/Image_Create{C.RESET}")
        lines.append(f"    {C.YELLOW}   fails, and only with your explicit go-ahead each time.{C.RESET}")
        lines.append(f"    {C.YELLOW}   Image generation via Puter is unverified and may simply fail.{C.RESET}")
        lines.append(f"    {C.DIM}   vision model: {config.PUTER_VISION_MODEL}  |  image-gen model: {config.PUTER_IMAGE_GEN_MODEL}{C.RESET}")

    lines.append("")
    lines.append(f"{C.BOLD}Deep Thinking{C.RESET}:")
    lines.append(f"  Gemini: {'ON' if config.DEEP_THINKING_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings thinking on|off){C.RESET}")
    if config.DEEP_THINKING_ENABLED:
        budget_label = "dynamic/auto" if config.DEEP_THINKING_BUDGET == -1 else str(config.DEEP_THINKING_BUDGET)
        lines.append(f"    {C.DIM}budget: {budget_label}  |  show thoughts: "
                      f"{'yes' if config.DEEP_THINKING_INCLUDE_THOUGHTS else 'no'}{C.RESET}")
    lines.append(f"  Puter.js {C.MAGENTA}[BETA]{C.RESET}: {'ON' if config.PUTER_DEEP_THINKING_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings puter thinking on|off){C.RESET}")
    if config.PUTER_DEEP_THINKING_ENABLED:
        lines.append(f"    {C.DIM}reasoning effort: {config.PUTER_DEEP_THINKING_EFFORT}{C.RESET}")

    lines.append("")
    lines.append(f"{C.BOLD}Deep Research{C.RESET} (Google AI Studio):")
    lines.append(f"  Model: {C.CYAN}{config.DEEP_RESEARCH_MODEL}{C.RESET}")
    lines.append(f"  Run with: {C.CYAN}/deepresearch <question>{C.RESET}")

    lines.append("")
    lines.append(f"{C.DIM}To change things:{C.RESET}")
    lines.append(f"  {C.CYAN}/settings models{C.RESET}          list every model your API key can use")
    lines.append(f"  {C.CYAN}/settings role <role> <model>{C.RESET}  assign a model to a role, e.g.")
    lines.append(f"                                 /settings role planner gemini-2.5-pro")
    lines.append(f"  {C.CYAN}/settings chain <m1,m2,...>{C.RESET}    replace the failover chain order")
    lines.append(f"  {C.CYAN}/settings puter chat on|off{C.RESET}    use Puter.js models in main chat")
    lines.append(f"  {C.CYAN}/settings puter free on|off{C.RESET}    restrict Puter.js calls to models with 'free' in the name")
    lines.append(f"  {C.CYAN}/settings puter tools on|off{C.RESET}   {C.MAGENTA}[BETA]{C.RESET} let Puter.js models call the agent's tools")
    lines.append(f"  {C.CYAN}/settings puter images on|off{C.RESET}  {C.MAGENTA}[BETA]{C.RESET} offer Puter.js as a fallback when Gemini image tools fail")
    lines.append(f"  {C.CYAN}/settings puter vision-model <model>{C.RESET}  Puter model for Image_Fetch_Puter")
    lines.append(f"  {C.CYAN}/settings puter image-model <model>{C.RESET}   Puter model for Image_Create_Puter")
    lines.append(f"  {C.CYAN}/settings thinking on|off{C.RESET}      toggle Gemini Deep Thinking")
    lines.append(f"  {C.CYAN}/settings thinking budget <n|auto>{C.RESET}  set thinking token budget (-1/auto = dynamic)")
    lines.append(f"  {C.CYAN}/settings thinking show on|off{C.RESET} show/hide the model's thought summaries")
    lines.append(f"  {C.CYAN}/settings puter thinking on|off{C.RESET}       {C.MAGENTA}[BETA]{C.RESET} toggle Puter.js Deep Thinking (reasoning_effort)")
    lines.append(f"  {C.CYAN}/settings puter thinking effort <low|medium|high>{C.RESET}  set Puter reasoning effort")
    lines.append(f"  {C.CYAN}/settings deepresearch model <model-id>{C.RESET}  set the Deep Research model")
    lines.append("")
    lines.append(f"{C.BOLD}System access{C.RESET}: {'ON' if config.SYSTEM_ACCESS_ENABLED else 'OFF'} "
                  f"{C.DIM}(toggle with /settings system access on|off){C.RESET}")
    lines.append(f"    {C.DIM}Gates Available_Active_Windows (open windows) and{C.RESET}")
    lines.append(f"    {C.DIM}List_System_Processes (Task-Manager-style process list). Off by default.{C.RESET}")
    lines.append(f"  {C.CYAN}/settings system access on|off{C.RESET}  allow/deny window & process access tools")

    print(draw_box("Settings", lines, color=C.TEAL))


def handle_settings_subcommand(agent, rest: str):
    """
    Handles '/settings <subcommand> ...' — models / role / chain. Called
    from the main loop when user_input starts with '/settings ' (i.e. has
    an argument), as opposed to bare '/settings' which just shows the menu.
    """
    parts = rest.strip().split(maxsplit=2)
    if not parts:
        print_settings_menu(agent)
        return

    sub = parts[0].lower()

    if sub == "models":
        print(f"{C.DIM}Fetching available models for your API key...{C.RESET}")
        try:
            available = model_router.fetch_available_models(config.GEMINI_API_KEY or config.load_saved_api_key())
        except Exception as e:
            print(f"{C.RED}❌ Could not fetch models: {e}{C.RESET}")
            return
        if not available:
            print(f"{C.DIM}No text-generation-capable models found for this key.{C.RESET}")
            return
        lines = []
        for m in available:
            in_chain = " (in chain)" if m["name"] in [c["name"] for c in config.MODEL_CHAIN] else ""
            lines.append(f"  {C.CYAN}{m['name']}{C.RESET}{C.DIM}{in_chain}{C.RESET}")
            lines.append(f"    {C.DIM}{m['display_name']} — in:{m['input_token_limit']} out:{m['output_token_limit']}{C.RESET}")
        print(draw_box(f"Available models ({len(available)})", lines, color=C.TEAL))
        return

    if sub == "role":
        if len(parts) < 3:
            print(f"{C.YELLOW}⚠️  Usage: /settings role <role> <model>  "
                  f"(roles: {', '.join(config.MULTI_AGENT_ROLES.keys())}){C.RESET}")
            return
        role, model_name = parts[1], parts[2].strip()
        if role not in config.MULTI_AGENT_ROLES:
            print(f"{C.YELLOW}⚠️  Unknown role '{role}'. Valid roles: "
                  f"{', '.join(config.MULTI_AGENT_ROLES.keys())}{C.RESET}")
            return
        config.MULTI_AGENT_ROLES[role] = model_name
        _persist_current_settings()
        print(f"{C.GREEN}✅ Role '{role}' assigned to model '{model_name}'.{C.RESET}")
        return

    if sub == "system":
        if len(parts) < 3 or parts[1].strip().lower() != "access" or parts[2].strip().lower() not in ("on", "off"):
            print(f"{C.YELLOW}⚠️  Usage: /settings system access on|off{C.RESET}")
            return
        state = parts[2].strip().lower() == "on"
        config.SYSTEM_ACCESS_ENABLED = state
        _persist_current_settings()
        if state:
            print(f"{C.MAGENTA}🔓 System access is now ON.{C.RESET}")
            print(f"{C.YELLOW}⚠️  The AI can now use:{C.RESET}")
            print(f"{C.YELLOW}   • Available_Active_Windows — list your open windows and screenshot/describe each{C.RESET}")
            print(f"{C.YELLOW}   • List_System_Processes — a Task-Manager-style listing of ALL system processes{C.RESET}")
            print(f"{C.DIM}   Turn it back off any time with /settings system access off.{C.RESET}")
        else:
            print(f"{C.GREEN}✅ System access is now OFF — those two tools will be refused.{C.RESET}")
        return

    if sub == "chain":
        if len(parts) < 2:
            print(f"{C.YELLOW}⚠️  Usage: /settings chain <model1,model2,...>{C.RESET}")
            return
        model_names = [m.strip() for m in " ".join(parts[1:]).split(",") if m.strip()]
        if not model_names:
            print(f"{C.YELLOW}⚠️  No model names provided.{C.RESET}")
            return
        new_chain = [{"name": name, "max_requests_per_session": 200} for name in model_names]
        config.MODEL_CHAIN.clear()
        config.MODEL_CHAIN.extend(new_chain)
        agent.router.chain = config.MODEL_CHAIN
        agent.router.current_index = 0
        agent.router.request_counts = {m["name"]: 0 for m in config.MODEL_CHAIN}
        _persist_current_settings()
        print(f"{C.GREEN}✅ Model chain updated: {' → '.join(model_names)}{C.RESET}")
        return

    if sub == "provider":
        if len(parts) < 2:
            configured = [p for p in providers.PROVIDERS if providers.has_provider_key(p)]
            lines = [f"{C.DIM}For multi-agent roles only (executor always uses Gemini):{C.RESET}"]
            for pid, info in providers.PROVIDERS.items():
                if pid == "gemini":
                    continue
                status = f"{C.GREEN}configured{C.RESET}" if pid in configured else f"{C.DIM}not set{C.RESET}"
                lines.append(f"  {C.CYAN}{pid:<10}{C.RESET} {info['label']:<26} {status}")
            lines.append("")
            lines.append(f"  {C.CYAN}/settings provider <name> <key>{C.RESET}   set a key/token")
            lines.append(f"  {C.CYAN}/settings provider puter models{C.RESET}   list models available via Puter")
            print(draw_box("Providers", lines, color=C.TEAL))
            return

        provider_id = parts[1].lower()
        if provider_id not in providers.PROVIDERS or provider_id == "gemini":
            print(f"{C.YELLOW}⚠️  Unknown provider '{provider_id}'. Valid: "
                  f"{', '.join(p for p in providers.PROVIDERS if p != 'gemini')}{C.RESET}")
            return

        if provider_id == "puter" and len(parts) >= 3 and parts[2].strip().lower() == "models":
            if not providers.has_provider_key("puter"):
                print(f"{C.YELLOW}⚠️  No Puter token configured yet — use /puterJS first.{C.RESET}")
                return
            try:
                models = providers.puter_list_models()
            except Exception as e:
                print(f"{C.RED}❌ Could not fetch Puter models: {e}{C.RESET}")
                return
            lines = [f"  {C.CYAN}{m}{C.RESET}" for m in models[:40]]
            if len(models) > 40:
                lines.append(f"  {C.DIM}...and {len(models) - 40} more{C.RESET}")
            print(draw_box(f"Puter models ({len(models)})", lines, color=C.TEAL))
            return

        if len(parts) < 3:
            print(f"{C.YELLOW}⚠️  Usage: /settings provider {provider_id} <key-or-token>{C.RESET}")
            return
        key_value = parts[2].strip()
        providers.save_provider_api_key(provider_id, key_value)
        print(f"{C.GREEN}✅ {providers.PROVIDERS[provider_id]['label']} key saved.{C.RESET}")
        return

    if sub == "puter":
        if len(parts) >= 2 and parts[1].strip().lower() == "vision-model":
            if len(parts) < 3 or not parts[2].strip():
                print(f"{C.YELLOW}⚠️  Usage: /settings puter vision-model <model-name>{C.RESET}")
                return
            config.PUTER_VISION_MODEL = parts[2].strip()
            _persist_current_settings()
            print(f"{C.GREEN}✅ Puter.js vision model (Image_Fetch_Puter) set to '{config.PUTER_VISION_MODEL}'.{C.RESET}")
            return

        if len(parts) >= 2 and parts[1].strip().lower() == "image-model":
            if len(parts) < 3 or not parts[2].strip():
                print(f"{C.YELLOW}⚠️  Usage: /settings puter image-model <model-name>{C.RESET}")
                return
            config.PUTER_IMAGE_GEN_MODEL = parts[2].strip()
            _persist_current_settings()
            print(f"{C.GREEN}✅ Puter.js image-generation model (Image_Create_Puter) set to "
                  f"'{config.PUTER_IMAGE_GEN_MODEL}'.{C.RESET}")
            return

        if len(parts) >= 2 and parts[1].strip().lower() == "thinking":
            # /settings puter thinking on|off
            # /settings puter thinking effort <low|medium|high>
            rest2 = parts[2].strip() if len(parts) >= 3 else ""
            sub2 = rest2.split(maxsplit=1)
            if sub2 and sub2[0].lower() == "effort":
                effort = sub2[1].strip().lower() if len(sub2) > 1 else ""
                if effort not in ("low", "medium", "high"):
                    print(f"{C.YELLOW}⚠️  Usage: /settings puter thinking effort <low|medium|high>{C.RESET}")
                    return
                config.PUTER_DEEP_THINKING_EFFORT = effort
                _persist_current_settings()
                print(f"{C.GREEN}✅ Puter.js Deep Thinking reasoning effort set to '{effort}'.{C.RESET}")
                return
            if rest2.lower() not in ("on", "off"):
                print(f"{C.YELLOW}⚠️  Usage: /settings puter thinking on|off   "
                      f"or   /settings puter thinking effort <low|medium|high>{C.RESET}")
                return
            state2 = rest2.lower() == "on"
            config.PUTER_DEEP_THINKING_ENABLED = state2
            _persist_current_settings()
            if state2:
                print(f"{C.MAGENTA}🧪 [BETA] Puter.js Deep Thinking is now ON "
                      f"(reasoning_effort={config.PUTER_DEEP_THINKING_EFFORT}).{C.RESET}")
                print(f"{C.YELLOW}⚠️  Only reasoning-capable models (o1/o3, deepseek-reasoner, Claude{C.RESET}")
                print(f"{C.YELLOW}   extended-thinking, etc.) actually use this — other models ignore it.{C.RESET}")
            else:
                print(f"{C.GREEN}✅ Puter.js Deep Thinking is now OFF.{C.RESET}")
            return

        if len(parts) < 3 or parts[1].strip().lower() not in ("chat", "free", "tools", "images") or parts[2].strip().lower() not in ("on", "off"):
            print(f"{C.YELLOW}⚠️  Usage: /settings puter chat on|off   or   /settings puter free on|off   "
                  f"or   /settings puter tools on|off {C.MAGENTA}[BETA]{C.RESET}   "
                  f"or   /settings puter images on|off {C.MAGENTA}[BETA]{C.RESET}   "
                  f"or   /settings puter thinking on|off {C.MAGENTA}[BETA]{C.RESET}")
            return
        toggle, state = parts[1].strip().lower(), parts[2].strip().lower() == "on"
        if toggle == "chat":
            config.PUTER_CHAT_ENABLED = state
            _persist_current_settings()
            if state:
                print(f"{C.GREEN}✅ Puter.js models can now be used in the main chat.{C.RESET} "
                      f"{C.DIM}Note: by default this is plain-text chat only — enable "
                      f"/settings puter tools on {C.MAGENTA}[BETA]{C.RESET}{C.DIM} to also let Puter "
                      f"models call the agent's tools.{C.RESET}")
            else:
                print(f"{C.GREEN}✅ Puter.js is now limited to /puterJS and /free only (not used in main chat).{C.RESET}")
        elif toggle == "free":
            config.PUTER_FREE_ONLY = state
            _persist_current_settings()
            if state:
                print(f"{C.GREEN}✅ Puter.js calls are now restricted to models with 'free' in the name.{C.RESET}")
            else:
                print(f"{C.GREEN}✅ Puter.js calls can now use any model from your account, not just free ones.{C.RESET}")
        elif toggle == "tools":
            config.PUTER_TOOL_CALLING_ENABLED = state
            _persist_current_settings()
            if state:
                print(f"{C.MAGENTA}🧪 [BETA] Puter.js tool calling is now ON.{C.RESET}")
                print(f"{C.YELLOW}⚠️  This is a beta feature and may not work reliably on all models:{C.RESET}")
                print(f"{C.YELLOW}   • Some Puter models may ignore the available tools entirely.{C.RESET}")
                print(f"{C.YELLOW}   • Some may return malformed or hallucinated tool calls.{C.RESET}")
                print(f"{C.YELLOW}   • Behavior has only been spot-checked, not fully verified across{C.RESET}")
                print(f"{C.YELLOW}     Puter's 500+ models.{C.RESET}")
                if not config.PUTER_CHAT_ENABLED:
                    print(f"{C.DIM}   Note: /settings puter chat is currently OFF — turn it on too "
                          f"({C.CYAN}/settings puter chat on{C.RESET}{C.DIM}) for this to take effect.{C.RESET}")
            else:
                print(f"{C.GREEN}✅ Puter.js tool calling is now OFF (plain text only, if Puter chat is enabled).{C.RESET}")
        else:  # images
            config.PUTER_IMAGE_TOOLS_ENABLED = state
            _persist_current_settings()
            if state:
                print(f"{C.MAGENTA}🧪 [BETA] Puter.js image fallback is now ON.{C.RESET}")
                print(f"{C.YELLOW}⚠️  This is a beta feature:{C.RESET}")
                print(f"{C.YELLOW}   • Only offered after Gemini's Image_Fetch/Image_Create fails.{C.RESET}")
                print(f"{C.YELLOW}   • The agent will ask your permission each time before using it.{C.RESET}")
                print(f"{C.YELLOW}   • Image GENERATION via Puter is unverified — Puter's own docs only{C.RESET}")
                print(f"{C.YELLOW}     show it working in the browser, not through this REST path.{C.RESET}")
                print(f"{C.YELLOW}     It may simply fail; Gemini remains the reliable option.{C.RESET}")
                if not config.load_puter_token():
                    print(f"{C.DIM}   Note: no Puter.js token configured yet — use /puterJS first, or{C.RESET}")
                    print(f"{C.DIM}   this fallback won't actually be offered.{C.RESET}")
            else:
                print(f"{C.GREEN}✅ Puter.js image fallback is now OFF (Gemini-only for images).{C.RESET}")
        return

    if sub == "thinking":
        # /settings thinking on|off
        # /settings thinking budget <n|auto>
        # /settings thinking show on|off
        if len(parts) >= 2 and parts[1].strip().lower() == "budget":
            if len(parts) < 3 or not parts[2].strip():
                print(f"{C.YELLOW}⚠️  Usage: /settings thinking budget <n|auto>  "
                      f"(auto/-1 = let the model decide, 0 = disable thinking){C.RESET}")
                return
            raw = parts[2].strip().lower()
            if raw == "auto":
                config.DEEP_THINKING_BUDGET = -1
            else:
                try:
                    config.DEEP_THINKING_BUDGET = int(raw)
                except ValueError:
                    print(f"{C.YELLOW}⚠️  Budget must be an integer or 'auto'.{C.RESET}")
                    return
            _persist_current_settings()
            label = "dynamic/auto" if config.DEEP_THINKING_BUDGET == -1 else str(config.DEEP_THINKING_BUDGET)
            print(f"{C.GREEN}✅ Gemini Deep Thinking budget set to {label}.{C.RESET}")
            return

        if len(parts) >= 2 and parts[1].strip().lower() == "show":
            if len(parts) < 3 or parts[2].strip().lower() not in ("on", "off"):
                print(f"{C.YELLOW}⚠️  Usage: /settings thinking show on|off{C.RESET}")
                return
            config.DEEP_THINKING_INCLUDE_THOUGHTS = parts[2].strip().lower() == "on"
            _persist_current_settings()
            print(f"{C.GREEN}✅ Showing thought summaries: "
                  f"{'ON' if config.DEEP_THINKING_INCLUDE_THOUGHTS else 'OFF'}.{C.RESET}")
            return

        if len(parts) < 2 or parts[1].strip().lower() not in ("on", "off"):
            print(f"{C.YELLOW}⚠️  Usage: /settings thinking on|off   or   /settings thinking budget <n|auto>   "
                  f"or   /settings thinking show on|off{C.RESET}")
            return
        state3 = parts[1].strip().lower() == "on"
        config.DEEP_THINKING_ENABLED = state3
        _persist_current_settings()
        if state3:
            print(f"{C.GREEN}✅ Gemini Deep Thinking is now ON "
                  f"(budget={'auto' if config.DEEP_THINKING_BUDGET == -1 else config.DEEP_THINKING_BUDGET}).{C.RESET}")
            print(f"{C.DIM}   Only thinking-capable Gemini models (2.5/3.x line) actually use this — {C.RESET}")
            print(f"{C.DIM}   other models ignore it. Thinking uses extra tokens and can be slower.{C.RESET}")
        else:
            print(f"{C.GREEN}✅ Gemini Deep Thinking is now OFF.{C.RESET}")
        return

    if sub == "deepresearch":
        if len(parts) < 3 or parts[1].strip().lower() != "model" or not parts[2].strip():
            print(f"{C.YELLOW}⚠️  Usage: /settings deepresearch model <model-id>{C.RESET}")
            print(f"{C.DIM}   Current: {config.DEEP_RESEARCH_MODEL}{C.RESET}")
            return
        config.DEEP_RESEARCH_MODEL = parts[2].strip()
        _persist_current_settings()
        print(f"{C.GREEN}✅ Deep Research model set to '{config.DEEP_RESEARCH_MODEL}'.{C.RESET}")
        return

    print(f"{C.YELLOW}⚠️  Unknown /settings subcommand '{sub}'. Try /settings for the menu.{C.RESET}")


def _send_and_print(agent, message: str, image_paths: list = None, puter_model: Optional[str] = None):
    """
    Shared helper: sends a message to the agent via the standard streaming
    flow (status line wired up, error handling included) and prints the
    reply. Used by both the normal message loop and special commands like
    /image and /force_review that construct their own message text.

    If multi-agent mode is enabled (config.MULTI_AGENT_ENABLED, toggled via
    /multi-agent) and no images are attached, this routes through the
    classifier -> plan -> execute -> review pipeline instead of the plain
    single-model flow. Image attachments always use the plain flow since
    the multi-agent prompts aren't built to carry image parts through the
    planning stage.

    Args:
        puter_model: BETA. If set (via /model puter <id>), routes this
            message through Puter instead of Gemini for the rest of this
            session — see handle_model_command()'s docstring. Takes
            priority over multi-agent mode (multi-agent's classify/plan/
            execute/review pipeline is Gemini-only; a Puter model active
            via /model always uses the plain single-model flow instead).
    """
    status = LiveStatusLine()
    tools.set_activity_callback(make_activity_printer(status))
    agent.router.set_status_callback(status.set_activity)
    status.start()

    if config.MULTI_AGENT_ENABLED and not image_paths and not puter_model:
        try:
            print_multi_agent_turn(agent.run_multi_agent_turn(message), status)
        except RuntimeError as e:
            status.stop()
            print(f"\n{C.RED}❌ All models failed: {e}{C.RESET}\n")
        except Exception as e:
            status.stop()
            print(f"\n{C.RED}❌ Unexpected error: {e}{C.RESET}\n")
        return

    try:
        print_reply_streaming(agent.send_stream(message, image_paths=image_paths, puter_model=puter_model), status)
    except RuntimeError as e:
        status.stop()
        tb.set_error()
        print(f"\n{C.RED}❌ All models failed: {e}{C.RESET}\n")
    except Exception as e:
        status.stop()
        tb.set_error()
        print(f"\n{C.RED}❌ Unexpected error: {e}{C.RESET}\n")


def _build_force_review_message() -> Optional[str]:
    """
    Builds a strong directive message instructing the Agent to read and
    understand every file in the workspace, one at a time, using
    list_files() to enumerate them (respecting .agentignore) and read_file()
    on each. Returns None if the workspace has no files to review.
    """
    listing = tools.list_files()
    if listing.startswith("❌") or listing.startswith("📁 Directory is empty"):
        return None

    file_paths = [
        line.split(" ", 1)[1] for line in listing.split("\n")
        if line.startswith("📄 ")
    ]
    if not file_paths:
        return None

    file_list_str = "\n".join(f"- {p}" for p in file_paths)
    return (
        "FORCED FULL PROJECT REVIEW: You must read and understand EVERY file "
        "listed below before responding, one at a time, using read_file on "
        "each — do not skip any, and do not summarize from assumptions or "
        "partial memory of earlier messages. This is the complete file list "
        f"({len(file_paths)} file(s)):\n\n{file_list_str}\n\n"
        "After reading all of them, give a structured summary covering: "
        "(1) what this project is and does, (2) how the files relate to "
        "each other, (3) any issues, TODOs, or incomplete parts you noticed "
        "while reading. Read every single file before writing your summary — "
        "do not respond until you have."
    )


def print_tree(path: str = "."):
    """Renders the workspace listing as an indented tree using list_files'
    ignore-aware output."""
    listing = tools.list_files(path)
    if listing.startswith("❌") or listing.startswith("📁 Directory is empty"):
        print(listing)
        return

    lines = listing.split("\n")
    tree_lines = []
    for line in lines:
        # list_files returns "📄 relative/path" or "📁 relative/path"
        icon, _, rel = line.partition(" ")
        depth = rel.count("/")
        indent = "  " * depth
        name = rel.rsplit("/", 1)[-1]
        color = C.BLUE if icon == "📁" else C.RESET
        tree_lines.append(f"{indent}{icon} {color}{name}{C.RESET}")
    print(draw_box(f"Workspace: {config.WORKSPACE_DIR.name}", tree_lines, color=C.TEAL))


def main():
    print_banner()
    _setup_slash_completion()

    api_key = config.GEMINI_API_KEY
    if not api_key:
        print(f"{C.YELLOW}🔑 No saved API key found.{C.RESET}")
        api_key = input(f"{C.BOLD}Enter GEMINI_API_KEY: {C.RESET}").strip()
        if not api_key:
            print(f"{C.RED}❌ No API key provided. Cannot continue.{C.RESET}")
            sys.exit(1)
        config.save_api_key(api_key)
        print(f"{C.GREEN}✅ Key saved to: {config.API_KEY_FILE}{C.RESET}")
        print(f"{C.DIM}   (you won't be asked again next time){C.RESET}\n")

    try:
        agent = GeminiAgent(api_key=api_key)
    except Exception as e:
        print(f"{C.RED}❌ Failed to initialize the Agent: {e}{C.RESET}")
        sys.exit(1)

    info_lines = [
        f"{C.DIM}📂 Workspace:{C.RESET} {config.WORKSPACE_DIR}",
        f"{C.DIM}🧠 Model:{C.RESET}     {C.BOLD}{agent.router.current_model_name}{C.RESET}",
        f"{C.DIM}Type /help for commands, or just{C.RESET} {C.CYAN}/{C.RESET} {C.DIM}for suggestions.{C.RESET}",
    ]
    print(draw_box("Session", info_lines, color=C.CYAN))
    print()

    current_puter_model: Optional[str] = None  # /model — session-only Gemini/Puter override, see handle_model_command()

    while True:
        try:
            prompt_model_tag = f" {C.DIM}[{current_puter_model} via Puter]{C.RESET}" if current_puter_model else ""
            user_input = _clean_user_input(input(f"{C.BOLD}{C.MAGENTA}You{C.RESET}{prompt_model_tag} {C.MAGENTA}›{C.RESET} "))
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.CYAN}👋 Goodbye!{C.RESET}")
            break

        if not user_input:
            continue

        # ---------- slash-command suggestions ----------
        # Checked defensively (not a strict "== '/'") in case readline or the
        # terminal inserts stray whitespace/control characters around a
        # lone '/' keystroke on some platforms.
        stripped = user_input.strip()
        if stripped == "/" or (stripped.startswith("/") and len(stripped) <= 2 and not any(c.isalnum() for c in stripped[1:])):
            print_slash_suggestions()
            continue

        # ---------- special commands ----------
        if user_input in ("/exit", "/quit"):
            print(f"{C.CYAN}👋 Goodbye!{C.RESET}")
            break

        if user_input == "/clear":
            print("\033[2J\033[H", end="")  # ANSI clear screen + move cursor home
            print_banner()
            continue

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/memory":
            mem = agent.recall_all()
            if not mem:
                print(f"{C.DIM}🧠 No saved information yet.{C.RESET}")
            else:
                lines = [
                    f"{C.CYAN}{k}{C.RESET}: {v['value']}  {C.DIM}({v['category']}){C.RESET}"
                    for k, v in mem.items()
                ]
                print(draw_box("Long-term memory", lines, color=C.PINK))
            continue

        if user_input.startswith("/remember "):
            payload = user_input[len("/remember "):].strip()
            if "=" not in payload:
                print(f"{C.YELLOW}⚠️  Correct syntax: /remember key=value{C.RESET}")
                continue
            k, v = payload.split("=", 1)
            agent.remember(k.strip(), v.strip())
            print(f"{C.GREEN}✅ Saved: {k.strip()} = {v.strip()}{C.RESET}")
            continue

        if user_input.startswith("/forget "):
            key = user_input[len("/forget "):].strip()
            agent.memory.forget(key)
            print(f"{C.YELLOW}🗑️  Deleted '{key}' from memory (if it existed).{C.RESET}")
            continue

        if user_input == "/undo":
            result = tools.undo_last_change()
            print(f"{C.YELLOW}{result}{C.RESET}")
            continue

        if user_input == "/tree":
            print_tree()
            continue

        if user_input == "/ps":
            listing = tools.list_background_processes()
            print(draw_box("Background processes", listing.split("\n"), color=C.ORANGE))
            continue

        if user_input == "/log":
            entries = agent.recent_actions()
            if not entries:
                print(f"{C.DIM}ℹ️  No actions taken yet this session.{C.RESET}")
            else:
                # draw_box sizes the box to the terminal width and wraps
                # each line inside "│ " + content + " │" — so the actual
                # budget for arbitrary content (tool args, result text) has
                # to account for everything else already on that line
                # (timestamp, status icon, tool name, parens) too, not just
                # be some arbitrary flat number. Using a fixed length here
                # without that accounting was the actual bug: the fixed
                # truncation point was too generous once the tool
                # name/timestamp/etc. were added back on top of it, so the
                # combined line still overflowed the box width.
                from colors import term_width
                box_inner_width = term_width() - 4  # matches draw_box's own inner-width math
                MAX_RESULT_LINE_LEN = max(20, box_inner_width - 20)  # reserve ~20 cols for the prefix

                lines = []
                for e in entries:
                    import time as _time
                    when = _time.strftime("%H:%M:%S", _time.localtime(e["ts"]))
                    status_color = C.GREEN if e["success"] else C.RED
                    status_icon = "✓" if e["success"] else "✗"
                    args_str = ", ".join(f"{k}={v}" for k, v in e.get("args", {}).items())
                    # Budget for args_str: box width minus space already used
                    # by "[HH:MM:SS] ✓ tool_name(" and the closing ")".
                    prefix_len = len(f"[{when}] {status_icon} {e['tool']}(")
                    args_budget = max(10, box_inner_width - prefix_len - 1)
                    if len(args_str) > args_budget:
                        args_str = args_str[:args_budget - 3] + "..."
                    lines.append(
                        f"{C.DIM}[{when}]{C.RESET} {status_color}{status_icon}{C.RESET} "
                        f"{C.CYAN}{e['tool']}{C.RESET}({args_str})"
                    )
                    result_str = str(e["result"]).replace("\n", " ")
                    result_budget = max(10, box_inner_width - 6)  # "    → " prefix is 6 chars
                    if len(result_str) > result_budget:
                        result_str = result_str[:result_budget - 3] + "..."
                    lines.append(f"    {C.DIM}→ {result_str}{C.RESET}")
                print(draw_box("Execution log", lines, color=C.ORANGE))
            continue

        if user_input == "/clearlog":
            agent.clear_execution_log()
            print(f"{C.YELLOW}🗑️  Execution log cleared.{C.RESET}")
            continue

        if user_input == "/image":
            image_paths = _prompt_for_images()
            if not image_paths:
                print(f"{C.DIM}ℹ️  No valid images provided — nothing sent.{C.RESET}")
                continue
            message = input(f"{C.BOLD}{C.MAGENTA}Message about the image(s){C.RESET} "
                             f"{C.DIM}(optional, Enter for a general description){C.RESET} "
                             f"{C.MAGENTA}›{C.RESET} ").strip()
            if not message:
                message = "Describe these images." if len(image_paths) > 1 else "Describe this image."
            _send_and_print(agent, message, image_paths=image_paths, puter_model=current_puter_model)
            continue

        if user_input == "/force_review":
            review_message = _build_force_review_message()
            if review_message is None:
                print(f"{C.DIM}ℹ️  Workspace is empty — nothing to review.{C.RESET}")
                continue
            file_count = review_message.count("\n- ")
            print(f"{C.YELLOW}🔍 Forcing a full review of {file_count} file(s)... "
                  f"this may take a while and use more tokens than usual.{C.RESET}")
            _send_and_print(agent, review_message, puter_model=current_puter_model)
            continue

        if user_input == "/multi-agent":
            config.MULTI_AGENT_ENABLED = not config.MULTI_AGENT_ENABLED
            _persist_current_settings()
            state = f"{C.GREEN}ON{C.RESET}" if config.MULTI_AGENT_ENABLED else f"{C.DIM}OFF{C.RESET}"
            lines = [f"Multi-agent mode: {state}"]
            if config.MULTI_AGENT_ENABLED:
                lines.append("")
                lines.append(f"{C.DIM}Simple requests still get a direct answer; complex ones{C.RESET}")
                lines.append(f"{C.DIM}get a plan -> execute -> review pipeline. Current roles:{C.RESET}")
                for role, model in config.MULTI_AGENT_ROLES.items():
                    lines.append(f"  {C.CYAN}{role:<12}{C.RESET} {model}")
                lines.append("")
                lines.append(f"{C.DIM}Change role models with /settings.{C.RESET}")
            print(draw_box("Multi-agent", lines, color=C.VIOLET))
            continue

        if user_input == "/settings":
            print_settings_menu(agent)
            continue

        if user_input.startswith("/settings "):
            handle_settings_subcommand(agent, user_input[len("/settings "):])
            continue

        if user_input == "/model" or user_input.startswith("/model "):
            model_args = user_input[len("/model"):].strip()
            current_puter_model = handle_model_command(agent, model_args, current_puter_model)
            continue

        if user_input == "/stats":
            print(agent.usage_report())
            continue

        if user_input == "/workspace":
            print(f"{C.BLUE}📂 {config.WORKSPACE_DIR}{C.RESET}")
            continue

        if user_input == "/resetkey":
            if config.API_KEY_FILE.exists():
                config.API_KEY_FILE.unlink()
                print(f"{C.YELLOW}🗑️  Saved API key deleted. Restart the program to enter a new one.{C.RESET}")
            else:
                print(f"{C.DIM}ℹ️  No saved key exists.{C.RESET}")
            continue

        if user_input == "/keys":
            print_keys_menu(agent)
            continue

        if user_input.startswith("/keys "):
            handle_keys_subcommand(agent, user_input[len("/keys "):])
            continue

        if user_input == "/puterJS":
            handle_puterjs_command()
            continue

        if user_input == "/free":
            handle_free_command(agent)
            continue

        if user_input == "/free-puter-models-only" or user_input.startswith("/free-puter-models-only "):
            free_args = user_input[len("/free-puter-models-only"):].strip()
            current_puter_model = handle_free_puter_models_only(free_args)
            continue

        if user_input == "/deepresearch" or user_input.startswith("/deepresearch "):
            query = user_input[len("/deepresearch"):].strip()
            handle_deepresearch_command(agent, query)
            continue

        # ---------- unrecognized slash command ----------
        # Anything starting with '/' that didn't match a known command above
        # (including a lone '/' that somehow slipped past the earlier check,
        # e.g. due to a readline quirk on some terminals) shows suggestions
        # instead of being sent to the model as a literal message.
        if user_input.startswith("/"):
            print(f"{C.YELLOW}Unknown command: {user_input}{C.RESET}")
            print_slash_suggestions()
            continue

        # ---------- regular message sent to the Agent (streamed) ----------
        tb.set_indeterminate()
        _send_and_print(agent, user_input, puter_model=current_puter_model)


if __name__ == "__main__":
    main()
