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
from agent import GeminiAgent
from colors import C, draw_box
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
    "/multi-agent", "/settings",
    "/stats", "/workspace", "/resetkey", "/exit", "/quit",
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
    "/stats": "model usage report and switches",
    "/workspace": "show the workspace path",
    "/resetkey": "delete the saved API key",
    "/exit": "quit the program",
    "/quit": "quit the program",
}


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
        ("/stats", "model usage report and automatic switches"),
        ("/workspace", "show the workspace path"),
        ("/resetkey", "delete the saved API key"),
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

    ensure_prefix()  # in case the reply was empty / only tool calls happened

    if line_buffer:
        if not first_line:
            print()
        print(render_markdown(line_buffer), end="", flush=True)

    print("\n")
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
                status.start()
                continue

            if event.kind == "step_start":
                status.stop()
                n = event.data["step_number"]
                total = event.data["total_steps"]
                current_step_number = n
                step_action_lines[n] = []
                print(f"{C.BOLD}{C.CYAN}Plan {n}:{C.RESET}")
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
        print(f"\n{C.RED}❌ Multi-agent turn failed: {e}{C.RESET}")
        print(f"{C.DIM}(Any steps completed above were still carried out and are not undone.){C.RESET}\n")
        return "".join(full_text_parts)
    finally:
        status.stop()

    ensure_reply_prefix()
    if line_buffer:
        if not first_text_line:
            print()
        print(render_markdown(line_buffer), end="", flush=True)

    print("\n")
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
    lines.append(f"{C.DIM}To change things:{C.RESET}")
    lines.append(f"  {C.CYAN}/settings models{C.RESET}          list every model your API key can use")
    lines.append(f"  {C.CYAN}/settings role <role> <model>{C.RESET}  assign a model to a role, e.g.")
    lines.append(f"                                 /settings role planner gemini-2.5-pro")
    lines.append(f"  {C.CYAN}/settings chain <m1,m2,...>{C.RESET}    replace the failover chain order")

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

    print(f"{C.YELLOW}⚠️  Unknown /settings subcommand '{sub}'. Try /settings for the menu.{C.RESET}")


def _send_and_print(agent, message: str, image_paths: list = None):
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
    """
    status = LiveStatusLine()
    tools.set_activity_callback(make_activity_printer(status))
    agent.router.set_status_callback(status.set_activity)
    status.start()

    if config.MULTI_AGENT_ENABLED and not image_paths:
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
        print_reply_streaming(agent.send_stream(message, image_paths=image_paths), status)
    except RuntimeError as e:
        status.stop()
        print(f"\n{C.RED}❌ All models failed: {e}{C.RESET}\n")
    except Exception as e:
        status.stop()
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

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.MAGENTA}You{C.RESET} {C.MAGENTA}›{C.RESET} ").strip()
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
            _send_and_print(agent, message, image_paths=image_paths)
            continue

        if user_input == "/force_review":
            review_message = _build_force_review_message()
            if review_message is None:
                print(f"{C.DIM}ℹ️  Workspace is empty — nothing to review.{C.RESET}")
                continue
            file_count = review_message.count("\n- ")
            print(f"{C.YELLOW}🔍 Forcing a full review of {file_count} file(s)... "
                  f"this may take a while and use more tokens than usual.{C.RESET}")
            _send_and_print(agent, review_message)
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
        _send_and_print(agent, user_input)


if __name__ == "__main__":
    main()
