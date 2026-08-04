"""
Tools/functions the model uses via function calling to work with files.
All operations are restricted to WORKSPACE_DIR to prevent access to files
outside the sandboxed environment (security).
"""
from pathlib import Path
from typing import List, Callable, Optional
import difflib
import fnmatch
import json
import os
import re
import shutil
import subprocess
import time

import config

# ---------------- shared Gemini client for image tools ----------------
# Image_Fetch and Image_Create both need to call the Gemini API directly
# (vision input / image-output generation respectively), which is a
# different concern from every other tool here (pure local file/process
# operations). Rather than have agent.py wire a client through every tool
# call, we lazily build one client here on first use, keyed off the same
# saved API key the rest of the app uses.
_image_client = None


def _get_image_client():
    global _image_client
    if _image_client is None:
        from google import genai
        api_key = config.GEMINI_API_KEY or config.load_saved_api_key()
        if not api_key:
            raise RuntimeError("No API key available for image tools.")
        _image_client = genai.Client(api_key=api_key)
    return _image_client


# Models used specifically for image tools (separate from the text
# MODEL_CHAIN in config.py, since not every chat model supports vision
# input or image-generation output the same way).
IMAGE_UNDERSTANDING_MODEL = "gemini-2.5-flash"  # any current Flash/Pro model handles vision input fine
# Image generation model chain now lives in config.IMAGE_MODEL_CHAIN (with
# automatic fallback across the list) — see Image_Create below.

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_IMAGE_MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}

# Command timeout in seconds — prevents a hung/long-running process from
# blocking the agent forever. Only applies to commands NOT detected as
# long-running servers (see _looks_like_server_command below); those are
# started in the background instead of being waited on.
COMMAND_TIMEOUT_SECONDS = 60

# Substrings that are always blocked regardless of the command, since they
# are common ways to escape the sandbox or destroy the host system. This is
# a defense-in-depth measure, not a full sandbox — treat run_command as
# "trusted user, but let's not shoot ourselves in the foot" rather than a
# hard security boundary against a malicious model.
BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf /*", ":(){ :|:& };:",  # fork bomb
    "mkfs", "dd if=", "> /dev/sda", "shutdown", "reboot",
    "sudo ", "su -", "chmod -R 777 /", "chown -R",
]

# Command patterns that almost always start a long-running server / dev
# process (blocking indefinitely, e.g. until Ctrl+C) rather than completing
# and returning. Running these with subprocess.run() would hang until the
# COMMAND_TIMEOUT_SECONDS timeout every single time, freezing the whole
# Agent. Instead, run_command detects these and launches them in the
# background (non-blocking), returning immediately with the PID and a log
# file the model/user can check.
SERVER_COMMAND_PATTERNS = [
    r"\bnpm\s+(run\s+)?(dev|start|serve)\b",
    r"\byarn\s+(dev|start|serve)\b",
    r"\bpnpm\s+(dev|start|serve)\b",
    r"\bnext\s+dev\b",
    r"\bvite\b(?!\s+build)",
    r"\bflask\s+run\b",
    r"\bpython[3]?\s+.*\bapp\.py\b",
    r"\bpython[3]?\s+.*manage\.py\s+runserver\b",
    r"\bpython[3]?\s+-m\s+http\.server\b",
    r"\bpython[3]?\s+-m\s+flask\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bnodemon\b",
    r"\bnode\s+.*(server|index)\.js\b",
    r"\bphp\s+-S\b",
    r"\bhttp-server\b",
    r"\bserve\s+-s\b",
    r"\bng\s+serve\b",
    r"\brails\s+s(erver)?\b",
    r"-{1,2}watch\b",
]


def _looks_like_server_command(command: str) -> bool:
    return any(re.search(pattern, command, re.IGNORECASE) for pattern in SERVER_COMMAND_PATTERNS)


# ---------------- background process registry ----------------
# Tracks processes started via run_command's background path, so they can be
# listed/stopped later (e.g. via a stop_background_process tool or /ps in
# the CLI). Kept in-memory (per Agent run) plus mirrored to a small JSON file
# so `list_background_processes` reflects reality even after a restart.
BACKGROUND_LOG_DIR = config.WORKSPACE_DIR / ".undo_history" / "bg_logs"
BACKGROUND_REGISTRY_FILE = config.WORKSPACE_DIR / ".undo_history" / "bg_processes.json"
_background_processes: dict = {}  # pid -> subprocess.Popen (only for this process's lifetime)

# Default ignore patterns applied even without a .agentignore file, since
# these directories are huge/noisy and almost never useful to scan/list.
DEFAULT_IGNORE_PATTERNS = [
    "node_modules", "node_modules/*", ".git", ".git/*",
    "__pycache__", "__pycache__/*", "*.pyc",
    ".venv", ".venv/*", "venv", "venv/*",
    ".undo_history", ".undo_history/*",  # our own undo backups shouldn't show up in listings
    ".agentignore",  # the ignore file itself is metadata, not project content
]

# ---------------- undo journal ----------------
UNDO_DIR = config.WORKSPACE_DIR / ".undo_history"
UNDO_LOG = UNDO_DIR / "log.jsonl"

# ---------------- live activity hook ----------------
# The CLI registers a callback here so tools can report what they're doing in
# real time (e.g. "Creating... main.py" then "Created main.py"), independent
# of the model's own text reply. This lets the terminal show file activity
# even during a plain streaming reply, similar to how Gemini CLI / editors
# show live file-change notifications.
_activity_callback: Optional[Callable[[str, str], None]] = None


def set_activity_callback(callback: Optional[Callable[[str, str], None]]):
    """
    Registers a function called as activity_callback(stage, path) whenever a
    file tool starts or finishes an operation. `stage` is one of:
    "creating", "created", "editing", "edited", "deleting", "deleted",
    "running", "ran". Pass None to disable.
    """
    global _activity_callback
    _activity_callback = callback


def _notify(stage: str, path: str):
    if _activity_callback:
        try:
            _activity_callback(stage, path)
        except Exception:
            pass  # never let a UI hook crash a tool call


def _load_ignore_patterns() -> List[str]:
    """Reads .agentignore (gitignore-style, one glob pattern per line) from
    the workspace root if present, merged with sensible defaults."""
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore_file = config.WORKSPACE_DIR / ".agentignore"
    if ignore_file.exists():
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(rel_path: Path, patterns: List[str]) -> bool:
    rel_str = str(rel_path)
    parts = rel_path.parts
    for pattern in patterns:
        if fnmatch.fnmatch(rel_str, pattern):
            return True
        # also match if any path component matches a bare-name pattern
        # (e.g. "node_modules" should match "src/node_modules/foo.js")
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        # a "dir/*" pattern should also hide the directory entry itself,
        # not just its contents (so empty/ignored dirs don't clutter listings)
        if pattern.endswith("/*") and fnmatch.fnmatch(rel_str, pattern[:-2]):
            return True
    return False


def _record_undo(action: str, path: str, previous_content: Optional[str], existed: bool):
    """
    Saves a snapshot of a file's previous state before create/edit/delete, so
    it can be restored later with undo_last_change(). Snapshots are stored as
    plain files under .undo_history, indexed by a JSONL log (newest last).
    """
    UNDO_DIR.mkdir(exist_ok=True)
    snapshot_id = f"{int(time.time() * 1000)}_{abs(hash(path)) % 100000}"
    snapshot_path = UNDO_DIR / snapshot_id
    if existed and previous_content is not None:
        snapshot_path.write_text(previous_content, encoding="utf-8")

    entry = {
        "ts": time.time(),
        "action": action,      # "create" | "edit" | "delete"
        "path": path,
        "existed_before": existed,
        "snapshot_id": snapshot_id if (existed and previous_content is not None) else None,
    }
    with open(UNDO_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def undo_last_change() -> str:
    """
    Reverts the most recent file change made by create_file, edit_file, or
    delete_file. If the file didn't exist before that change, undo deletes it
    (undoing a creation). If it existed, undo restores its previous content.
    Can be called multiple times to step back through several changes.
    """
    if not UNDO_LOG.exists():
        return "ℹ️ No changes to undo."

    lines = UNDO_LOG.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "ℹ️ No changes to undo."

    last_line = lines[-1]
    entry = json.loads(last_line)
    remaining = lines[:-1]
    UNDO_LOG.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")

    path = entry["path"]
    try:
        target = _safe_path(path)
    except PermissionError as e:
        return f"❌ {e}"

    if not entry["existed_before"]:
        # The file didn't exist before this change (it was newly created) -> delete it
        if target.exists():
            target.unlink()
        return f"↩️ Undid creation of '{path}' (file removed)."
    else:
        # Restore previous content from the snapshot
        snapshot_id = entry.get("snapshot_id")
        if not snapshot_id:
            return f"⚠️ No snapshot available to restore '{path}'."
        snapshot_path = UNDO_DIR / snapshot_id
        if not snapshot_path.exists():
            return f"⚠️ Snapshot for '{path}' is missing, cannot restore."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(snapshot_path.read_text(encoding="utf-8"), encoding="utf-8")
        return f"↩️ Restored '{path}' to its previous content."


def _safe_path(relative_path: str) -> Path:
    """
    Converts a relative path into an absolute path inside WORKSPACE_DIR, and
    rejects any attempt to escape it (e.g. ../../etc/passwd) to protect the
    host system.
    """
    target = (config.WORKSPACE_DIR / relative_path).resolve()
    workspace = config.WORKSPACE_DIR.resolve()
    if workspace not in target.parents and target != workspace:
        raise PermissionError(
            f"Access denied: '{relative_path}' attempts to escape the allowed workspace."
        )
    return target


def create_file(path: str, content: str) -> str:
    """
    Creates a new file with the given content inside the workspace. Missing
    parent directories in the path are created automatically. If a file
    already exists at that path, its previous content is snapshotted first so
    the change can be undone with undo_last_change().

    Args:
        path: relative file path (e.g. "src/main.py")
        content: file content as text
    """
    _notify("creating", path)
    p = _safe_path(path)

    existed = p.exists()
    previous_content = p.read_text(encoding="utf-8", errors="replace") if existed else None
    _record_undo("create", path, previous_content, existed)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _notify("created", path)
    return f"✅ File created: {path} ({len(content)} chars)"


def read_file(path: str) -> str:
    """
    Reads the contents of an existing file inside the workspace.

    Args:
        path: relative file path
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    if not p.is_file():
        return f"❌ Path is not a file: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    Replaces the first match of some text (old_text) with new text (new_text)
    inside an existing file. Useful for precise edits instead of rewriting
    the whole file. The previous content is snapshotted first so the change
    can be undone with undo_last_change().

    Args:
        path: relative file path
        old_text: the text to replace (must be unique in the file)
        new_text: the new text
    """
    _notify("editing", path)
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    content = p.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        return f"❌ Text to replace was not found in {path}"
    if count > 1:
        return f"⚠️ Text appears {count} times in {path}. Use a more specific snippet to avoid ambiguity."

    _record_undo("edit", path, content, existed=True)

    new_content = content.replace(old_text, new_text, 1)
    p.write_text(new_content, encoding="utf-8")
    _notify("edited", path)
    return f"✅ File edited: {path}"


def delete_file(path: str) -> str:
    """
    Deletes an existing file inside the workspace. The content is snapshotted
    first so the deletion can be undone with undo_last_change().

    Args:
        path: relative file path
    """
    _notify("deleting", path)
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"

    previous_content = p.read_text(encoding="utf-8", errors="replace")
    _record_undo("delete", path, previous_content, existed=True)

    p.unlink()
    _notify("deleted", path)
    return f"🗑️ File deleted: {path}"


def list_files(path: str = ".") -> str:
    """
    Lists files and directories under a given path inside the workspace.
    Respects .agentignore (if present) plus sensible defaults (node_modules,
    .git, __pycache__, venvs, etc. are always skipped).

    Args:
        path: relative directory path (defaults to the workspace root)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ Path not found: {path}"
    if not p.is_dir():
        return f"❌ Path is not a directory: {path}"

    patterns = _load_ignore_patterns()
    entries = sorted(p.rglob("*"))
    lines = []
    for e in entries:
        rel = e.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        kind = "📁" if e.is_dir() else "📄"
        lines.append(f"{kind} {rel}")
    if not lines:
        return "📁 Directory is empty (or everything in it is ignored via .agentignore)"
    return "\n".join(lines)


def search_in_files(query: str, extensions: List[str] = None) -> str:
    """
    Searches for a given text across all files in the workspace (bulk scan)
    and returns results grouped by file, with the full relative path of each
    matching file and the line number(s)/content where the text was found.
    Respects .agentignore (if present) plus sensible defaults.

    Args:
        query: the text to search for
        extensions: optional list of extensions to restrict the search
            (e.g. [".py", ".md"])
    """
    patterns = _load_ignore_patterns()
    matches_by_file: dict = {}  # rel path (str) -> list of "line N: content"
    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        if extensions and p.suffix not in extensions:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches_by_file.setdefault(str(rel), []).append(f"  line {i}: {line.strip()}")

    if not matches_by_file:
        return f"🔍 No results for '{query}'"

    total_matches = sum(len(v) for v in matches_by_file.values())
    lines = [f"🔍 Found '{query}' in {len(matches_by_file)} file(s), {total_matches} total match(es):\n"]
    shown = 0
    for rel_path, file_matches in matches_by_file.items():
        if shown >= 100:
            lines.append(f"... and more (truncated at 100 total matches)")
            break
        full_path = str((config.WORKSPACE_DIR / rel_path))
        lines.append(f"📄 {rel_path}  (path: {full_path})")
        for m in file_matches:
            if shown >= 100:
                break
            lines.append(m)
            shown += 1
        lines.append("")
    return "\n".join(lines).rstrip()


def find_file(pattern: str) -> str:
    """
    Searches for files by name across the whole workspace, matching a
    filename or glob pattern (e.g. "main.py", "*.test.js", "config*").
    Returns the relative and full path of every match. Respects
    .agentignore plus sensible defaults (node_modules, .git, etc. skipped).

    Args:
        pattern: filename or glob pattern to match against filenames
            (matched case-insensitively; wrap in *...* automatically if no
            wildcard is given, so "main" also finds "main.py")
    """
    ignore_patterns = _load_ignore_patterns()
    search_pattern = pattern if any(ch in pattern for ch in "*?[") else f"*{pattern}*"

    matches = []
    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, ignore_patterns):
            continue
        if fnmatch.fnmatch(p.name.lower(), search_pattern.lower()):
            matches.append(f"📄 {rel}  (path: {p})")

    if not matches:
        return f"🔍 No files found matching '{pattern}'."
    return f"Found {len(matches)} file(s):\n" + "\n".join(matches[:100])


def find_folder(pattern: str) -> str:
    """
    Searches for folders by name across the whole workspace, matching a
    folder name or glob pattern (e.g. "components", "test*"). Returns the
    relative and full path of every match. Respects .agentignore plus
    sensible defaults (node_modules, .git, etc. skipped).

    Args:
        pattern: folder name or glob pattern to match (matched
            case-insensitively; wrap in *...* automatically if no wildcard
            is given)
    """
    ignore_patterns = _load_ignore_patterns()
    search_pattern = pattern if any(ch in pattern for ch in "*?[") else f"*{pattern}*"

    matches = []
    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_dir():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, ignore_patterns):
            continue
        if fnmatch.fnmatch(p.name.lower(), search_pattern.lower()):
            matches.append(f"📁 {rel}  (path: {p})")

    if not matches:
        return f"🔍 No folders found matching '{pattern}'."
    return f"Found {len(matches)} folder(s):\n" + "\n".join(matches[:100])


def diff_preview(path: str, new_content: str) -> str:
    """
    Shows a preview diff between a file's current content and a proposed new
    content, without actually applying the change. Useful before large edits.

    Args:
        path: relative file path
        new_content: the proposed new content
    """
    p = _safe_path(path)
    old_content = p.read_text(encoding="utf-8") if p.exists() else ""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    result = "".join(diff)
    return result if result else "No differences"


def _load_bg_registry() -> dict:
    if BACKGROUND_REGISTRY_FILE.exists():
        try:
            return json.loads(BACKGROUND_REGISTRY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_bg_registry(registry: dict):
    BACKGROUND_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    BACKGROUND_REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_pid_alive(pid: int) -> bool:
    try:
        import os
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        return False


def start_background_process(command: str, working_dir: str = ".") -> str:
    """
    Starts a long-running command (e.g. a dev server like "npm run dev",
    "flask run", "uvicorn main:app") in the background instead of waiting for
    it to finish — since these commands run indefinitely, waiting on them
    would hang forever. Output is redirected to a log file you can inspect
    with read_background_log. Returns the process ID (PID) so you can stop it
    later with stop_background_process.

    Args:
        command: the shell command to run (e.g. "npm run dev")
        working_dir: relative directory inside the workspace to run it from
    """
    lowered = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lowered:
            return f"🚫 Command blocked for safety (matched pattern: '{pattern}')."

    try:
        cwd = _safe_path(working_dir)
    except PermissionError as e:
        return f"❌ {e}"
    if not cwd.exists() or not cwd.is_dir():
        return f"❌ Working directory not found: {working_dir}"

    BACKGROUND_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = BACKGROUND_LOG_DIR / f"{ts}.log"

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    except Exception as e:
        return f"❌ Failed to start background process: {e}"

    _background_processes[proc.pid] = proc
    registry = _load_bg_registry()
    registry[str(proc.pid)] = {
        "command": command,
        "working_dir": working_dir,
        "started_at": ts,
        "log_file": str(log_path.relative_to(config.WORKSPACE_DIR)),
    }
    _save_bg_registry(registry)

    return (
        f"🚀 Started in background (PID {proc.pid}): {command}\n"
        f"This looks like a long-running server, so it won't block — it keeps "
        f"running after this tool returns.\n"
        f"Log file: {log_path.relative_to(config.WORKSPACE_DIR)}\n"
        f"Use read_background_log(pid={proc.pid}) to check its output, or "
        f"stop_background_process(pid={proc.pid}) to stop it."
    )


def list_background_processes() -> str:
    """Lists all background processes started via run_command/
    start_background_process, showing whether each is still running."""
    registry = _load_bg_registry()
    if not registry:
        return "ℹ️ No background processes have been started."
    lines = []
    for pid_str, info in registry.items():
        pid = int(pid_str)
        alive = _is_pid_alive(pid)
        status = "🟢 running" if alive else "⚪ stopped"
        lines.append(f"PID {pid} [{status}]: {info['command']}  (log: {info['log_file']})")
    return "\n".join(lines)


def read_background_log(pid: int, tail_lines: int = 50) -> str:
    """
    Reads the output log of a background process started via run_command or
    start_background_process. Useful for checking if a dev server started
    successfully, which port it's on, or whether it crashed.

    Args:
        pid: the process ID returned when the background process was started
        tail_lines: how many of the most recent log lines to return (default 50)
    """
    registry = _load_bg_registry()
    info = registry.get(str(pid))
    if not info:
        return f"❌ No known background process with PID {pid}."
    log_path = config.WORKSPACE_DIR / info["log_file"]
    if not log_path.exists():
        return f"❌ Log file missing for PID {pid}."
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-tail_lines:]
    alive = _is_pid_alive(pid)
    status = "🟢 still running" if alive else "⚪ process has stopped"
    return f"Status: {status}\n--- last {len(tail)} line(s) of output ---\n" + "\n".join(tail)


def stop_background_process(pid: int) -> str:
    """
    Stops a background process (e.g. a dev server) previously started by
    run_command or start_background_process.

    Args:
        pid: the process ID to stop
    """
    import os
    import signal

    if not _is_pid_alive(pid):
        return f"ℹ️ PID {pid} is not running (already stopped)."
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.3)
        if _is_pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        return f"🛑 Stopped background process PID {pid}."
    except Exception as e:
        return f"❌ Failed to stop PID {pid}: {e}"


def wait_process(pid: int, timeout: int = 60, poll_interval: float = 1.0) -> str:
    """
    Waits (blocks) until a background process finishes, or until the timeout
    is reached — whichever comes first. Useful when a later step depends on
    the background process actually being done (e.g. "run the build, then
    wait for it to finish, then check the output") instead of just firing it
    and moving on.

    Args:
        pid: the process ID to wait for (as returned by start_background_process
             or run_command's background path)
        timeout: max seconds to wait before giving up and returning control
                 (default 60)
        poll_interval: seconds between liveness checks (default 1.0)
    """
    registry = _load_bg_registry()
    if str(pid) not in registry and pid not in _background_processes:
        return f"❌ No known background process with PID {pid}."

    if not _is_pid_alive(pid):
        return f"✅ PID {pid} has already finished (nothing to wait for)."

    waited = 0.0
    while waited < timeout:
        if not _is_pid_alive(pid):
            info = registry.get(str(pid), {})
            log_hint = f" (log: {info['log_file']})" if info.get("log_file") else ""
            return f"✅ PID {pid} finished after ~{waited:.1f}s.{log_hint} Use read_background_log({pid}) to see its output."
        time.sleep(poll_interval)
        waited += poll_interval

    return (
        f"⏳ Timed out after {timeout}s — PID {pid} is still running. "
        f"Call wait_process again to keep waiting, or use read_background_log({pid}) "
        f"to check progress so far."
    )


import platform
import shlex

# ---------------- Unix command compatibility (for Windows) ----------------
# On Windows, subprocess.run(cmd, shell=True) invokes cmd.exe, which has no
# idea what `ls`, `grep`, `cp`, `cat`, `rm`, `mv`, `touch`, `mkdir -p`, `head`,
# `tail`, `wc`, `find`, or `pwd` are — commands the model (trained mostly on
# Unix-style shells) will very naturally reach for. Rather than have every
# such command just fail with "'grep' is not recognized...", we detect
# Windows and rewrite the small set of common Unix commands into a
# PowerShell invocation, since PowerShell (available by default on Windows
# 10+) already provides real equivalents for nearly all of these — this
# gives the model working `ls`/`grep`/`cp`/etc. without needing WSL/Git Bash
# installed.
IS_WINDOWS = platform.system() == "Windows"


def _translate_unix_command_for_windows(command: str) -> str:
    """
    If running on Windows and the command's first word is a common Unix
    command we know how to translate, rewrites it into a working PowerShell
    invocation — including converting the most common Unix flags (-r, -f,
    -la, -i, etc.) into their PowerShell equivalents, since passing raw Unix
    flags straight through to a PowerShell cmdlet would just fail (PowerShell
    parameters use entirely different names/syntax, e.g. -Recurse not -r).
    Otherwise returns the command unchanged.

    This is a best-effort convenience shim, not a full POSIX shell emulator
    — complex pipelines, globs, or uncommon flag combinations may not
    translate perfectly. It's meant to make the most common single commands
    (listing files, copying, searching text, removing/creating things) work
    out of the box on Windows without requiring WSL or Git Bash.
    """
    if not IS_WINDOWS:
        return command

    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return command  # unbalanced quotes etc. — let it fail naturally downstream
    if not tokens:
        return command

    first = tokens[0].lower()
    rest_tokens = tokens[1:]

    if first == "pwd":
        return 'powershell -NoProfile -NonInteractive -Command "Get-Location"'
    if first == "clear":
        return 'powershell -NoProfile -NonInteractive -Command "Clear-Host"'

    if first == "ls":
        flags, paths = _split_flags(rest_tokens, {"l", "a", "h", "R"})
        cmd = "Get-ChildItem"
        if "R" in flags:
            cmd += " -Recurse"
        if "a" not in flags:
            pass  # Get-ChildItem already shows hidden files differently; not worth over-engineering
        if paths:
            cmd += " " + " ".join(paths)
        return f'powershell -NoProfile -NonInteractive -Command "{cmd}"'

    if first == "cat":
        if not rest_tokens:
            return command
        return f'powershell -NoProfile -NonInteractive -Command "Get-Content {" ".join(rest_tokens)}"'

    if first == "cp":
        flags, paths = _split_flags(rest_tokens, {"r", "R", "f"})
        if len(paths) < 2:
            return command
        recurse = " -Recurse" if ("r" in flags or "R" in flags) else ""
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Copy-Item{recurse} -Force {paths[0]} {paths[1]}"'
        )

    if first == "mv":
        flags, paths = _split_flags(rest_tokens, {"f"})
        if len(paths) < 2:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Move-Item -Force {paths[0]} {paths[1]}"'
        )

    if first == "rm":
        flags, paths = _split_flags(rest_tokens, {"r", "R", "f"})
        if not paths:
            return command
        recurse = " -Recurse" if ("r" in flags or "R" in flags) else ""
        force = " -Force" if "f" in flags else ""
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Remove-Item{recurse}{force} {" ".join(paths)}"'
        )

    if first == "mkdir":
        flags, paths = _split_flags(rest_tokens, {"p"})
        if not paths:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"New-Item -ItemType Directory -Force {" ".join(paths)}"'
        )

    if first == "touch":
        if not rest_tokens:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"New-Item -ItemType File -Force {" ".join(rest_tokens)}"'
        )

    if first == "grep":
        flags, remaining = _split_flags(rest_tokens, {"r", "R", "i", "n", "v"})
        if not remaining:
            return command
        pattern = remaining[0]
        paths = remaining[1:]
        opts = []
        if "i" in flags:
            opts.append("-CaseSensitive:$false")
        recurse = " -Recurse" if ("r" in flags or "R" in flags) else ""
        path_arg = " ".join(paths) if paths else "*"
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Select-String -Pattern {pattern} -Path {path_arg}{recurse}"'
        )

    if first == "head":
        count, paths = _extract_dash_n(rest_tokens)
        if not paths:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Get-Content {paths[0]} | Select-Object -First {count}"'
        )

    if first == "tail":
        count, paths = _extract_dash_n(rest_tokens)
        if not paths:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Get-Content {paths[0]} | Select-Object -Last {count}"'
        )

    if first == "wc":
        _, paths = _split_flags(rest_tokens, {"l", "w", "c"})
        if not paths:
            return command
        return (
            f'powershell -NoProfile -NonInteractive -Command '
            f'"Get-Content {paths[0]} | Measure-Object -Line -Word -Character"'
        )

    if first == "which":
        if not rest_tokens:
            return command
        return f'powershell -NoProfile -NonInteractive -Command "Get-Command {rest_tokens[0]}"'

    if first == "find" and len(rest_tokens) >= 1:
        # Only handle the extremely common "find <path> -name <pattern>" form;
        # anything more complex is left untranslated (will error naturally).
        if len(rest_tokens) >= 3 and rest_tokens[1] == "-name":
            path, pattern = rest_tokens[0], rest_tokens[2]
            return (
                f'powershell -NoProfile -NonInteractive -Command '
                f'"Get-ChildItem -Recurse -Path {path} -Filter {pattern}"'
            )
        return command

    return command


def _extract_dash_n(tokens, default: int = 10):
    """
    Extracts a "-n <count>" (or "-<count>") argument from a token list, as
    used by head/tail (e.g. "head -n 5 file.txt" or "head -5 file.txt").
    Returns (count, remaining_non_flag_tokens).
    """
    count = default
    remaining = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-n" and i + 1 < len(tokens) and tokens[i + 1].lstrip("-").isdigit():
            count = int(tokens[i + 1])
            i += 2
            continue
        if tok.startswith("-") and tok[1:].isdigit():
            count = int(tok[1:])
            i += 1
            continue
        remaining.append(tok)
        i += 1
    return count, remaining


def _split_flags(tokens, known_flags: set):
    """
    Splits a token list into (set of single-letter flags seen, remaining
    non-flag arguments). Handles combined short flags like "-la" (splits
    into 'l' and 'a') as well as separate flags like "-r", "-f". Only
    recognizes flags in `known_flags`; anything else starting with '-' is
    left in the remaining-args list untouched (so unsupported flags don't
    silently vanish — the command will just likely fail visibly instead of
    behaving unexpectedly).
    """
    flags = set()
    remaining = []
    for tok in tokens:
        if tok.startswith("-") and len(tok) > 1 and all(c in known_flags for c in tok[1:]):
            flags.update(tok[1:])
        else:
            remaining.append(tok)
    return flags, remaining


def run_command(command: str, working_dir: str = ".") -> str:
    """
    Runs a shell command inside the project workspace and returns its output.
    Use this for things like installing dependencies (npm install, pip install),
    running scripts (python script.py), checking git status, running tests, etc.

    IMPORTANT: if the command looks like it starts a long-running server or
    dev process (npm run dev, flask run, uvicorn, vite, nodemon, etc.), it is
    automatically started in the BACKGROUND instead of being waited on — this
    prevents the agent from freezing forever waiting for a process that never
    exits on its own. In that case this returns immediately with the PID and
    a log file path (see read_background_log / stop_background_process).

    For normal one-shot commands, this runs synchronously with a timeout and
    returns their full stdout/stderr.

    The command runs with its working directory set inside the workspace and
    cannot access files outside it. A small blocklist rejects obviously
    destructive commands (e.g. rm -rf /, shutdown, fork bombs).

    Args:
        command: the shell command to run (e.g. "npm install", "python main.py", "git status")
        working_dir: relative directory inside the workspace to run the command from
            (defaults to the workspace root)
    """
    if _looks_like_server_command(command):
        _notify("running", command)
        result = start_background_process(command, working_dir)
        _notify("ran", command)
        return result

    _notify("running", command)
    lowered = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lowered:
            return f"🚫 Command blocked for safety (matched pattern: '{pattern}')."

    try:
        cwd = _safe_path(working_dir)
    except PermissionError as e:
        return f"❌ {e}"

    if not cwd.exists() or not cwd.is_dir():
        return f"❌ Working directory not found: {working_dir}"

    effective_command = _translate_unix_command_for_windows(command)

    try:
        result = subprocess.run(
            effective_command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"⏱️ Command timed out after {COMMAND_TIMEOUT_SECONDS}s: {command}\n"
            f"If this command is meant to run indefinitely (a server/watcher), "
            f"use start_background_process instead so it doesn't block."
        )
    except Exception as e:
        return f"❌ Failed to run command: {e}"

    _notify("ran", command)
    output_parts = [f"$ {command}"]
    if effective_command != command:
        output_parts.append(f"(translated for Windows: {effective_command})")
    output_parts.append(f"(exit code: {result.returncode})")
    if result.stdout:
        output_parts.append(f"--- stdout ---\n{result.stdout.strip()}")
    if result.stderr:
        output_parts.append(f"--- stderr ---\n{result.stderr.strip()}")
    return "\n".join(output_parts)


def move_file(source_path: str, destination_path: str) -> str:
    """
    Moves or renames a file within the workspace (e.g. moving it to a
    different folder, or renaming it in place). Missing destination parent
    directories are created automatically. The previous location's content
    is snapshotted first so this can be undone with undo_last_change().

    Args:
        source_path: the file's current relative path
        destination_path: the new relative path (rename if same folder, move if different)
    """
    _notify("editing", f"{source_path} → {destination_path}")
    src = _safe_path(source_path)
    dst = _safe_path(destination_path)

    if not src.exists():
        return f"❌ Source file not found: {source_path}"
    if not src.is_file():
        return f"❌ Source is not a file: {source_path}"
    if dst.exists():
        return f"⚠️ Destination already exists: {destination_path}. Choose a different name or delete it first."

    content = src.read_text(encoding="utf-8", errors="replace")
    # Record as a delete-of-source + create-of-destination so undo can
    # reverse either half; we log two undo entries (destination create, then
    # source delete) so a single undo_last_change() call moves it back.
    _record_undo("delete", source_path, content, existed=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    _record_undo("create", destination_path, None, existed=False)

    _notify("edited", f"{source_path} → {destination_path}")
    return f"📦 Moved: {source_path} → {destination_path}"


def rename_file(path: str, new_name: str) -> str:
    """
    Renames a file in place (keeps it in the same folder). Convenience
    wrapper around move_file for the common "just rename it" case.

    Args:
        path: the file's current relative path
        new_name: the new filename only (not a full path), e.g. "utils.py"
    """
    if "/" in new_name or "\\" in new_name:
        return "❌ new_name should be a filename only (no slashes) — use move_file to change folders too."
    destination_path = str(Path(path).parent / new_name) if Path(path).parent != Path(".") else new_name
    return move_file(path, destination_path)


def _ensure_undo_history_gitignored(cwd: Path):
    """
    Ensures the workspace's .gitignore excludes common noise (our own
    .undo_history snapshots, __pycache__, node_modules, venvs) so a plain
    'git add .' / git_commit doesn't accidentally track generated files.
    Only adds entries that aren't already present, and never overwrites an
    existing .gitignore's other content.
    """
    gitignore = cwd / ".gitignore"
    default_entries = [
        ".undo_history/",
        "__pycache__/",
        "*.pyc",
        "node_modules/",
        ".venv/",
        "venv/",
    ]
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    existing_lines = set(line.strip() for line in existing.splitlines())
    missing = [e for e in default_entries if e not in existing_lines and e.rstrip("/") not in existing_lines]
    if not missing:
        return
    new_content = existing.rstrip("\n") + ("\n" if existing.strip() else "") + "\n".join(missing) + "\n"
    gitignore.write_text(new_content.lstrip("\n"), encoding="utf-8")


def git_clone(repo_url: str, destination: str = None, depth: int = 1) -> str:
    """
    Clones a git repository into the workspace. Only HTTPS URLs are allowed
    (no SSH/git:// URLs) so cloning never hangs waiting for SSH key/passphrase
    prompts or asks for credentials interactively. Defaults to a shallow
    clone (depth=1, just the latest commit) to keep it fast and avoid
    accidentally pulling huge repository histories.

    Args:
        repo_url: HTTPS URL of the repository (e.g.
            "https://github.com/user/repo.git")
        destination: relative folder name to clone into inside the
            workspace (defaults to the repository's own name, derived from
            the URL)
        depth: how many commits of history to fetch (default 1 = shallow
            clone, fastest; pass a larger number or 0 for full history if
            you actually need it)
    """
    if not repo_url.startswith("https://"):
        return (
            "🚫 Only HTTPS repository URLs are allowed (e.g. "
            "https://github.com/user/repo.git) — SSH/git:// URLs are blocked "
            "since they can hang waiting for credential/key prompts."
        )

    if not destination:
        # Derive a folder name from the URL, e.g.
        # https://github.com/user/repo.git -> "repo"
        destination = repo_url.rstrip("/").rsplit("/", 1)[-1]
        if destination.endswith(".git"):
            destination = destination[:-4]
        if not destination:
            return "❌ Could not derive a destination folder name from that URL — please specify `destination` explicitly."

    try:
        dest_path = _safe_path(destination)
    except PermissionError as e:
        return f"❌ {e}"

    if dest_path.exists() and any(dest_path.iterdir()):
        return f"⚠️ Destination '{destination}' already exists and is not empty. Choose a different destination or remove it first."

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone"]
    if depth and depth > 0:
        cmd += ["--depth", str(depth)]
    cmd += [repo_url, str(dest_path)]

    git_env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",  # never wait for a credential prompt
        "GIT_ASKPASS": "true",        # if something still asks, "true" exits 0 immediately with no password
    }

    _notify("creating", destination)
    try:
        result = subprocess.run(
            cmd, cwd=str(config.WORKSPACE_DIR), capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS, env=git_env,
        )
    except subprocess.TimeoutExpired:
        # Clean up a partial clone so a retry doesn't hit the "not empty" check above.
        shutil.rmtree(dest_path, ignore_errors=True)
        return f"⏱️ git clone timed out after {COMMAND_TIMEOUT_SECONDS}s (repository may be too large — try a smaller `depth` or check the URL)."
    except Exception as e:
        return f"❌ Failed to run git clone: {e}"

    if result.returncode != 0:
        shutil.rmtree(dest_path, ignore_errors=True)
        return f"❌ git clone failed:\n{result.stderr.strip()}"

    _notify("created", destination)
    return f"✅ Cloned {repo_url} → {destination}"


def git_diff(path: str = ".") -> str:
    """
    Shows the current git diff (unstaged + staged changes) for the workspace
    or a specific file/folder within it, using the project's own git history.
    Requires the workspace (or the given path) to be inside a git repository.

    Args:
        path: relative path to limit the diff to (defaults to the whole repo)
    """
    try:
        cwd = _safe_path(".")
    except PermissionError as e:
        return f"❌ {e}"

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "⏱️ git command timed out while checking the repository."
    if check.returncode != 0:
        return "ℹ️ This workspace is not a git repository (git init hasn't been run, or there's no .git folder)."

    _ensure_undo_history_gitignored(cwd)

    cmd = ["git", "diff", "--", path] if path != "." else ["git", "diff"]
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "⏱️ git diff timed out."
    except Exception as e:
        return f"❌ Failed to run git diff: {e}"

    if result.returncode != 0:
        return f"❌ git diff failed:\n{result.stderr.strip()}"
    if not result.stdout.strip():
        return "✅ No changes (working tree matches the last commit)."
    return result.stdout


def lint_check(path: str) -> str:
    """
    Runs a linter/syntax check appropriate for the file's extension and
    returns any errors or warnings found. Useful right after writing or
    editing code to catch mistakes immediately.

    Supported: .py (uses 'python -m py_compile' for a fast syntax check, plus
    'ruff'/'flake8' if installed for style/lint issues), .js/.jsx/.ts/.tsx
    (uses 'node --check' for syntax, plus 'eslint' if available), .json
    (validated by parsing it), .html (basic well-formedness via the
    standard library parser).

    Args:
        path: relative path of the file to check
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"

    ext = p.suffix.lower()
    cwd = config.WORKSPACE_DIR
    issues = []

    try:
        if ext == ".py":
            syntax = subprocess.run(
                ["python3", "-m", "py_compile", str(p)],
                cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
            )
            if syntax.returncode != 0:
                issues.append(f"Syntax error:\n{syntax.stderr.strip()}")
            else:
                for linter in (["ruff", "check", str(p)], ["flake8", str(p)]):
                    if shutil.which(linter[0]):
                        lint = subprocess.run(
                            linter, cwd=str(cwd), capture_output=True, text=True,
                            timeout=COMMAND_TIMEOUT_SECONDS,
                        )
                        output = (lint.stdout + lint.stderr).strip()
                        if output:
                            issues.append(f"{linter[0]} output:\n{output}")
                        break  # only run the first available linter

        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            if shutil.which("node"):
                syntax = subprocess.run(
                    ["node", "--check", str(p)], cwd=str(cwd), capture_output=True, text=True,
                    timeout=COMMAND_TIMEOUT_SECONDS,
                )
                if syntax.returncode != 0:
                    issues.append(f"Syntax error:\n{syntax.stderr.strip()}")
            if shutil.which("eslint"):
                lint = subprocess.run(
                    ["eslint", str(p)], cwd=str(cwd), capture_output=True, text=True,
                    timeout=COMMAND_TIMEOUT_SECONDS,
                )
                output = (lint.stdout + lint.stderr).strip()
                if output:
                    issues.append(f"eslint output:\n{output}")
    except subprocess.TimeoutExpired:
        return f"⏱️ Linter timed out after {COMMAND_TIMEOUT_SECONDS}s while checking {path}."

    if ext == ".json":
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")

    elif ext in (".html", ".htm"):
        from html.parser import HTMLParser

        class _Checker(HTMLParser):
            pass  # a parse error alone (raised below) indicates malformed HTML

        try:
            _Checker().feed(p.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"HTML parsing issue: {e}")

    elif ext not in (".py", ".js", ".jsx", ".ts", ".tsx"):
        return f"ℹ️ No linter configured for '{ext}' files."

    if not issues:
        return f"✅ No issues found in {path}."
    return f"⚠️ Issues in {path}:\n\n" + "\n\n".join(issues)


def file_stats(path: str) -> str:
    """
    Returns quick metadata about a file — size, line count, and last
    modified time — without reading its full content. Useful to check
    before deciding whether/how to read a potentially large file.

    Args:
        path: relative file path
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    if not p.is_file():
        return f"❌ Path is not a file: {path}"

    stat = p.stat()
    size_bytes = stat.st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        line_count = len(text.splitlines())
    except Exception:
        line_count = None

    modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    lines = [
        f"📄 {path}",
        f"Size: {size_str}",
        f"Lines: {line_count if line_count is not None else 'N/A (binary file)'}",
        f"Last modified: {modified}",
    ]
    return "\n".join(lines)


def detect_language(path: str) -> str:
    """
    Guesses the programming language / file type of a file based on its
    extension and, for ambiguous cases, a quick peek at its content (e.g.
    shebang lines). Useful before deciding how to process an unfamiliar
    file.

    Args:
        path: relative file path
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"

    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript (React)",
        ".ts": "TypeScript", ".tsx": "TypeScript (React)", ".java": "Java",
        ".c": "C", ".h": "C header", ".cpp": "C++", ".hpp": "C++ header",
        ".cs": "C#", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
        ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".m": "Objective-C",
        ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
        ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
        ".xml": "XML", ".md": "Markdown", ".sh": "Shell script",
        ".sql": "SQL", ".r": "R", ".lua": "Lua", ".dart": "Dart",
        ".vue": "Vue", ".svelte": "Svelte",
    }
    ext = p.suffix.lower()
    if ext in ext_map:
        return f"{path}: {ext_map[ext]} (by extension {ext})"

    # No/unknown extension — peek at the first line for a shebang
    try:
        first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        if first_line.startswith("#!"):
            return f"{path}: script with shebang '{first_line.strip()}'"
    except Exception:
        pass
    return f"{path}: unknown/unrecognized type (extension: '{ext or 'none'}')"


def count_files(extension: str = None) -> str:
    """
    Counts files in the workspace, optionally filtered by extension, broken
    down by extension. Respects .agentignore plus sensible defaults.

    Args:
        extension: optional extension to filter by (e.g. ".py"). If omitted,
            returns a breakdown of counts for every extension found.
    """
    patterns = _load_ignore_patterns()
    counts: dict = {}
    total = 0
    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        ext = p.suffix.lower() or "(no extension)"
        if extension and ext != extension.lower():
            continue
        counts[ext] = counts.get(ext, 0) + 1
        total += 1

    if extension:
        return f"{counts.get(extension.lower(), 0)} file(s) with extension '{extension}'."

    if not counts:
        return "No files found in the workspace."
    lines = [f"{total} file(s) total:"]
    for ext, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {ext}: {count}")
    return "\n".join(lines)


def replace_in_files(
    old_text: str,
    new_text: str,
    extensions: List[str] = None,
    whole_word: bool = False,
    case_insensitive: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Replaces every occurrence of old_text with new_text across all matching
    files in the workspace (e.g. renaming a variable/function project-wide).
    Each modified file's previous content is snapshotted so this can be
    undone with undo_last_change() (one undo entry per file changed — call
    undo_last_change() repeatedly to revert all of them).

    Args:
        old_text: the text to find and replace
        new_text: the replacement text
        extensions: optional list of extensions to restrict the operation
            (e.g. [".py", ".js"]) — strongly recommended for safety on
            broad replacements
        whole_word: if True, only replaces old_text when it appears as a
            whole word (not part of a longer word) — e.g. with
            whole_word=True, replacing "name" won't touch "namespace" or
            "username". Recommended for renaming variables/identifiers.
        case_insensitive: if True, matches "Name", "NAME", "name", etc. all
            the same way — note new_text is still inserted exactly as given
            (it does not try to preserve the original casing per match).
        dry_run: if True, shows which files/lines WOULD change (with a
            preview) without actually writing anything — use this first on
            any broad or risky replacement to confirm the scope before
            committing to it.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    if whole_word:
        pattern = re.compile(r"\b" + re.escape(old_text) + r"\b", flags)
    else:
        pattern = re.compile(re.escape(old_text), flags)

    patterns = _load_ignore_patterns()
    changed_files = []
    preview_lines = []

    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        if extensions and p.suffix not in extensions:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        matches = list(pattern.finditer(content))
        if not matches:
            continue

        if dry_run:
            lines = content.splitlines()
            line_starts = []
            offset = 0
            for line in lines:
                line_starts.append(offset)
                offset += len(line) + 1
            matched_line_numbers = set()
            for m in matches:
                for i, start in enumerate(line_starts):
                    if start > m.start():
                        break
                    line_no = i
                matched_line_numbers.add(line_no + 1)
            preview_lines.append(f"📄 {rel} ({len(matches)} match(es)):")
            for line_no in sorted(matched_line_numbers)[:10]:
                preview_lines.append(f"  line {line_no}: {lines[line_no - 1].strip()}")
            if len(matched_line_numbers) > 10:
                preview_lines.append(f"  ... and {len(matched_line_numbers) - 10} more line(s)")
            continue

        _notify("editing", str(rel))
        count = len(matches)
        _record_undo("edit", str(rel), content, existed=True)
        new_content = pattern.sub(lambda m: new_text, content)
        p.write_text(new_content, encoding="utf-8")
        _notify("edited", str(rel))
        changed_files.append(f"  {rel} ({count} replacement(s))")

    if dry_run:
        if not preview_lines:
            return "ℹ️ No occurrences of the given text were found in any matching file (dry run — nothing would change)."
        return (
            f"🔍 DRY RUN — no files were modified. This is what would change:\n\n"
            + "\n".join(preview_lines)
            + "\n\nRe-run with dry_run=False to actually apply these changes."
        )

    if not changed_files:
        return "ℹ️ No occurrences of the given text were found in any matching file."
    return f"✅ Replaced text in {len(changed_files)} file(s):\n" + "\n".join(changed_files)


def create_folder(path: str) -> str:
    """
    Creates a new empty folder (and any missing parent folders) inside the
    workspace. Note: git doesn't track empty folders, so this is mainly
    useful for organizing the workspace before adding files to it.

    Args:
        path: relative folder path to create (e.g. "src/components")
    """
    p = _safe_path(path)
    if p.exists():
        if p.is_dir():
            return f"ℹ️ Folder already exists: {path}"
        return f"❌ A file (not a folder) already exists at: {path}"
    p.mkdir(parents=True)
    return f"✅ Folder created: {path}"


def copy_file(source_path: str, destination_path: str) -> str:
    """
    Copies a file to a new location within the workspace (unlike move_file,
    the original stays in place). Missing destination parent directories are
    created automatically.

    Args:
        source_path: the file to copy
        destination_path: where to copy it to
    """
    _notify("creating", destination_path)
    src = _safe_path(source_path)
    dst = _safe_path(destination_path)

    if not src.exists():
        return f"❌ Source file not found: {source_path}"
    if not src.is_file():
        return f"❌ Source is not a file: {source_path}"
    if dst.exists():
        return f"⚠️ Destination already exists: {destination_path}."

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    _record_undo("create", destination_path, None, existed=False)
    _notify("created", destination_path)
    return f"✅ Copied: {source_path} → {destination_path}"


def compare_files(path_a: str, path_b: str) -> str:
    """
    Shows a diff between two existing files already in the workspace (as
    opposed to diff_preview, which compares an existing file against
    proposed-but-not-yet-written content). Useful for comparing two versions
    of a file, e.g. before and after a refactor saved under different names.

    Args:
        path_a: relative path of the first file
        path_b: relative path of the second file
    """
    a = _safe_path(path_a)
    b = _safe_path(path_b)
    if not a.exists():
        return f"❌ File not found: {path_a}"
    if not b.exists():
        return f"❌ File not found: {path_b}"

    content_a = a.read_text(encoding="utf-8", errors="ignore")
    content_b = b.read_text(encoding="utf-8", errors="ignore")
    diff = difflib.unified_diff(
        content_a.splitlines(keepends=True),
        content_b.splitlines(keepends=True),
        fromfile=f"a/{path_a}",
        tofile=f"b/{path_b}",
    )
    result = "".join(diff)
    return result if result else "✅ Files are identical."


def count_todos() -> str:
    """
    Scans the workspace for TODO, FIXME, HACK, XXX, and similar marker
    comments across all files, returning each occurrence with its file and
    line number. Respects .agentignore plus sensible defaults.
    """
    markers = ["TODO", "FIXME", "HACK", "XXX", "BUG"]
    patterns = _load_ignore_patterns()
    matches_by_file: dict = {}

    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in markers):
                matches_by_file.setdefault(str(rel), []).append(f"  line {i}: {line.strip()}")

    if not matches_by_file:
        return "✅ No TODO/FIXME/HACK/XXX/BUG markers found in the workspace."

    total = sum(len(v) for v in matches_by_file.values())
    lines = [f"📝 Found {total} marker(s) across {len(matches_by_file)} file(s):\n"]
    for rel_path, file_matches in matches_by_file.items():
        lines.append(f"📄 {rel_path}")
        lines.extend(file_matches)
        lines.append("")
    return "\n".join(lines).rstrip()


def check_file_syntax_all() -> str:
    """
    Runs lint_check on every supported file (.py, .js/.jsx/.ts/.tsx, .json,
    .html) in the workspace and returns a summary of which files have
    issues. Respects .agentignore plus sensible defaults. Useful for a
    project-wide sanity check, e.g. before committing.
    """
    patterns = _load_ignore_patterns()
    supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".html", ".htm"}
    results = []
    checked = 0

    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        if p.suffix.lower() not in supported_exts:
            continue
        checked += 1
        result = lint_check(str(rel))
        if not result.startswith("✅"):
            results.append(f"📄 {rel}:\n{result}")

    if checked == 0:
        return "ℹ️ No supported files (.py/.js/.ts/.json/.html) found to check."
    if not results:
        return f"✅ Checked {checked} file(s) — no issues found."
    return f"⚠️ Checked {checked} file(s), issues found in {len(results)}:\n\n" + "\n\n".join(results)


def git_status() -> str:
    """
    Shows the current git status of the workspace (modified/added/deleted/
    untracked files), similar to running 'git status --short'. Requires the
    workspace to be a git repository.
    """
    try:
        cwd = _safe_path(".")
    except PermissionError as e:
        return f"❌ {e}"

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
        if check.returncode != 0:
            return "ℹ️ This workspace is not a git repository."

        _ensure_undo_history_gitignored(cwd)
        result = subprocess.run(
            ["git", "status", "--short"], cwd=str(cwd), capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "⏱️ git status timed out."

    if result.returncode != 0:
        return f"❌ git status failed:\n{result.stderr.strip()}"
    if not result.stdout.strip():
        return "✅ Working tree is clean (no changes)."
    return result.stdout


def git_log(count: int = 10) -> str:
    """
    Shows recent git commit history (hash, date, message), similar to
    'git log --oneline'. Requires the workspace to be a git repository.

    Args:
        count: how many recent commits to show (default 10)
    """
    try:
        cwd = _safe_path(".")
    except PermissionError as e:
        return f"❌ {e}"

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
        if check.returncode != 0:
            return "ℹ️ This workspace is not a git repository."

        result = subprocess.run(
            ["git", "log", f"-{count}", "--pretty=format:%h  %ad  %s", "--date=short"],
            cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "⏱️ git log timed out."

    if result.returncode != 0:
        return f"❌ git log failed (the repository may have no commits yet):\n{result.stderr.strip()}"
    if not result.stdout.strip():
        return "ℹ️ No commits yet."
    return result.stdout


def git_commit(message: str, add_all: bool = True) -> str:
    """
    Creates a git commit with the given message. By default stages all
    changes first (git add .) before committing. Requires the workspace to
    be a git repository with a configured user.name/user.email.

    Args:
        message: the commit message
        add_all: whether to run 'git add .' before committing (default True)
    """
    try:
        cwd = _safe_path(".")
    except PermissionError as e:
        return f"❌ {e}"

    # Force git to never wait on interactive input (credential prompts, an
    # editor, or a pager) — belt-and-suspenders alongside the timeouts below,
    # since a prompt waiting on stdin wouldn't be caught by a subprocess
    # timeout on some git builds/hooks.
    git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true", "GIT_PAGER": "cat"}

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
            env=git_env,
        )
        if check.returncode != 0:
            return "ℹ️ This workspace is not a git repository (run 'git init' first)."

        _ensure_undo_history_gitignored(cwd)

        if add_all:
            add_result = subprocess.run(
                ["git", "add", "."], cwd=str(cwd), capture_output=True, text=True,
                timeout=COMMAND_TIMEOUT_SECONDS, env=git_env,
            )
            if add_result.returncode != 0:
                return f"❌ git add failed:\n{add_result.stderr.strip()}"

        result = subprocess.run(
            ["git", "commit", "-m", message], cwd=str(cwd), capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS, env=git_env,
        )
    except subprocess.TimeoutExpired:
        return "⏱️ git commit timed out."

    if result.returncode != 0:
        combined = (result.stdout + result.stderr).strip()
        if "nothing to commit" in combined.lower():
            return "ℹ️ Nothing to commit — working tree is clean."
        return f"❌ git commit failed:\n{combined}"
    return f"✅ Committed: {result.stdout.strip()}"


def create_zip(source_path: str, zip_path: str) -> str:
    """
    Compresses a file or folder within the workspace into a .zip archive.

    Args:
        source_path: relative path of the file or folder to compress
        zip_path: relative path for the resulting .zip file (e.g. "backup.zip")
    """
    src = _safe_path(source_path)
    if not src.exists():
        return f"❌ Source not found: {source_path}"

    zip_target = _safe_path(zip_path)
    if not str(zip_target).endswith(".zip"):
        return "❌ zip_path must end with .zip"

    zip_target.parent.mkdir(parents=True, exist_ok=True)
    base_name = str(zip_target)[:-4]  # shutil wants the path WITHOUT .zip

    _notify("creating", zip_path)
    try:
        if src.is_dir():
            shutil.make_archive(base_name, "zip", root_dir=str(src))
        else:
            import zipfile
            with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(src, arcname=src.name)
    except Exception as e:
        return f"❌ Failed to create zip: {e}"
    _notify("created", zip_path)
    return f"✅ Created archive: {zip_path}"


def extract_zip(zip_path: str, destination_path: str = ".") -> str:
    """
    Extracts a .zip archive within the workspace into a destination folder.

    Args:
        zip_path: relative path of the .zip file to extract
        destination_path: relative folder to extract into (defaults to workspace root)
    """
    import zipfile

    src = _safe_path(zip_path)
    if not src.exists():
        return f"❌ Zip file not found: {zip_path}"

    dst = _safe_path(destination_path)
    dst.mkdir(parents=True, exist_ok=True)

    _notify("creating", destination_path)
    try:
        with zipfile.ZipFile(src, "r") as zf:
            # Basic zip-slip protection: refuse to extract if any entry would
            # escape the destination folder.
            for member in zf.namelist():
                member_path = (dst / member).resolve()
                if dst.resolve() not in member_path.parents and member_path != dst.resolve():
                    return f"🚫 Refusing to extract: archive contains an unsafe path ('{member}')."
            zf.extractall(dst)
    except zipfile.BadZipFile:
        return f"❌ '{zip_path}' is not a valid zip file."
    except Exception as e:
        return f"❌ Failed to extract zip: {e}"
    _notify("created", destination_path)
    return f"✅ Extracted {zip_path} → {destination_path}"


def env_var_check(names: List[str]) -> str:
    """
    Checks whether given environment variable names are set (in the shell
    environment this Agent's process runs in), and separately checks
    whether a .env file exists in the workspace defining them. Useful before
    running a project that expects certain environment variables (API keys,
    database URLs, etc.) to be configured. Does not reveal variable values —
    only whether each is set, to avoid leaking secrets into the
    conversation.

    Args:
        names: list of environment variable names to check, e.g.
            ["DATABASE_URL", "API_KEY"]
    """
    import os as _os

    lines = ["Environment variable check (values are not shown, only presence):"]
    for name in names:
        in_shell = name in _os.environ
        lines.append(f"  {name}: {'✅ set' if in_shell else '❌ not set'} (in current shell environment)")

    env_file = config.WORKSPACE_DIR / ".env"
    if env_file.exists():
        env_content = env_file.read_text(encoding="utf-8", errors="ignore")
        defined_in_dotenv = set()
        for line in env_content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                defined_in_dotenv.add(line.split("=", 1)[0].strip())
        lines.append("\n.env file found. Defined there:")
        for name in names:
            status = "✅ defined" if name in defined_in_dotenv else "❌ not defined"
            lines.append(f"  {name}: {status}")
    else:
        lines.append("\nNo .env file found in the workspace.")

    return "\n".join(lines)


def _puter_image_fallback_hint(action_desc: str) -> str:
    """
    Builds the suffix appended to an Image_Fetch/Image_Create failure
    message when a Puter.js fallback is potentially available, telling the
    MODEL (not the user directly) to ask the user for permission before
    trying it — tools can't interactively prompt mid-call, so the model's
    next reply is what actually asks. Returns "" (no suffix) if the
    fallback isn't eligible at all (feature toggle off, or no Puter token
    configured), so the plain Gemini error message is all that's shown in
    that case rather than dangling a fallback that isn't actually usable.
    """
    if not config.PUTER_IMAGE_TOOLS_ENABLED:
        return ""
    if not config.load_puter_token():
        return ""
    return (
        f"\n\nℹ️ A Puter.js fallback for {action_desc} is available (BETA — "
        f"may not work with every model). Ask the user if they'd like you "
        f"to try {action_desc} via Puter.js instead before calling the "
        f"corresponding _Puter tool."
    )


def Image_Fetch(path: str, question: str = "Describe this image in detail.") -> str:
    """
    Looks at one or more image files already in the workspace and answers a
    question about them (or gives a general description if no question is
    given). Use this whenever you need to actually SEE an image's content —
    e.g. to verify a generated image looks right, to read text/diagrams in a
    screenshot, or to describe a photo the user uploaded into the workspace.

    Supported formats: .png, .jpg, .jpeg, .webp, .gif, .bmp

    Args:
        path: relative path of the image file (for multiple images, pass a
            single string with paths separated by commas, e.g.
            "shot1.png, shot2.png")
        question: what you want to know about the image(s) — defaults to a
            general description if not specified
    """
    paths = [p.strip() for p in path.split(",") if p.strip()]
    if not paths:
        return "❌ No image path provided."

    image_parts = []
    for rel_path in paths:
        try:
            p = _safe_path(rel_path)
        except PermissionError as e:
            return f"❌ {e}"
        if not p.exists():
            return f"❌ Image not found: {rel_path}"
        ext = p.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return f"❌ Unsupported image format '{ext}' for {rel_path}. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        image_parts.append((p, ext))

    try:
        from google.genai import types
        client = _get_image_client()

        parts = [question]
        for p, ext in image_parts:
            data = p.read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type=_IMAGE_MIME_TYPES[ext]))

        _notify("running", f"viewing {', '.join(paths)}")
        response = client.models.generate_content(
            model=IMAGE_UNDERSTANDING_MODEL,
            contents=parts,
        )
        _notify("ran", f"viewed {', '.join(paths)}")
        return response.text or "(The model didn't return a text description.)"
    except RuntimeError as e:
        return f"❌ {e}{_puter_image_fallback_hint('viewing this image')}"
    except Exception as e:
        return f"❌ Failed to analyze image(s): {e}{_puter_image_fallback_hint('viewing this image')}"


def Image_Fetch_Puter(path: str, question: str = "Describe this image in detail.") -> str:
    """
    BETA. Same as Image_Fetch, but via a Puter.js vision-capable model
    instead of Gemini. ONLY call this after the user has explicitly agreed
    to try Puter.js for this — e.g. after Image_Fetch failed and you asked
    them, per the fallback hint in its error message. Do not call this
    proactively or as a first choice; Image_Fetch (Gemini) remains the
    default, well-tested path.

    Uses config.PUTER_VISION_MODEL (default "gpt-4o" — configurable via
    /settings puter vision-model <model>). Requires a Puter token to
    already be configured.

    Args:
        path: relative path of the image file (for multiple images, pass a
            single string with paths separated by commas)
        question: what you want to know about the image(s)
    """
    if not config.PUTER_IMAGE_TOOLS_ENABLED:
        return "❌ Puter.js image tools are turned off. Enable with /settings puter images on (BETA)."
    if not config.load_puter_token():
        return "❌ No Puter.js token configured. Use /puterJS to connect one first."

    paths = [p.strip() for p in path.split(",") if p.strip()]
    if not paths:
        return "❌ No image path provided."
    if len(paths) > 1:
        return "❌ Image_Fetch_Puter supports one image at a time (unlike Image_Fetch). Call it separately per image."

    try:
        p = _safe_path(paths[0])
    except PermissionError as e:
        return f"❌ {e}"
    if not p.exists():
        return f"❌ Image not found: {paths[0]}"
    ext = p.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return f"❌ Unsupported image format '{ext}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"

    try:
        import providers
        _notify("running", f"viewing {paths[0]} via Puter.js")
        result = providers.puter_vision_describe(
            config.PUTER_VISION_MODEL, p.read_bytes(), _IMAGE_MIME_TYPES[ext], question
        )
        _notify("ran", f"viewed {paths[0]} via Puter.js")
        return result or "(The Puter.js model didn't return a text description.)"
    except Exception as e:
        return f"❌ Puter.js vision request failed: {e}"


_LAST_SCREENSHOT_PATH = None  # tracks the previous screenshot for frame-differencing


def view_screen(question: str = "Describe what's currently on screen.",
                 only_if_changed: bool = False, change_threshold: float = 2.0) -> str:
    """
    Takes a SINGLE screenshot of the user's screen right now and asks the
    model to describe or answer a question about it. This is
    EVENT-DRIVEN — it only captures when explicitly called (e.g. because
    the user asked "what's on my screen" or "did the install finish"), not
    on any kind of timer or continuous loop. Do not call this repeatedly in
    a polling loop; call it once per meaningful moment (e.g. right after
    starting an installer, or when the user asks about the screen).

    Requires a graphical display to be active (works on Windows/macOS
    natively; on Linux requires an active X11/Wayland session — headless
    servers/containers without a display will get a clear error, not a
    crash).

    Args:
        question: what to ask about the screen (default: general description)
        only_if_changed: if True, compares this screenshot to the last one
            taken via view_screen and returns early (without spending an API
            call) if the screen looks essentially unchanged — see
            change_threshold. Useful for "let me know when this finishes"
            style follow-ups without wasting requests on an unchanged
            screen. Default False (always analyze).
        change_threshold: how different (0-100, percentage of differing
            pixels) the new screenshot must be from the last one to count as
            "changed" when only_if_changed=True. Default 2.0 (small UI
            changes like a blinking cursor won't trigger it; a new window or
            dialog will).
    """
    global _LAST_SCREENSHOT_PATH
    try:
        from PIL import Image, ImageGrab, ImageChops
    except ImportError:
        return "❌ Screen viewing requires the 'Pillow' package (pip install Pillow)."

    _notify("running", "capturing screen")
    try:
        screenshot = ImageGrab.grab()
    except Exception as e:
        _notify("ran", "capturing screen")
        return (
            f"❌ Could not capture the screen: {e}. This usually means no "
            f"graphical display is available (e.g. running headless/over SSH "
            f"without X11 forwarding)."
        )
    _notify("ran", "capturing screen")

    screenshots_dir = config.WORKSPACE_DIR / ".undo_history" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    new_path = screenshots_dir / f"screen_{int(time.time() * 1000)}.png"
    screenshot.save(new_path, "PNG")

    if only_if_changed and _LAST_SCREENSHOT_PATH and _LAST_SCREENSHOT_PATH.exists():
        try:
            from PIL import ImageStat
            previous = Image.open(_LAST_SCREENSHOT_PATH).convert("RGB")
            current = screenshot.convert("RGB")
            if previous.size == current.size:
                # Percentage of pixels that changed meaningfully between the
                # two screenshots: take the per-pixel difference, threshold
                # it into a binary changed/unchanged mask, then the mean of
                # that binary mask directly gives the changed-pixel
                # percentage (0=none changed, 255=all changed -> /255*100).
                diff_gray = ImageChops.difference(previous, current).convert("L")
                binary_mask = diff_gray.point(lambda p: 255 if p > 10 else 0)
                changed_pct = ImageStat.Stat(binary_mask).mean[0] / 255.0 * 100.0

                if changed_pct < change_threshold:
                    _LAST_SCREENSHOT_PATH = new_path
                    return (
                        f"ℹ️ Screen looks essentially unchanged since the last check "
                        f"(~{changed_pct:.1f}% of pixels differ, threshold {change_threshold}%) "
                        f"— skipped analysis to save a request. Call again with "
                        f"only_if_changed=False to force analysis anyway."
                    )
        except Exception:
            pass  # if diffing fails for any reason, just fall through to a normal analysis

    _LAST_SCREENSHOT_PATH = new_path

    try:
        from google.genai import types
        client = _get_image_client()
        response = client.models.generate_content(
            model=IMAGE_UNDERSTANDING_MODEL,
            contents=[question, types.Part.from_bytes(data=new_path.read_bytes(), mime_type="image/png")],
        )
        return response.text or "(The model didn't return a description.)"
    except RuntimeError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Failed to analyze the screenshot: {e}"


def view_screen_puter(question: str = "Describe what's currently on screen.") -> str:
    """
    BETA. Same as view_screen, but analyzes the screenshot via a Puter.js
    vision-capable model (config.PUTER_VISION_MODEL) instead of Gemini. Use
    this as a fallback if view_screen fails (e.g. Gemini quota exhausted),
    or if the user explicitly asked to use Puter.js for screen viewing.
    Requires a Puter token to already be configured and
    config.PUTER_IMAGE_TOOLS_ENABLED to be on.

    Args:
        question: what to ask about the screen (default: general description)
    """
    global _LAST_SCREENSHOT_PATH
    if not config.PUTER_IMAGE_TOOLS_ENABLED:
        return "❌ Puter.js image tools are turned off. Enable with /settings puter images on (BETA)."
    if not config.load_puter_token():
        return "❌ No Puter.js token configured. Use /puterJS to connect one first."

    try:
        from PIL import ImageGrab
    except ImportError:
        return "❌ Screen viewing requires the 'Pillow' package (pip install Pillow)."

    _notify("running", "capturing screen")
    try:
        screenshot = ImageGrab.grab()
    except Exception as e:
        _notify("ran", "capturing screen")
        return f"❌ Could not capture the screen: {e}. No graphical display available?"
    _notify("ran", "capturing screen")

    screenshots_dir = config.WORKSPACE_DIR / ".undo_history" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    new_path = screenshots_dir / f"screen_{int(time.time() * 1000)}.png"
    screenshot.save(new_path, "PNG")
    _LAST_SCREENSHOT_PATH = new_path

    try:
        import providers
        _notify("running", "viewing screen via Puter.js")
        result = providers.puter_vision_describe(
            config.PUTER_VISION_MODEL, new_path.read_bytes(), "image/png", question
        )
        _notify("ran", "viewed screen via Puter.js")
        return result or "(The Puter.js model didn't return a description.)"
    except Exception as e:
        return f"❌ Puter.js vision request failed: {e}"


def watch_screen(question: str = "Describe what's currently on screen, and note anything new or changed.",
                  duration_seconds: int = 30, interval_seconds: int = 5,
                  change_threshold: float = 2.0, use_puter: bool = False) -> str:
    """
    Watches the user's screen like a live feed for a period of time: takes
    repeated screenshots every `interval_seconds`, skips any that look
    essentially unchanged from the previous frame (saving API calls), and
    analyzes the ones that DID change. Returns a single timestamped log of
    everything observed during the watch window — this is the closest thing
    to "streaming" the screen to the model, since a real continuous video
    stream isn't possible through the text/image API used here.

    Use this when the user wants ongoing monitoring — e.g. "tell me when the
    build finishes", "watch what I'm doing and guide me", "let me know if an
    error dialog pops up" — instead of calling view_screen over and over
    yourself. Runs synchronously for the requested duration, so keep
    duration_seconds reasonable (this ties up the current turn); for
    longer/background watching, prefer running it multiple times with
    shorter windows, or wrap it via start_background_process if truly
    unattended monitoring is needed.

    Args:
        question: what to look for / describe on each changed frame
        duration_seconds: total time to watch, in seconds (default 30, keep
            this modest — e.g. 15-120 — since it blocks the conversation)
        interval_seconds: seconds between each screenshot check (default 5;
            lower = more responsive but more API calls/cost)
        change_threshold: percentage of changed pixels (0-100) required to
            count a frame as "changed" and worth analyzing (default 2.0)
        use_puter: if True, analyze changed frames via Puter.js
            (view_screen_puter's model) instead of Gemini — requires a
            configured Puter token and config.PUTER_IMAGE_TOOLS_ENABLED
    """
    try:
        from PIL import Image, ImageGrab, ImageChops, ImageStat
    except ImportError:
        return "❌ Screen watching requires the 'Pillow' package (pip install Pillow)."

    if use_puter:
        if not config.PUTER_IMAGE_TOOLS_ENABLED:
            return "❌ Puter.js image tools are turned off. Enable with /settings puter images on (BETA)."
        if not config.load_puter_token():
            return "❌ No Puter.js token configured. Use /puterJS to connect one first."

    duration_seconds = max(5, min(duration_seconds, 600))  # hard safety cap: 10 minutes max
    interval_seconds = max(1, interval_seconds)

    screenshots_dir = config.WORKSPACE_DIR / ".undo_history" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    previous_img = None
    start = time.time()
    frame_count = 0
    analyzed_count = 0

    _notify("running", f"watching screen for {duration_seconds}s")
    try:
        while time.time() - start < duration_seconds:
            try:
                shot = ImageGrab.grab()
            except Exception as e:
                log_lines.append(f"[error] could not capture screen: {e}")
                break
            frame_count += 1
            elapsed = round(time.time() - start, 1)
            current_img = shot.convert("RGB")

            changed = True
            changed_pct = 100.0
            if previous_img is not None and previous_img.size == current_img.size:
                diff_gray = ImageChops.difference(previous_img, current_img).convert("L")
                binary_mask = diff_gray.point(lambda p: 255 if p > 10 else 0)
                changed_pct = ImageStat.Stat(binary_mask).mean[0] / 255.0 * 100.0
                changed = changed_pct >= change_threshold

            if changed:
                frame_path = screenshots_dir / f"watch_{int(time.time() * 1000)}.png"
                shot.save(frame_path, "PNG")
                try:
                    if use_puter:
                        import providers
                        desc = providers.puter_vision_describe(
                            config.PUTER_VISION_MODEL, frame_path.read_bytes(), "image/png", question
                        )
                    else:
                        from google.genai import types
                        client = _get_image_client()
                        response = client.models.generate_content(
                            model=IMAGE_UNDERSTANDING_MODEL,
                            contents=[question, types.Part.from_bytes(
                                data=frame_path.read_bytes(), mime_type="image/png")],
                        )
                        desc = response.text
                    analyzed_count += 1
                    log_lines.append(f"[t={elapsed}s, {changed_pct:.1f}% changed] {desc or '(no description)'}")
                except Exception as e:
                    log_lines.append(f"[t={elapsed}s] ❌ analysis failed: {e}")
            previous_img = current_img

            time.sleep(max(0, interval_seconds - (time.time() - start - elapsed)))
    finally:
        _notify("ran", f"watched screen for {round(time.time() - start, 1)}s")

    if not log_lines:
        return (
            f"ℹ️ Watched the screen for {round(time.time() - start, 1)}s "
            f"({frame_count} frame(s) captured) — no meaningful changes detected "
            f"(threshold {change_threshold}%)."
        )

    header = (
        f"👁️ Screen watch log — {round(time.time() - start, 1)}s, "
        f"{frame_count} frame(s) captured, {analyzed_count} analyzed "
        f"(via {'Puter.js' if use_puter else 'Gemini'}):\n"
    )
    return header + "\n".join(log_lines)


def _require_system_access() -> Optional[str]:
    """
    Returns None if system access (windows/processes) is permitted, or an
    error string to return from the calling tool if not. This gate is
    OFF by default and can only be turned on by the USER (via
    /settings system access on) — never by the model itself. If it's off,
    the model must ask the user for permission IN ENGLISH before the user
    can enable it, e.g.: "Do you allow the AI to access your open windows
    and system processes? If so, please run: /settings system access on".
    """
    if not config.SYSTEM_ACCESS_ENABLED:
        return (
            "❌ System access (windows/processes) is not permitted yet. "
            "Ask the user, in English, something like: \"Do you allow the AI "
            "to access your open windows and running system processes?\" "
            "If they agree, tell them to run /settings system access on — "
            "the model cannot enable this itself."
        )
    return None


def Available_Active_Windows(include_screenshots: bool = True, max_windows: int = 10) -> str:
    """
    Lists the user's currently open/active windows (title + owning
    application), and — if include_screenshots is True — captures a
    screenshot of each visible window and describes it (including any
    visible text) using the vision model. This gives the AI a picture of
    everything the user has open right now, not just the single
    foreground window that view_screen/watch_screen capture.

    REQUIRES EXPLICIT USER PERMISSION first — see _require_system_access.
    If permission hasn't been granted, this returns an error telling you
    to ask the user in English before it can be used.

    Platform support: Windows and macOS via the 'pygetwindow' package
    (pip install pygetwindow); Linux via the 'wmctrl' command-line tool
    (install with e.g. apt install wmctrl). Falls back to a clear error if
    neither is available.

    Args:
        include_screenshots: if True (default), also screenshots and
            visually describes each window (slower, costs one vision
            request per window). If False, only lists titles/apps —
            fast, no vision calls.
        max_windows: safety cap on how many windows to screenshot/describe
            (default 10) — a list of titles is always returned in full
            regardless of this cap.
    """
    perm_error = _require_system_access()
    if perm_error:
        return perm_error

    windows = []  # list of dicts: title, app, bbox (left, top, right, bottom) or None

    try:
        import pygetwindow as gw
        for w in gw.getAllWindows():
            title = (w.title or "").strip()
            if not title:
                continue
            bbox = None
            try:
                if w.visible and w.width > 0 and w.height > 0:
                    bbox = (w.left, w.top, w.left + w.width, w.top + w.height)
            except Exception:
                bbox = None
            windows.append({"title": title, "app": "", "bbox": bbox})
    except ImportError:
        # Linux fallback: wmctrl -l gives "<id> <desktop> <host> <title>"
        import subprocess
        try:
            out = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or "wmctrl failed")
            for line in out.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) == 4:
                    windows.append({"title": parts[3].strip(), "app": "", "bbox": None})
        except FileNotFoundError:
            return (
                "❌ Could not list windows: install 'pygetwindow' (pip install "
                "pygetwindow) on Windows/macOS, or 'wmctrl' (apt install wmctrl) "
                "on Linux."
            )
        except Exception as e:
            return f"❌ Could not list windows via wmctrl: {e}"
    except Exception as e:
        return f"❌ Could not list windows: {e}"

    if not windows:
        return "ℹ️ No open windows detected (or the window manager didn't report any titles)."

    lines = [f"🪟 {len(windows)} open window(s):"]
    for i, w in enumerate(windows, 1):
        lines.append(f"{i}. {w['title']}")

    if not include_screenshots:
        return "\n".join(lines)

    try:
        from PIL import ImageGrab
        from google.genai import types
    except ImportError:
        lines.append("\n⚠️ Pillow not installed — skipping per-window screenshots.")
        return "\n".join(lines)

    screenshots_dir = config.WORKSPACE_DIR / ".undo_history" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    described = 0
    lines.append("")
    for w in windows:
        if described >= max_windows:
            lines.append(f"... ({len(windows) - described} more window(s) not screenshotted — max_windows={max_windows} reached)")
            break
        if not w["bbox"]:
            continue
        try:
            _notify("running", f"capturing window: {w['title']}")
            shot = ImageGrab.grab(bbox=w["bbox"])
            frame_path = screenshots_dir / f"win_{int(time.time() * 1000)}.png"
            shot.save(frame_path, "PNG")
            client = _get_image_client()
            response = client.models.generate_content(
                model=IMAGE_UNDERSTANDING_MODEL,
                contents=["Briefly describe what's shown in this window, including any visible text.",
                          types.Part.from_bytes(data=frame_path.read_bytes(), mime_type="image/png")],
            )
            lines.append(f"🔎 [{w['title']}]: {response.text or '(no description)'}")
            described += 1
            _notify("ran", f"captured window: {w['title']}")
        except Exception as e:
            lines.append(f"⚠️ [{w['title']}]: could not capture/describe ({e})")

    return "\n".join(lines)


def List_System_Processes(sort_by: str = "cpu", limit: int = 20) -> str:
    """
    Task-Manager-style listing of ALL processes currently running on the
    system (not just ones started by this agent — see
    list_background_processes for that narrower, always-available tool).
    Shows PID, process name, CPU%, memory%, and status for each.

    REQUIRES EXPLICIT USER PERMISSION first — see _require_system_access.
    If permission hasn't been granted, this returns an error telling you
    to ask the user in English before it can be used.

    Requires the 'psutil' package (pip install psutil).

    Args:
        sort_by: "cpu" (default) or "memory" — which usage column to sort
            the results by, highest first
        limit: max number of processes to show (default 20)
    """
    perm_error = _require_system_access()
    if perm_error:
        return perm_error

    try:
        import psutil
    except ImportError:
        return "❌ System process listing requires the 'psutil' package (pip install psutil)."

    sort_by = sort_by.strip().lower()
    if sort_by not in ("cpu", "memory"):
        return "❌ sort_by must be 'cpu' or 'memory'."

    procs = []
    # First pass primes cpu_percent's internal sampling; a short interval
    # gives a real (non-zero) reading instead of always 0.0 on first call.
    for p in psutil.process_iter(["pid", "name", "status", "memory_percent"]):
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(0.15)
    for p in psutil.process_iter(["pid", "name", "status", "memory_percent"]):
        try:
            info = p.info
            cpu = p.cpu_percent(None)
            procs.append({
                "pid": info["pid"],
                "name": info["name"] or "?",
                "cpu": cpu,
                "mem": info.get("memory_percent") or 0.0,
                "status": info.get("status") or "?",
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    key = "cpu" if sort_by == "cpu" else "mem"
    procs.sort(key=lambda x: x[key], reverse=True)
    procs = procs[:max(1, limit)]

    lines = [f"🖥️ Top {len(procs)} processes by {sort_by.upper()} ({len(list(psutil.process_iter()))} total running):",
             f"{'PID':>7}  {'CPU%':>6}  {'MEM%':>6}  STATUS      NAME"]
    for p in procs:
        lines.append(f"{p['pid']:>7}  {p['cpu']:>6.1f}  {p['mem']:>6.1f}  {p['status']:<10}  {p['name']}")
    return "\n".join(lines)


def Image_Create(prompt: str, output_path: str, aspect_ratio: str = "1:1") -> str:
    """
    Generates an image from a text description using Gemini's built-in
    image generation ("Nano Banana") and saves it to the workspace. Free
    tier: up to 500 requests/day.

    Args:
        prompt: a detailed description of the image to generate
        output_path: relative path to save the generated image to (e.g.
            "assets/logo.png") — should end in .png
        aspect_ratio: one of "1:1", "16:9", "9:16", "4:3", "3:2" (default "1:1")
    """
    valid_ratios = {"1:1", "16:9", "9:16", "4:3", "3:2"}
    if aspect_ratio not in valid_ratios:
        return f"❌ Invalid aspect_ratio '{aspect_ratio}'. Must be one of: {sorted(valid_ratios)}"

    try:
        out_path = _safe_path(output_path)
    except PermissionError as e:
        return f"❌ {e}"
    if out_path.suffix.lower() != ".png":
        return "❌ output_path must end in .png"

    _notify("creating", output_path)
    from google.genai import types
    try:
        client = _get_image_client()
    except RuntimeError as e:
        return f"❌ {e}"

    response = None
    last_error = None
    for model_name in config.IMAGE_MODEL_CHAIN:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
            break  # success — stop trying further models
        except Exception as e:
            last_error = e
            continue  # try the next image model in the chain

    if response is None:
        return f"❌ Failed to generate image (all image models failed): {last_error}{_puter_image_fallback_hint('generating this image')}"

    image_bytes = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                image_bytes = inline.data
                break
        if image_bytes:
            break

    if not image_bytes:
        return (
            "❌ The model didn't return image data. It may have refused the "
            f"prompt — check response text/safety filters.{_puter_image_fallback_hint('generating this image')}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)
    _record_undo("create", output_path, None, existed=False)
    _notify("created", output_path)
    return f"✅ Image generated and saved: {output_path} ({len(image_bytes)} bytes)"


def Image_Create_Puter(prompt: str, output_path: str) -> str:
    """
    BETA — genuinely experimental (more so than the other Puter tools in
    this file). Attempts image generation via Puter.js instead of Gemini.
    ONLY call this after the user has explicitly agreed to try Puter.js for
    this — e.g. after Image_Create failed and you asked them, per the
    fallback hint in its error message. Do not call this proactively.

    Important honesty note for whoever is reading this tool's result: this
    specific capability (image generation through Puter's REST/OpenAI-
    compatible endpoint, as opposed to Puter's browser-only txt2img JS
    function) has NOT been confirmed to exist by Puter's own documentation
    — it may fail outright. If it fails, say so plainly rather than
    implying Gemini and Puter are equally reliable fallbacks for image
    generation specifically (they are NOT — only Image_Fetch_Puter, the
    vision/understanding direction, rests on solidly documented ground).

    Uses config.PUTER_IMAGE_GEN_MODEL (default "gpt-image-1" — configurable
    via /settings puter image-model <model>). No aspect_ratio parameter
    (unlike Image_Create) since Puter's supported options here are
    unverified.

    Args:
        prompt: a detailed description of the image to generate
        output_path: relative path to save the generated image to (e.g.
            "assets/logo.png") — should end in .png
    """
    if not config.PUTER_IMAGE_TOOLS_ENABLED:
        return "❌ Puter.js image tools are turned off. Enable with /settings puter images on (BETA)."
    if not config.load_puter_token():
        return "❌ No Puter.js token configured. Use /puterJS to connect one first."

    try:
        out_path = _safe_path(output_path)
    except PermissionError as e:
        return f"❌ {e}"
    if out_path.suffix.lower() != ".png":
        return "❌ output_path must end in .png"

    try:
        import providers
        _notify("creating", f"{output_path} via Puter.js")
        image_bytes = providers.puter_image_generate(config.PUTER_IMAGE_GEN_MODEL, prompt)
    except Exception as e:
        return (
            f"❌ Puter.js image generation failed: {e}\n\n"
            f"ℹ️ This was expected to be possible but is unverified — Puter's own docs "
            f"only demonstrate image generation via their browser JavaScript SDK, not "
            f"this REST path. Gemini (Image_Create) remains the reliable option."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)
    _record_undo("create", output_path, None, existed=False)
    _notify("created", output_path)
    return f"✅ Image generated via Puter.js and saved: {output_path} ({len(image_bytes)} bytes)"


def list_dependencies() -> str:
    """
    Reads the project's dependency manifest(s) (package.json, requirements.txt,
    Pipfile, pyproject.toml, Cargo.toml, go.mod — whichever are present) and
    returns a summary of declared dependencies. Useful before adding a new
    dependency (to check it isn't already there) or to understand what a
    project needs to run.
    """
    found_any = False
    sections = []

    pkg_json = config.WORKSPACE_DIR / "package.json"
    if pkg_json.exists():
        found_any = True
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            lines = ["📦 package.json (Node.js):"]
            if deps:
                lines.append("  dependencies:")
                lines.extend(f"    {name}: {ver}" for name, ver in deps.items())
            if dev_deps:
                lines.append("  devDependencies:")
                lines.extend(f"    {name}: {ver}" for name, ver in dev_deps.items())
            if not deps and not dev_deps:
                lines.append("  (no dependencies declared)")
            sections.append("\n".join(lines))
        except json.JSONDecodeError:
            sections.append("📦 package.json exists but is not valid JSON.")

    req_txt = config.WORKSPACE_DIR / "requirements.txt"
    if req_txt.exists():
        found_any = True
        lines = [l.strip() for l in req_txt.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        section = ["🐍 requirements.txt (Python):"]
        section.extend(f"  {l}" for l in lines) if lines else section.append("  (empty)")
        sections.append("\n".join(section))

    pyproject = config.WORKSPACE_DIR / "pyproject.toml"
    if pyproject.exists():
        found_any = True
        sections.append("🐍 pyproject.toml exists (Python — use read_file to inspect [project.dependencies] or [tool.poetry.dependencies]).")

    cargo_toml = config.WORKSPACE_DIR / "Cargo.toml"
    if cargo_toml.exists():
        found_any = True
        sections.append("🦀 Cargo.toml exists (Rust — use read_file to inspect [dependencies]).")

    go_mod = config.WORKSPACE_DIR / "go.mod"
    if go_mod.exists():
        found_any = True
        sections.append("🐹 go.mod exists (Go — use read_file to inspect require blocks).")

    if not found_any:
        return "ℹ️ No recognized dependency manifest found (package.json, requirements.txt, pyproject.toml, Cargo.toml, go.mod)."
    return "\n\n".join(sections)


def add_dependency(package: str, dev: bool = False, version: str = None) -> str:
    """
    Adds a dependency to the project's manifest file, auto-detecting the
    project type from what's present in the workspace (package.json ->
    npm/Node, requirements.txt -> Python). For Node projects this actually
    runs the package manager (npm install) so package.json AND
    package-lock.json/node_modules stay consistent; for Python it appends a
    line to requirements.txt (does not install it — follow up with
    run_command("pip install -r requirements.txt") if you want it installed
    immediately too).

    Args:
        package: package name (e.g. "express", "requests")
        dev: whether this is a dev-only dependency (default False)
        version: optional version/version-specifier to pin (e.g. "^4.18.0"
            for npm, "==2.31.0" for pip). If omitted, the latest version is used.
    """
    pkg_json = config.WORKSPACE_DIR / "package.json"
    req_txt = config.WORKSPACE_DIR / "requirements.txt"

    if pkg_json.exists():
        pkg_spec = f"{package}@{version}" if version else package
        cmd = f"npm install {'--save-dev' if dev else '--save'} {pkg_spec}"
        return run_command(cmd)

    if req_txt.exists() or not pkg_json.exists():
        # Default to Python/requirements.txt if nothing else is detected —
        # the most common case for a fresh/empty workspace.
        line = f"{package}{version}" if version else package
        existing = req_txt.read_text(encoding="utf-8") if req_txt.exists() else ""
        if any(line.split("==")[0].split(">=")[0].strip() == package for line in existing.splitlines()):
            return f"ℹ️ '{package}' already appears in requirements.txt."
        _notify("editing", "requirements.txt")
        new_content = existing.rstrip("\n") + ("\n" if existing.strip() else "") + line + "\n"
        _record_undo("edit" if req_txt.exists() else "create", "requirements.txt", existing if req_txt.exists() else None, existed=req_txt.exists())
        req_txt.write_text(new_content.lstrip("\n"), encoding="utf-8")
        _notify("edited", "requirements.txt")
        return f"✅ Added '{line}' to requirements.txt (run 'pip install -r requirements.txt' to actually install it)."

    return "❌ Could not determine project type (no package.json or requirements.txt found)."


def run_tests(test_command: str = None) -> str:
    """
    Runs the project's test suite, auto-detecting the right command if not
    given explicitly: package.json's "test" script for Node projects, or
    pytest/unittest for Python projects. Returns the test output including
    pass/fail summary.

    Args:
        test_command: optional explicit command to run instead of
            auto-detection (e.g. "pytest tests/ -v")
    """
    if test_command:
        return run_command(test_command)

    pkg_json = config.WORKSPACE_DIR / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            if "test" in data.get("scripts", {}):
                return run_command("npm test")
        except json.JSONDecodeError:
            pass

    has_test_files = bool(list(config.WORKSPACE_DIR.rglob("test_*.py")) + list(config.WORKSPACE_DIR.rglob("*_test.py")))
    if has_test_files:
        # Prefer pytest (handles both plain test_*() functions and
        # unittest.TestCase classes) — check via `python3 -m pytest
        # --version` rather than shutil.which("pytest"), since pytest is
        # often only importable as a module (`python3 -m pytest`) without a
        # standalone `pytest` binary on PATH.
        pytest_check = subprocess.run(
            ["python3", "-m", "pytest", "--version"],
            cwd=str(config.WORKSPACE_DIR), capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        if pytest_check.returncode == 0:
            return run_command("python3 -m pytest -v")

        # No pytest available — unittest's discovery only finds
        # unittest.TestCase-based tests, not plain pytest-style test_*()
        # functions, so warn about that limitation rather than silently
        # reporting "0 tests ran" with no explanation.
        result = run_command("python3 -m unittest discover -v")
        return (
            result + "\n\nℹ️ Note: pytest isn't installed, so only "
            "unittest.TestCase-style tests are discovered — plain test_*() "
            "functions (pytest-style) won't be picked up by unittest. "
            "Install pytest (add_dependency('pytest')) for full compatibility."
        )

    return (
        "ℹ️ Could not auto-detect a test command (no package.json test script, "
        "no test_*.py/*_test.py files found, or pytest not installed). "
        "Pass test_command explicitly if you know how tests should be run."
    )


def create_test_file(source_path: str, test_path: str = None) -> str:
    """
    Creates a starter test file for a given source file, with a basic
    template appropriate to the language (pytest-style for Python,
    Jest-style for JS/TS). Does not write actual test logic — just
    scaffolding (imports, a describe/test-function skeleton) for you to
    fill in with real assertions.

    Args:
        source_path: relative path of the source file to create tests for
            (e.g. "src/utils.py")
        test_path: optional relative path for the test file — if omitted,
            follows convention (test_utils.py next to utils.py for Python,
            utils.test.js next to utils.js for JS/TS)
    """
    src = _safe_path(source_path)
    if not src.exists():
        return f"❌ Source file not found: {source_path}"

    ext = src.suffix.lower()
    stem = src.stem
    rel_parent = Path(source_path).parent  # relative parent, e.g. "src" — NOT src.parent (which is absolute)

    if ext == ".py":
        default_test_path = str(rel_parent / f"test_{stem}.py") if str(rel_parent) != "." else f"test_{stem}.py"
        content = (
            f'"""Tests for {source_path}."""\n'
            f"import pytest\n\n\n"
            f"def test_{stem}_placeholder():\n"
            f"    # TODO: import from '{stem}' and write real assertions\n"
            f"    assert True\n"
        )
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        default_test_path = str(rel_parent / f"{stem}.test{ext}") if str(rel_parent) != "." else f"{stem}.test{ext}"
        content = (
            f"// Tests for {source_path}\n"
            f"describe('{stem}', () => {{\n"
            f"  test('placeholder', () => {{\n"
            f"    // TODO: import from './{stem}' and write real assertions\n"
            f"    expect(true).toBe(true);\n"
            f"  }});\n"
            f"}});\n"
        )
    else:
        return f"❌ Don't know how to scaffold tests for '{ext}' files (supported: .py, .js, .jsx, .ts, .tsx)."

    final_path = test_path or default_test_path
    return create_file(final_path, content)


def generate_readme(project_name: str = None) -> str:
    """
    Generates a basic README.md for the project by inspecting the workspace
    (detected languages, dependency manifests, existing folder structure)
    and writing a starter README with sections for description, setup, and
    usage. Meant as a scaffold to edit further, not a final polished doc.

    Args:
        project_name: name to use as the README title — if omitted, uses
            the workspace folder's name
        """
    name = project_name or config.WORKSPACE_DIR.name or "Project"

    # Detect what kind of project this looks like
    has_package_json = (config.WORKSPACE_DIR / "package.json").exists()
    has_requirements = (config.WORKSPACE_DIR / "requirements.txt").exists()
    has_pyproject = (config.WORKSPACE_DIR / "pyproject.toml").exists()

    setup_lines = []
    if has_package_json:
        setup_lines.append("```bash\nnpm install\n```")
    if has_requirements:
        setup_lines.append("```bash\npip install -r requirements.txt\n```")
    if has_pyproject:
        setup_lines.append("```bash\npip install .\n```")
    if not setup_lines:
        setup_lines.append("_(Add setup instructions here.)_")

    run_lines = []
    if has_package_json:
        try:
            data = json.loads((config.WORKSPACE_DIR / "package.json").read_text(encoding="utf-8"))
            if "start" in data.get("scripts", {}):
                run_lines.append("```bash\nnpm start\n```")
            elif "dev" in data.get("scripts", {}):
                run_lines.append("```bash\nnpm run dev\n```")
        except json.JSONDecodeError:
            pass
    if not run_lines:
        run_lines.append("_(Add run instructions here.)_")

    content = f"""# {name}

## Description

_(Add a short description of what this project does.)_

## Setup

{chr(10).join(setup_lines)}

## Usage

{chr(10).join(run_lines)}

## Project Structure

{tools_generate_readme_tree()}
"""
    return create_file("README.md", content)


def tools_generate_readme_tree() -> str:
    """Internal helper: a short top-level directory listing formatted for
    embedding into generate_readme's output (not registered as a tool
    itself)."""
    patterns = _load_ignore_patterns()
    lines = []
    try:
        entries = sorted(config.WORKSPACE_DIR.iterdir())
    except FileNotFoundError:
        return "_(empty)_"
    for e in entries:
        rel = e.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        icon = "📁" if e.is_dir() else "📄"
        lines.append(f"- {icon} `{rel.name}`")
    return "\n".join(lines) if lines else "_(empty)_"


def extract_docstrings(path: str) -> str:
    """
    Extracts all docstrings (module/class/function-level) from a Python
    file using the ast module, without needing to read/understand the full
    source. Fast way to get an overview of what a file's functions/classes
    do without dumping the entire implementation.

    Args:
        path: relative path of the Python file
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    if p.suffix.lower() != ".py":
        return f"❌ extract_docstrings only supports Python (.py) files, got: {p.suffix}"

    import ast
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return f"❌ Could not parse {path} (syntax error): {e}"

    results = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        results.append(f"Module docstring:\n  {module_doc}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if doc:
                results.append(f"{kind} {node.name}() (line {node.lineno}):\n  {doc}")
            else:
                results.append(f"{kind} {node.name}() (line {node.lineno}): (no docstring)")

    if not results:
        return f"ℹ️ No functions, classes, or module docstring found in {path}."
    return "\n\n".join(results)


def save_checkpoint(name: str, description: str = "") -> str:
    """
    Saves a full snapshot of the current workspace state under a named
    checkpoint, so you can return to this exact point later with
    load_checkpoint even after many further changes — stronger than
    undo_last_change, which only reverts one step at a time. Useful before
    a risky refactor or big experimental change.

    Args:
        name: a short identifier for this checkpoint (e.g. "before-refactor")
        description: optional longer description of what state this captures
    """
    checkpoints_dir = config.WORKSPACE_DIR / ".undo_history" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    checkpoint_zip = checkpoints_dir / f"{safe_name}.zip"

    if checkpoint_zip.exists():
        return f"⚠️ A checkpoint named '{name}' already exists. Choose a different name or load/delete the existing one first."

    import zipfile
    patterns = _load_ignore_patterns()
    with zipfile.ZipFile(checkpoint_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in config.WORKSPACE_DIR.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(config.WORKSPACE_DIR)
            if _is_ignored(rel, patterns) or ".undo_history" in rel.parts:
                continue
            zf.write(p, arcname=str(rel))

    meta = {
        "name": name,
        "description": description,
        "ts": time.time(),
    }
    (checkpoints_dir / f"{safe_name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"✅ Checkpoint '{name}' saved."


def load_checkpoint(name: str) -> str:
    """
    Restores the workspace to a previously saved checkpoint (see
    save_checkpoint), overwriting current files with the checkpoint's
    snapshot. Files created after the checkpoint that aren't part of it are
    left alone (this restores/overwrites, it doesn't delete extras) — use
    list_checkpoints first if unsure which checkpoints exist.

    Args:
        name: the checkpoint's name to restore
    """
    checkpoints_dir = config.WORKSPACE_DIR / ".undo_history" / "checkpoints"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    checkpoint_zip = checkpoints_dir / f"{safe_name}.zip"

    if not checkpoint_zip.exists():
        return f"❌ No checkpoint named '{name}' found. Use list_checkpoints to see available ones."

    import zipfile
    _notify("editing", f"workspace (restoring checkpoint '{name}')")
    with zipfile.ZipFile(checkpoint_zip, "r") as zf:
        for member in zf.namelist():
            member_path = (config.WORKSPACE_DIR / member).resolve()
            if config.WORKSPACE_DIR.resolve() not in member_path.parents and member_path != config.WORKSPACE_DIR.resolve():
                return f"🚫 Refusing to restore: checkpoint contains an unsafe path ('{member}')."
        zf.extractall(config.WORKSPACE_DIR)
    _notify("edited", f"workspace (restored checkpoint '{name}')")
    return f"✅ Workspace restored to checkpoint '{name}'."


def list_checkpoints() -> str:
    """Lists all saved checkpoints with their descriptions and timestamps."""
    checkpoints_dir = config.WORKSPACE_DIR / ".undo_history" / "checkpoints"
    if not checkpoints_dir.exists():
        return "ℹ️ No checkpoints saved yet."

    lines = []
    for meta_file in sorted(checkpoints_dir.glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("ts", 0)))
        desc = f" — {meta['description']}" if meta.get("description") else ""
        lines.append(f"📌 {meta['name']} ({when}){desc}")

    if not lines:
        return "ℹ️ No checkpoints saved yet."
    return "\n".join(lines)


def check_port_in_use(port: int) -> str:
    """
    Checks whether a network port is already in use on the local machine —
    useful before starting a dev server, to catch "port already in use"
    errors before they happen and suggest an alternative port.

    Args:
        port: the port number to check (e.g. 3000, 8080)
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", port))
        in_use = result == 0

    if in_use:
        return f"🔴 Port {port} is already in use. Try a different port or stop whatever is using it first (see list_background_processes if it's one of yours)."
    return f"🟢 Port {port} is free."


def count_lines_of_code() -> str:
    """
    Counts lines of code in the workspace, broken down by language/extension,
    distinguishing blank lines and comment-only lines from actual code lines
    where practical. Respects .agentignore plus sensible defaults. Gives a
    quick sense of project size beyond just a file count.
    """
    patterns = _load_ignore_patterns()
    # Extensions we know a single-line-comment prefix for, so we can roughly
    # separate comments from code. Anything else is just counted as
    # "lines" without the code/comment/blank breakdown.
    comment_prefixes = {
        ".py": "#", ".sh": "#", ".rb": "#", ".yaml": "#", ".yml": "#",
        ".js": "//", ".jsx": "//", ".ts": "//", ".tsx": "//", ".java": "//",
        ".c": "//", ".cpp": "//", ".h": "//", ".hpp": "//", ".cs": "//",
        ".go": "//", ".rs": "//", ".swift": "//", ".kt": "//",
    }

    stats_by_ext: dict = {}
    for p in config.WORKSPACE_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(config.WORKSPACE_DIR)
        if _is_ignored(rel, patterns):
            continue
        ext = p.suffix.lower() or "(no extension)"
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        entry = stats_by_ext.setdefault(ext, {"files": 0, "total": 0, "blank": 0, "comment": 0, "code": 0})
        entry["files"] += 1
        prefix = comment_prefixes.get(ext)
        for line in lines:
            stripped = line.strip()
            entry["total"] += 1
            if not stripped:
                entry["blank"] += 1
            elif prefix and stripped.startswith(prefix):
                entry["comment"] += 1
            else:
                entry["code"] += 1

    if not stats_by_ext:
        return "No files found in the workspace."

    lines_out = ["📊 Lines of code by extension:\n"]
    total_all = {"files": 0, "total": 0, "blank": 0, "comment": 0, "code": 0}
    for ext, s in sorted(stats_by_ext.items(), key=lambda x: -x[1]["total"]):
        for k in total_all:
            total_all[k] += s[k]
        lines_out.append(
            f"  {ext}: {s['files']} file(s), {s['total']} lines "
            f"({s['code']} code, {s['comment']} comment, {s['blank']} blank)"
        )
    lines_out.append(
        f"\nTotal: {total_all['files']} file(s), {total_all['total']} lines "
        f"({total_all['code']} code, {total_all['comment']} comment, {total_all['blank']} blank)"
    )
    return "\n".join(lines_out)


def http_request(url: str, method: str = "GET", headers: dict = None, body: str = None, timeout: int = 15) -> str:
    """
    Sends an HTTP request (GET/POST/PUT/DELETE/etc.) and returns the status
    code, response headers, and body. Useful for testing a local API you're
    building (e.g. http://localhost:3000/api/users) or checking a request
    against any reachable URL. Response body is truncated if very large.

    Args:
        url: the URL to request (e.g. "http://localhost:8000/api/health")
        method: HTTP method (default "GET")
        headers: optional dict of request headers (e.g. {"Content-Type": "application/json"})
        body: optional request body as a string (e.g. a JSON payload already serialized)
        timeout: request timeout in seconds (default 15)
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method=method.upper(), headers=headers or {})
    if body is not None:
        req.data = body.encode("utf-8")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp_headers = dict(resp.getheaders())
            resp_body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = e.code
        resp_headers = dict(e.headers or {})
        resp_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
    except urllib.error.URLError as e:
        return f"❌ Request failed: {e.reason}"
    except Exception as e:
        return f"❌ Request failed: {e}"

    if len(resp_body) > 3000:
        resp_body = resp_body[:3000] + f"\n... (truncated, {len(resp_body)} total chars)"

    header_lines = "\n".join(f"  {k}: {v}" for k, v in resp_headers.items())
    return (
        f"$ {method.upper()} {url}\n"
        f"Status: {status}\n"
        f"Headers:\n{header_lines}\n\n"
        f"Body:\n{resp_body}"
    )


def find_unused_imports(path: str) -> str:
    """
    Scans a Python file for imports that appear to be unused (imported but
    never referenced elsewhere in the file). Uses simple name-usage
    counting via the ast module — not a full type-aware analysis, so it can
    have false positives/negatives for dynamic usage (e.g. `__all__`,
    string-based references, re-exports), but catches the common case
    reliably.

    Args:
        path: relative path of the Python file to check
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    if p.suffix.lower() != ".py":
        return f"❌ find_unused_imports only supports Python (.py) files, got: {p.suffix}"

    import ast
    try:
        source = p.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Could not parse {path} (syntax error): {e}"

    imported_names = {}  # name -> line number
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imported_names[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue  # can't reliably track star-imports
                name = alias.asname or alias.name
                imported_names[name] = node.lineno

    if not imported_names:
        return f"ℹ️ No imports found in {path}."

    # Count all Name/Attribute usages in the file, excluding the import
    # statements themselves, to see which imported names are referenced.
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used_names.add(node.value.id)

    # Also check for __all__ exports and plain string mentions (best-effort
    # coverage for re-export patterns without full false-negative safety).
    # Also give __all__ exports a pass (a name listed there is being
    # "used" as a public re-export even if never referenced elsewhere).
    all_export_names = set()
    if "__all__" in source:
        try:
            all_export_names = set(ast.literal_eval(source.split("__all__", 1)[1].split("=", 1)[1].split("\n")[0].strip().rstrip(",")))
        except Exception:
            pass  # if __all__ isn't a simple literal list, just skip this extra check

    unused = []
    for name, lineno in sorted(imported_names.items(), key=lambda x: x[1]):
        if name not in used_names and name not in all_export_names:
            unused.append(f"  line {lineno}: '{name}' appears unused")

    if not unused:
        return f"✅ No unused imports detected in {path}."
    return f"⚠️ Possibly unused imports in {path} (double-check before removing — dynamic usage can cause false positives):\n" + "\n".join(unused)


def convert_file_format(source_path: str, destination_path: str) -> str:
    """
    Converts a data file between common formats based on the source and
    destination extensions. Supported conversions: .json <-> .yaml/.yml,
    .json <-> .csv (for a flat list of flat dicts), .csv <-> .json.

    Args:
        source_path: relative path of the file to convert
        destination_path: relative path for the converted output (its
            extension determines the target format)
    """
    src = _safe_path(source_path)
    if not src.exists():
        return f"❌ Source file not found: {source_path}"

    src_ext = src.suffix.lower()
    dst_ext = Path(destination_path).suffix.lower()

    try:
        if src_ext == ".json" and dst_ext in (".yaml", ".yml"):
            import yaml
            data = json.loads(src.read_text(encoding="utf-8"))
            output = yaml.dump(data, allow_unicode=True, sort_keys=False)
        elif src_ext in (".yaml", ".yml") and dst_ext == ".json":
            import yaml
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
            output = json.dumps(data, ensure_ascii=False, indent=2)
        elif src_ext == ".json" and dst_ext == ".csv":
            import csv
            import io
            data = json.loads(src.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                return "❌ .json -> .csv conversion requires a JSON array of flat objects (e.g. [{\"a\": 1, \"b\": 2}, ...])."
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
            output = buf.getvalue()
        elif src_ext == ".csv" and dst_ext == ".json":
            import csv
            import io
            reader = csv.DictReader(io.StringIO(src.read_text(encoding="utf-8")))
            output = json.dumps(list(reader), ensure_ascii=False, indent=2)
        else:
            return f"❌ Unsupported conversion: {src_ext} -> {dst_ext}. Supported: json<->yaml, json<->csv."
    except ImportError:
        return "❌ The 'pyyaml' package is required for YAML conversion — install it with add_dependency('pyyaml')."
    except Exception as e:
        return f"❌ Conversion failed: {e}"

    return create_file(destination_path, output)


def minify_file(path: str, output_path: str = None) -> str:
    """
    Minifies a JSON, CSS, or JS file (removes unnecessary whitespace/
    comments) to reduce size. For JS, this is a simple whitespace/comment
    stripper, not a full minifier — good enough for basic size reduction,
    not for advanced optimizations (variable renaming, dead code elimination).

    Args:
        path: relative path of the file to minify
        output_path: optional path to save the minified version to — if
            omitted, overwrites the original file (previous content is
            snapshotted first, so this can be undone)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"❌ File not found: {path}"

    ext = p.suffix.lower()
    content = p.read_text(encoding="utf-8")

    if ext == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON, cannot minify: {e}"
        minified = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    elif ext == ".css":
        # Strip /* comments */, collapse whitespace, remove space around
        # punctuation — a simple but effective CSS minifier.
        minified = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        minified = re.sub(r"\s+", " ", minified)
        minified = re.sub(r"\s*([{}:;,])\s*", r"\1", minified)
        minified = minified.strip()

    elif ext in (".js", ".jsx"):
        # Best-effort: strip // and /* */ comments and collapse blank lines.
        # Does NOT touch strings/regex containing "//" perfectly — good
        # enough for simple scripts, not safe for complex minification.
        minified = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        lines = [l for l in minified.splitlines() if l.strip() and not l.strip().startswith("//")]
        minified = "\n".join(lines)

    else:
        return f"❌ minify_file only supports .json, .css, .js/.jsx files, got: {ext}"

    target = output_path or path
    return create_file(target, minified)


def git_fetcher(repo: str, include_tree: bool = True) -> str:
    """
    Fetches metadata about a GitHub repository WITHOUT cloning it — repo
    description, star/fork/watcher counts, license, primary language,
    open issues count, default branch, last push date, and (optionally) the
    top-level file tree. Much faster than git_clone when you just need to
    understand what a repo is/contains before deciding whether to clone it.

    Works via GitHub's public REST API, no authentication required for
    public repositories (subject to GitHub's unauthenticated rate limit —
    60 requests/hour per IP). Only public repos are accessible without a
    connected GitHub account.

    Args:
        repo: repository in "owner/name" form (e.g. "octocat/Hello-World"),
            or a full GitHub URL (e.g. "https://github.com/octocat/Hello-World")
        include_tree: whether to also fetch and show the top-level file/
            folder listing (default True) — set False for a faster,
            metadata-only lookup
    """
    # Accept either "owner/repo" or a full github.com URL.
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo)
    if match:
        owner, name = match.group(1), match.group(2)
    elif "/" in repo and not repo.startswith("http"):
        owner, name = repo.split("/", 1)
    else:
        return f"❌ Could not parse '{repo}' as a GitHub repo — use 'owner/name' or a full github.com URL."

    api_base = f"https://api.github.com/repos/{owner}/{name}"
    meta_result = http_request(api_base)
    if meta_result.startswith("❌"):
        return meta_result

    # http_request returns a formatted "$ GET ...\nStatus: N\nHeaders:...\n\nBody:\n{...}"
    # string (built for human/model reading) — extract the JSON body back out.
    body_marker = "\n\nBody:\n"
    if body_marker not in meta_result:
        return f"❌ Unexpected response format fetching {owner}/{name}."
    raw_body = meta_result.split(body_marker, 1)[1]

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError:
        return f"❌ Could not parse GitHub's response for {owner}/{name} (it may not exist, or you hit GitHub's rate limit)."

    if data.get("message") == "Not Found":
        return f"❌ Repository '{owner}/{name}' not found (private, or doesn't exist)."
    if "API rate limit exceeded" in str(data.get("message", "")):
        return f"❌ GitHub API rate limit exceeded for this IP. Try again later, or use git_clone directly instead."

    license_name = (data.get("license") or {}).get("name", "None")
    lines = [
        f"📦 {data.get('full_name', f'{owner}/{name}')}",
        f"Description: {data.get('description') or '(none)'}",
        f"⭐ Stars: {data.get('stargazers_count', 0)}  "
        f"🍴 Forks: {data.get('forks_count', 0)}  "
        f"👁 Watchers: {data.get('watchers_count', 0)}  "
        f"🐛 Open issues: {data.get('open_issues_count', 0)}",
        f"License: {license_name}",
        f"Primary language: {data.get('language') or '(unknown)'}",
        f"Default branch: {data.get('default_branch', 'main')}",
        f"Last push: {data.get('pushed_at', 'unknown')}",
        f"Archived: {data.get('archived', False)}",
        f"URL: {data.get('html_url', f'https://github.com/{owner}/{name}')}",
    ]

    if include_tree:
        branch = data.get("default_branch", "main")
        tree_result = http_request(f"{api_base}/contents/")
        if not tree_result.startswith("❌") and body_marker in tree_result:
            try:
                tree_body = tree_result.split(body_marker, 1)[1]
                entries = json.loads(tree_body)
                if isinstance(entries, list):
                    lines.append("\nTop-level contents:")
                    for entry in sorted(entries, key=lambda e: (e.get("type") != "dir", e.get("name", ""))):
                        icon = "📁" if entry.get("type") == "dir" else "📄"
                        lines.append(f"  {icon} {entry.get('name')}")
            except (json.JSONDecodeError, KeyError):
                lines.append("\n(Could not fetch file tree)")

    return "\n".join(lines)


# List of all available tools (passed to the model via function calling)
ALL_TOOLS = [
    create_file,
    read_file,
    edit_file,
    delete_file,
    move_file,
    rename_file,
    copy_file,
    create_folder,
    list_files,
    find_file,
    find_folder,
    search_in_files,
    replace_in_files,
    compare_files,
    diff_preview,
    file_stats,
    detect_language,
    count_files,
    count_todos,
    run_command,
    start_background_process,
    list_background_processes,
    read_background_log,
    stop_background_process,
    wait_process,
    git_clone,
    git_fetcher,
    git_diff,
    git_status,
    git_log,
    git_commit,
    lint_check,
    check_file_syntax_all,
    create_zip,
    extract_zip,
    env_var_check,
    Image_Fetch,
    Image_Fetch_Puter,
    Image_Create,
    Image_Create_Puter,
    view_screen,
    view_screen_puter,
    watch_screen,
    Available_Active_Windows,
    List_System_Processes,
    list_dependencies,
    add_dependency,
    run_tests,
    create_test_file,
    generate_readme,
    extract_docstrings,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    check_port_in_use,
    count_lines_of_code,
    http_request,
    find_unused_imports,
    convert_file_format,
    minify_file,
    undo_last_change,
]
