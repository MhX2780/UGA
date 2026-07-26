"""
Small ANSI color helper used across the CLI and router for nicer terminal output.
Falls back gracefully (no crash) on terminals that don't render ANSI codes —
they'll just show the raw escape sequences, which is harmless.
"""
import itertools
import os
import re
import shutil
import sys
import threading
import time


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


_ENABLED = _supports_color()


class C:
    RESET = "\033[0m" if _ENABLED else ""
    BOLD = "\033[1m" if _ENABLED else ""
    DIM = "\033[2m" if _ENABLED else ""
    ITALIC = "\033[3m" if _ENABLED else ""

    RED = "\033[31m" if _ENABLED else ""
    GREEN = "\033[32m" if _ENABLED else ""
    YELLOW = "\033[33m" if _ENABLED else ""
    BLUE = "\033[34m" if _ENABLED else ""
    MAGENTA = "\033[35m" if _ENABLED else ""
    CYAN = "\033[36m" if _ENABLED else ""
    WHITE = "\033[37m" if _ENABLED else ""

    BRIGHT_RED = "\033[91m" if _ENABLED else ""
    BRIGHT_GREEN = "\033[92m" if _ENABLED else ""
    BRIGHT_YELLOW = "\033[93m" if _ENABLED else ""
    BRIGHT_BLUE = "\033[94m" if _ENABLED else ""
    BRIGHT_MAGENTA = "\033[95m" if _ENABLED else ""
    BRIGHT_CYAN = "\033[96m" if _ENABLED else ""

    # A couple of 256-color accents for a more distinctive "modern" palette
    VIOLET = "\033[38;5;141m" if _ENABLED else ""
    TEAL = "\033[38;5;80m" if _ENABLED else ""
    ORANGE = "\033[38;5;215m" if _ENABLED else ""
    PINK = "\033[38;5;213m" if _ENABLED else ""

    # Cursor / line control
    CLEAR_LINE = "\033[2K\r" if _ENABLED else ""
    HIDE_CURSOR = "\033[?25l" if _ENABLED else ""
    SHOW_CURSOR = "\033[?25h" if _ENABLED else ""


def term_width(default: int = 78) -> int:
    """Returns the current terminal width, capped to a sensible range. Capped
    lower than a typical desktop terminal since this app is also used on
    narrow mobile terminals (e.g. Termux on Android), where a too-wide box
    wraps mid-line and looks broken."""
    try:
        w = shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        w = default
    return max(30, min(w, 80))


def _visible_len(s: str) -> int:
    """
    Approximates how many terminal columns a string occupies: strips ANSI
    escape codes, then counts emoji/wide East-Asian characters as 2 columns
    each (most terminals render them double-width) instead of 1. Without
    this, box borders drift out of alignment on any line containing an
    emoji (📂, 🧠, ✅, etc.) since Python's len() counts them as a single
    character.
    """
    import unicodedata
    stripped = re.sub(r"\033\[[0-9;?]*[a-zA-Z]", "", s)
    width = 0
    for ch in stripped:
        code = ord(ch)
        # Common emoji ranges + box-drawing/symbol ranges that render wide
        # on most terminal emulators, including Termux.
        if (
            0x1F300 <= code <= 0x1FAFF  # misc symbols, emoji, supplemental
            or 0x2600 <= code <= 0x27BF  # misc symbols & dingbats (✅, ⚠, etc.)
            or 0x2190 <= code <= 0x21FF  # arrows (↩ etc.)
            or unicodedata.east_asian_width(ch) in ("W", "F")
        ):
            width += 2
        else:
            width += 1
    return width


def draw_box(title: str, lines, color: str = None, width: int = None) -> str:
    """
    Renders a box-drawn panel with a title and body lines, e.g.:

        ╭─ Title ───────────────╮
        │ line one              │
        │ line two              │
        ╰────────────────────────╯

    `lines` can contain ANSI codes; visible-length is computed by stripping
    escape sequences so borders still line up.
    """
    color = color or C.CYAN
    w = width or term_width()
    inner_w = w - 4  # account for "│ " + " │"

    top_title = f" {title} " if title else ""
    top_fill = "─" * max(0, w - 2 - _visible_len(top_title))
    top = f"{color}╭{top_title}{top_fill}╮{C.RESET}"

    body = []
    for line in lines:
        pad = max(0, inner_w - _visible_len(line))
        body.append(f"{color}│{C.RESET} {line}{' ' * pad} {color}│{C.RESET}")

    bottom = f"{color}╰{'─' * (w - 2)}╯{C.RESET}"
    return "\n".join([top] + body + [bottom])


class Spinner:
    """
    A simple animated terminal spinner shown while waiting for the first
    token of a response. Runs on a background thread so it doesn't block;
    call .stop() once output starts arriving.

    Usage:
        spinner = Spinner("Thinking")
        spinner.start()
        ...
        spinner.stop()
    """
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking", color: str = None):
        self.message = message
        self.color = color or C.VIOLET
        self._stop_event = threading.Event()
        self._thread = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            if _ENABLED:
                sys.stdout.write(
                    f"{C.CLEAR_LINE}{self.color}{frame} {self.message}...{C.RESET}"
                )
                sys.stdout.flush()
            time.sleep(0.08)

    def start(self):
        if not _ENABLED:
            print(f"{self.message}...")
            return
        sys.stdout.write(C.HIDE_CURSOR)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if _ENABLED:
            sys.stdout.write(f"{C.CLEAR_LINE}{C.SHOW_CURSOR}")
            sys.stdout.flush()
