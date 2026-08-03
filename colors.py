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
from typing import Optional


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


def _strip_ansi(s: str) -> str:
    """Removes ANSI escape codes only (keeps the actual text/emoji intact),
    used when we need to slice a string by visible characters without
    corrupting color codes."""
    return re.sub(r"\033\[[0-9;?]*[a-zA-Z]", "", s)


def _wrap_line(line: str, inner_w: int) -> list:
    """
    Splits a single body line into one or more chunks that each fit within
    inner_w visible columns, breaking on whitespace where possible so words
    aren't cut mid-way. This is what keeps the box's right-hand border
    aligned even when a line (e.g. a long model name or error message) is
    wider than the box — instead of overflowing past the border, it wraps
    onto additional lines inside the same box.

    Note: this strips ANSI color codes from the line before measuring/
    wrapping (color codes have zero visible width but their presence between
    characters makes plain slicing unsafe). Colored `lines` passed to
    draw_box will lose their color on wrap; callers that need color AND long
    text should pre-wrap and pass already-short lines instead.
    """
    if inner_w < 1:
        inner_w = 1
    plain = _strip_ansi(line)
    if _visible_len(plain) <= inner_w:
        return [line]

    words = plain.split(" ")
    chunks = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if _visible_len(candidate) <= inner_w:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # A single word longer than inner_w must be hard-split by
            # character, since there's no whitespace to break on.
            while _visible_len(word) > inner_w:
                # slice conservatively by plain character count; visible
                # width only exceeds character count for wide/emoji chars,
                # so this stays within bounds even if not perfectly tight
                cut = word[:inner_w]
                chunks.append(cut)
                word = word[inner_w:]
            current = word
    if current:
        chunks.append(current)
    return chunks or [""]


def draw_box(title: str, lines, color: str = None, width: int = None) -> str:
    """
    Renders a box-drawn panel with a title and body lines, e.g.:

        ╭─ Title ───────────────╮
        │ line one              │
        │ line two              │
        ╰────────────────────────╯

    `lines` can contain ANSI codes; visible-length is computed by stripping
    escape sequences so borders still line up. Any line (or the title)
    wider than the box wraps onto additional lines instead of overflowing
    past the right border, so the box stays a consistent width front to
    back regardless of content length.
    """
    color = color or C.CYAN
    w = width or term_width()
    inner_w = w - 4  # account for "│ " + " │"

    # If the title itself is too long for the box, widen the box to fit it
    # (up to the terminal width) rather than letting the top border overflow.
    title_visible = _visible_len(title) if title else 0
    min_w_for_title = title_visible + 6  # "╭─ " + " ─╮" + a little breathing room
    if title and min_w_for_title > w:
        w = min(min_w_for_title, term_width())
        inner_w = w - 4

    top_title = f" {title} " if title else ""
    top_fill = "─" * max(0, w - 2 - _visible_len(top_title))
    top = f"{color}╭{top_title}{top_fill}╮{C.RESET}"

    body = []
    for line in lines:
        for wrapped in _wrap_line(line, inner_w):
            pad = max(0, inner_w - _visible_len(wrapped))
            body.append(f"{color}│{C.RESET} {wrapped}{' ' * pad} {color}│{C.RESET}")

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


def _read_single_keypress() -> Optional[str]:
    """
    Reads one raw keypress from the terminal without waiting for Enter, and
    returns a normalized name: "up", "down", "enter", "q" (quit/cancel), or
    None if the key isn't one we care about (caller should just read again).
    Returns None immediately (no blocking) if raw key reading isn't possible
    in this environment (e.g. Windows without msvcrt for some reason, or
    input isn't a real interactive terminal) — callers must have a fallback
    for that case; see select_menu()'s numbered-input path.
    """
    if os.name == "nt":
        try:
            import msvcrt
        except ImportError:
            return "__unsupported__"
        ch = msvcrt.getch()
        if ch in (b"\xe0", b"\x00"):  # arrow key prefix on Windows
            ch2 = msvcrt.getch()
            if ch2 == b"H":
                return "up"
            if ch2 == b"P":
                return "down"
            return None
        if ch in (b"\r", b"\n"):
            return "enter"
        if ch in (b"q", b"Q", b"\x03"):  # 'q' or Ctrl+C
            return "q"
        return None
    else:
        try:
            import termios
            import tty
        except ImportError:
            return "__unsupported__"
        if not sys.stdin.isatty():
            return "__unsupported__"
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # ESC — start of an arrow-key escape sequence
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == "[" and ch3 == "A":
                    return "up"
                if ch2 == "[" and ch3 == "B":
                    return "down"
                return None
            if ch in ("\r", "\n"):
                return "enter"
            if ch in ("q", "Q", "\x03"):
                return "q"
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def select_menu(title: str, options: list, descriptions: list = None) -> Optional[int]:
    """
    Shows an arrow-key navigable menu (↑/↓ to move, Enter to select, q/Ctrl+C
    to cancel) and returns the selected option's index, or None if cancelled.

    Falls back automatically to plain numbered input ("1", "2", ... then
    Enter) if raw keypress reading isn't available in the current
    environment (e.g. piped/non-interactive stdin, or an unsupported
    platform) — the menu always works, just without live arrow-key
    highlighting in that fallback case.

    Args:
        title: heading shown above the options
        options: list of option label strings
        descriptions: optional list of one-line descriptions shown dimmed
            under each option (same length as options, or None)
    """
    if not options:
        return None

    selected = 0

    def render():
        lines = [f"{C.BOLD}{title}{C.RESET}", ""]
        for i, opt in enumerate(options):
            marker = f"{C.GREEN}❯{C.RESET}" if i == selected else " "
            label = f"{C.BOLD}{opt}{C.RESET}" if i == selected else opt
            lines.append(f"{marker} {label}")
            if descriptions and i < len(descriptions) and descriptions[i]:
                lines.append(f"   {C.DIM}{descriptions[i]}{C.RESET}")
        lines.append("")
        lines.append(f"{C.DIM}↑/↓ to move, Enter to select, q to cancel{C.RESET}")
        return "\n".join(lines)

    if not _ENABLED or not sys.stdin.isatty():
        # No interactive terminal (or colors disabled, usually correlated
        # with non-interactive input too) — plain numbered fallback.
        print(f"{title}\n")
        for i, opt in enumerate(options):
            print(f"  {i + 1}. {opt}")
            if descriptions and i < len(descriptions) and descriptions[i]:
                print(f"     {descriptions[i]}")
        try:
            raw = input(f"\nChoose 1-{len(options)} (or Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        try:
            choice = int(raw) - 1
            return choice if 0 <= choice < len(options) else None
        except ValueError:
            return None

    sys.stdout.write(C.HIDE_CURSOR)
    rendered = render()
    print(rendered)
    line_count = rendered.count("\n") + 1

    try:
        while True:
            key = _read_single_keypress()
            if key == "__unsupported__":
                # Raw key reading isn't available after all (detected only
                # once we actually tried) — erase what we drew and fall
                # back to numbered input instead of hanging.
                sys.stdout.write(f"\033[{line_count}A\033[J")
                sys.stdout.write(C.SHOW_CURSOR)
                print(f"{title}\n")
                for i, opt in enumerate(options):
                    print(f"  {i + 1}. {opt}")
                try:
                    raw = input(f"\nChoose 1-{len(options)} (or Enter to cancel): ").strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                if not raw:
                    return None
                try:
                    choice = int(raw) - 1
                    return choice if 0 <= choice < len(options) else None
                except ValueError:
                    return None
            elif key == "up":
                selected = (selected - 1) % len(options)
            elif key == "down":
                selected = (selected + 1) % len(options)
            elif key == "enter":
                return selected
            elif key == "q":
                return None
            else:
                continue  # unrecognized key — just wait for the next one

            # Redraw in place: move cursor up to the start of our block and
            # clear to the end of the screen before printing the new state.
            sys.stdout.write(f"\033[{line_count}A\033[J")
            rendered = render()
            print(rendered)
            line_count = rendered.count("\n") + 1
    finally:
        sys.stdout.write(C.SHOW_CURSOR)
        sys.stdout.flush()
