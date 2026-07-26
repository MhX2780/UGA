"""
A small, dependency-free Markdown-to-ANSI renderer for terminal output.

The Gemini API always replies in Markdown (bold, code blocks, headers, lists,
etc.) since that's how the model is trained to format text. Printing that raw
in a terminal shows literal '**', '```', '#' characters instead of actual
formatting. This module converts the common Markdown constructs into ANSI
escape codes so the CLI shows properly styled text instead.

This is intentionally simple (regex-based) rather than a full CommonMark
parser — it covers what LLM replies typically use: headers, bold, italic,
inline code, fenced code blocks, bullet/numbered lists, and blockquotes.
"""
import re

from colors import C


def render_markdown(text: str) -> str:
    lines = text.split("\n")
    out_lines = []
    in_code_block = False
    code_lang = ""

    for line in lines:
        # ---- fenced code blocks ```lang ... ``` ----
        fence_match = re.match(r"^\s*```(\w*)\s*$", line)
        if fence_match:
            if not in_code_block:
                in_code_block = True
                code_lang = fence_match.group(1)
                label = f" {code_lang}" if code_lang else ""
                out_lines.append(f"{C.DIM}┌─{label}{C.RESET}")
            else:
                in_code_block = False
                out_lines.append(f"{C.DIM}└─{C.RESET}")
            continue

        if in_code_block:
            out_lines.append(f"{C.DIM}│{C.RESET} {C.BRIGHT_CYAN}{line}{C.RESET}")
            continue

        out_lines.append(_render_inline(line))

    return "\n".join(out_lines)


def _render_inline(line: str) -> str:
    # ---- headers: #, ##, ### ----
    header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
    if header_match:
        content = header_match.group(2)
        return f"{C.BOLD}{C.BRIGHT_BLUE}{content}{C.RESET}"

    # ---- blockquote: > text ----
    if line.strip().startswith(">"):
        content = line.strip().lstrip(">").strip()
        return f"{C.DIM}▏ {content}{C.RESET}"

    # ---- bullet lists: -, *, + at line start ----
    bullet_match = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
    if bullet_match:
        indent, content = bullet_match.groups()
        content = _render_spans(content)
        return f"{indent}{C.CYAN}•{C.RESET} {content}"

    # ---- numbered lists: 1. text ----
    numbered_match = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
    if numbered_match:
        indent, num, content = numbered_match.groups()
        content = _render_spans(content)
        return f"{indent}{C.CYAN}{num}.{C.RESET} {content}"

    return _render_spans(line)


def _render_spans(text: str) -> str:
    # ---- inline code: `code` ----
    text = re.sub(
        r"`([^`]+)`",
        lambda m: f"{C.BRIGHT_CYAN}{m.group(1)}{C.RESET}",
        text,
    )
    # ---- bold: **text** or __text__ ----
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: f"{C.BOLD}{m.group(1)}{C.RESET}",
        text,
    )
    text = re.sub(
        r"__(.+?)__",
        lambda m: f"{C.BOLD}{m.group(1)}{C.RESET}",
        text,
    )
    # ---- italic: *text* or _text_ (after bold is handled, so ** is already consumed) ----
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        lambda m: f"{C.DIM}{m.group(1)}{C.RESET}",
        text,
    )
    return text
