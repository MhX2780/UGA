r"""
Windows Taskbar Progress Bar integration.

Uses the Windows 10/11 virtual terminal sequence `ESC]9;4;<state>[;value]ESC\`
to control the progress indicator on the application's taskbar icon.

States:
  0 = No progress / clear
  1 = Normal (percentage, 0-100)
  2 = Error (red)
  3 = Indeterminate (pulsing)
  4 = Paused (yellow)

This is a no-op on non-Windows platforms or when the terminal doesn't
support the sequence — the escape codes are simply not printed, causing
no visible effect.

Usage:
    import taskbar_progress as tb

    tb.set_indeterminate()   # pulsing loading
    tb.set_progress(50)       # 50%
    tb.clear()                # hide
"""
import os
import sys


# Detect Windows
_IS_WINDOWS = sys.platform == "win32"


def _write_sequence(seq: str):
    """Writes an escape sequence to stdout if on Windows.
    Silently ignored on other platforms."""
    if _IS_WINDOWS:
        try:
            sys.stdout.write(seq)
            sys.stdout.flush()
        except Exception:
            pass


def _get_esc() -> str:
    """Returns the ESC character (ASCII 27 / \x1b)."""
    return "\x1b"


def set_indeterminate():
    """Show pulsing/indeterminate progress on the taskbar icon."""
    esc = _get_esc()
    _write_sequence(f"{esc}]9;4;3{esc}\\")


def set_progress(percent: int):
    """
    Set the taskbar progress to a specific percentage (0-100).

    Args:
        percent: Integer from 0 to 100.
    """
    percent = max(0, min(100, int(percent)))
    esc = _get_esc()
    _write_sequence(f"{esc}]9;4;1;{percent}{esc}\\")


def set_error():
    """Show error state (red) on the taskbar icon."""
    esc = _get_esc()
    _write_sequence(f"{esc}]9;4;2{esc}\\")


def set_paused():
    """Show paused state (yellow) on the taskbar icon."""
    esc = _get_esc()
    _write_sequence(f"{esc}]9;4;4{esc}\\")


def clear():
    """Hide/clear the taskbar progress bar."""
    esc = _get_esc()
    _write_sequence(f"{esc}]9;4;0{esc}\\")


def set_step_progress(current_step: int, total_steps: int):
    """
    Update taskbar progress based on the current plan step.

    Logic:
      - Step 1: Indeterminate (we don't know how long it takes)
      - Step 2+: Show percentage of completed steps before this one
        percentage = ((current_step - 1) / total_steps) * 100

    Args:
        current_step: The step number currently running (1-based).
        total_steps: Total number of steps.
    """
    if current_step <= 1:
        set_indeterminate()
    else:
        percent = int(((current_step - 1) / total_steps) * 100)
        set_progress(percent)
