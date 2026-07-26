"""
Memory system.
- session_log.jsonl      : every message from every session (raw log, for tracking/review)
- long_term_memory.jsonl : persistent facts/preferences about the user (loaded as
                           context in every new session)

We use JSONL (one JSON record per line) instead of one big JSON file because:
  1) It allows fast appends without rewriting the whole file.
  2) It's resistant to partial corruption (if the program is interrupted mid-write,
     the remaining lines are still intact).
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import config


class MemoryStore:
    def __init__(self):
        self.long_term_file: Path = config.LONG_TERM_MEMORY_FILE
        self.session_file: Path = config.SESSION_LOG_FILE

    # ---------------- raw session log ----------------
    def log_message(self, role: str, content: str, meta: Optional[Dict] = None):
        """Logs every message (from the user or the model) to the raw log."""
        record = {
            "ts": time.time(),
            "role": role,           # "user" | "model" | "system" | "tool"
            "content": content,
            "meta": meta or {},
        }
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_recent_session(self, limit: int = config.MAX_HISTORY_MESSAGES) -> List[Dict]:
        """Returns the last N messages across all sessions (short/mid-term memory)."""
        if not self.session_file.exists():
            return []
        lines = self.session_file.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        return records[-limit:]

    # ---------------- persistent (long-term) memory ----------------
    def remember(self, key: str, value: str, category: str = "general"):
        """
        Saves a persistent fact/preference. Example:
            remember("user_name", "Ahmed", category="profile")
            remember("preferred_language", "Python", category="preference")
        """
        record = {
            "ts": time.time(),
            "key": key,
            "value": value,
            "category": category,
        }
        with open(self.long_term_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recall_all(self) -> Dict[str, Dict]:
        """
        Returns the latest value for each key (most recent update wins).
        Result: {key: {"value":..., "category":..., "ts":...}}
        """
        if not self.long_term_file.exists():
            return {}
        result: Dict[str, Dict] = {}
        for line in self.long_term_file.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            result[rec["key"]] = {
                "value": rec["value"],
                "category": rec["category"],
                "ts": rec["ts"],
            }
        return result

    def forget(self, key: str):
        """
        Deletes a key from long-term memory by rewriting the file without it.
        (JSONL is append-only, so actual deletion requires a rewrite — this is
        rarely called.)
        """
        if not self.long_term_file.exists():
            return
        lines = self.long_term_file.read_text(encoding="utf-8").strip().splitlines()
        kept = []
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["key"] != key:
                kept.append(line)
        self.long_term_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def memory_as_context_string(self) -> str:
        """Turns long-term memory into text ready to be added to the model's
        context (system prompt)."""
        mem = self.recall_all()
        if not mem:
            return ""
        lines = ["Saved information about the user from previous conversations:"]
        for key, data in mem.items():
            lines.append(f"- {key}: {data['value']} (category: {data['category']})")
        return "\n".join(lines)


class ExecutionLog:
    """
    A persistent, condensed log of "what tools were run and what happened",
    separate from the raw conversation history. Purpose: when the automatic
    model router switches models mid-task (e.g. due to a rate limit), the
    new model has no memory of what the previous model just did unless it's
    somewhere in the prompt context. Re-sending the full tool-call history
    (which can include large file contents, long command output, etc.) is
    wasteful and can blow past context limits. Instead, this keeps a short,
    one-line-per-action summary — "ran X, worked/failed" — that gets
    injected into every system prompt so any model (new or continuing) can
    see recent progress at a glance without redoing completed steps or
    re-reading files it already read moments ago.

    Stored as JSONL (see module docstring for why), capped to the most
    recent entries when building context, but the full history stays on
    disk for reference (e.g. a potential future /log command).
    """

    def __init__(self):
        self.log_file: Path = config.EXECUTION_LOG_FILE

    def record(self, tool_name: str, args: Dict, result: str, success: bool):
        """
        Records one tool invocation. `result` is stored condensed (first
        ~150 chars) since the log is meant for quick orientation, not full
        output retrieval — the model can always re-run a read-only tool
        (e.g. read_file) if it needs the full content again.
        """
        condensed_result = result.strip().replace("\n", " ")
        if len(condensed_result) > 150:
            condensed_result = condensed_result[:150] + "..."

        # Condense args too — mainly to avoid dumping huge file contents
        # (e.g. create_file's `content` arg) into the log. Newlines are
        # collapsed to spaces so each log entry stays on one line.
        condensed_args = {}
        for k, v in (args or {}).items():
            v_str = str(v).replace("\n", " ")
            condensed_args[k] = v_str if len(v_str) <= 60 else v_str[:57] + "..."

        record = {
            "ts": time.time(),
            "tool": tool_name,
            "args": condensed_args,
            "result": condensed_result,
            "success": success,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, limit: int = config.EXECUTION_LOG_CONTEXT_ENTRIES) -> List[Dict]:
        """Returns the most recent N execution log entries."""
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        return records[-limit:]

    def as_context_string(self, limit: int = config.EXECUTION_LOG_CONTEXT_ENTRIES) -> str:
        """
        Formats the recent execution log as text ready to inject into the
        system prompt, so any model (including one just switched to
        mid-task) immediately knows what was already done in this session.
        """
        entries = self.recent(limit)
        if not entries:
            return ""
        lines = [
            "Recent actions already taken in this session (do not repeat these "
            "unless something actually needs to be redone):"
        ]
        for e in entries:
            when = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            status = "✅" if e["success"] else "❌"
            args_str = ", ".join(f"{k}={v}" for k, v in e.get("args", {}).items())
            lines.append(f"- [{when}] {status} {e['tool']}({args_str}) → {e['result']}")
        return "\n".join(lines)

    def clear(self):
        """Wipes the execution log (e.g. for a fresh task / new conversation topic)."""
        if self.log_file.exists():
            self.log_file.unlink()
