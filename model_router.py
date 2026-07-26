"""
Automatic Model Router.

Automatically switches models in two cases:
  1) Request failure (rate limit / server error / connection issue) after
     exhausting the allowed number of retries.
  2) The request quota allowed for this model in the current session is
     exceeded (max_requests_per_session).

It also detects "hard zero quota" errors (e.g. a free-tier plan where a model
has limit: 0) and skips straight to the next model instead of wasting retries
and backoff time on a request that can never succeed.

Usage stats are stored in a simple JSON file (usage_stats.json) so they can be
read later or shown to the user in a report.
"""
import json
import re
import time
from typing import Optional, Any, Dict, Iterator

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import config
from colors import C


class ModelRouter:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.chain = config.MODEL_CHAIN
        self.current_index = 0
        self.request_counts: Dict[str, int] = {m["name"]: 0 for m in self.chain}
        self.switch_log = []  # log of every switch that happened (for transparency)
        self._status_callback = None  # set via set_status_callback; updates a live line instead of print()
        self._load_stats()

    def set_status_callback(self, callback):
        """
        Registers a function called as callback(message: str) whenever the
        router wants to report progress (retrying, rate limited, switching
        models). If set, these updates go through the callback (e.g. to
        update a single live status line) instead of being printed as
        separate lines. Pass None to fall back to plain print().
        """
        self._status_callback = callback

    def _report_status(self, message: str):
        if self._status_callback:
            try:
                self._status_callback(message)
                return
            except Exception:
                pass  # never let a UI hook crash the router
        print(message)

    @staticmethod
    def _short_error_reason(error: Exception) -> str:
        """
        Extracts a short, human-readable reason from an API error instead of
        dumping its full raw representation (which for quota errors includes
        a large nested JSON blob). Falls back to a generic label if nothing
        useful can be extracted.
        """
        status = getattr(error, "code", None)
        message = getattr(error, "message", None) or str(error)

        if status == 429:
            # Look for "Please retry in Ns" to surface just the wait hint
            retry_match = re.search(r"retry in ([\d.]+)s", message)
            if retry_match:
                seconds = round(float(retry_match.group(1)))
                return f"rate limited, retry in {seconds}s"
            return "rate limited"
        if status in (401, 403):
            return "authentication/permission error"
        if status == 400:
            return "invalid request"
        if status and status >= 500:
            return "temporary server error"
        # Generic fallback: first sentence/line only, capped in length
        first_line = message.strip().split("\n")[0].split(". ")[0]
        return first_line[:80] + ("..." if len(first_line) > 80 else "")

    # ---------- stats management ----------
    def _load_stats(self):
        if config.STATS_FILE.exists():
            try:
                self.stats = json.loads(config.STATS_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.stats = {}
        else:
            self.stats = {}

    def _save_stats(self):
        config.STATS_FILE.write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _bump_stat(self, model_name: str, kind: str):
        self.stats.setdefault(model_name, {"success": 0, "failure": 0, "switches_away": 0})
        self.stats[model_name][kind] = self.stats[model_name].get(kind, 0) + 1
        self._save_stats()

    # ---------- current model info ----------
    @property
    def current_model(self) -> Dict:
        return self.chain[self.current_index]

    @property
    def current_model_name(self) -> str:
        return self.current_model["name"]

    def _has_next_model(self) -> bool:
        return self.current_index < len(self.chain) - 1

    def _advance_model(self, reason: str):
        """Moves to the next model in the chain and logs the reason."""
        old_name = self.current_model_name
        if self._has_next_model():
            self.current_index += 1
            new_name = self.current_model_name
            self._bump_stat(old_name, "switches_away")
            entry = {
                "ts": time.time(),
                "from": old_name,
                "to": new_name,
                "reason": reason,
            }
            self.switch_log.append(entry)
            self._report_status(f"{C.BLUE}Switching model: {old_name} → {new_name}{C.RESET}")
        else:
            self._report_status(f"{C.RED}No fallback model left after {old_name}.{C.RESET}")

    def _check_quota_limit(self):
        """Checks whether the session request limit for the current model was
        reached, and switches to the next model if so."""
        limit = self.current_model.get("max_requests_per_session")
        if limit is not None and self.request_counts[self.current_model_name] >= limit:
            self._advance_model(reason=f"session request limit reached ({limit})")

    @staticmethod
    def _is_zero_quota_error(error: Exception) -> bool:
        """
        Detects a 'hard zero quota' condition, i.e. the account/plan simply has
        no quota at all for this model (limit: 0), which is permanent and not
        worth retrying/backing off on. This is common on free-tier plans where
        a given model (e.g. a Pro-tier model) isn't included at all.
        """
        details = getattr(error, "details", None)
        message = getattr(error, "message", None)
        combined = ""
        if details:
            try:
                combined += json.dumps(details)
            except TypeError:
                combined += str(details)
        if message:
            combined += " " + str(message)
        if not combined:
            combined = str(error)
        # Matches both JSON-style '"limit": 0' and free-text 'limit: 0'
        return bool(re.search(r'\blimit["\']?\s*:\s*0\b', combined))

    def _build_config(self, system_instruction, tools) -> types.GenerateContentConfig:
        """
        Builds the request config. Critically, this explicitly disables the
        google-genai SDK's built-in "automatic function calling" (AFC) via
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True).

        Why this matters: when `tools=` is a list of plain Python functions
        (as ours is, in tools.ALL_TOOLS), the SDK enables AFC by default —
        meaning IT will call those functions itself, in addition to us
        parsing function_calls and running them manually in agent.py's
        _run_tool_calls(). Without this explicit disable, tools can run
        twice (once via the SDK's AFC, once via our own loop), and worse: if
        a tool call is interrupted (e.g. Ctrl+C during time.sleep in
        stop_background_process) while the SDK's AFC path is invoking it,
        the exception surfaces from deep inside google/genai/_extra_utils.py
        instead of our own code, which is confusing to debug and was exactly
        what produced a KeyboardInterrupt traceback rooted in
        _extra_utils.get_function_response_parts instead of our own
        _run_tool_calls. Disabling AFC makes agent.py's manual loop the only
        thing that ever invokes a tool function.
        """
        return types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            tools=tools if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            if tools else None,
        )

    def _classify_error(self, model_name: str, error: Exception):
        """
        Records the failure and decides what to do about it:
        returns ("retry", wait_seconds) to retry the same model after waiting,
        or ("switch", None) to move on to the next model immediately.
        """
        self._bump_stat(model_name, "failure")
        short_reason = self._short_error_reason(error)

        if isinstance(error, genai_errors.ClientError):
            status = getattr(error, "code", None)
            if status == 429 and self._is_zero_quota_error(error):
                self._report_status(f"{C.BLUE}{model_name}: no quota on this plan, switching model{C.RESET}")
                return "switch", None
            elif status == 429:
                wait = config.RETRY_BACKOFF_BASE
                self._report_status(f"{C.BLUE}{model_name}: {short_reason}{C.RESET}")
                return "retry", wait
            else:
                # Bad key, invalid request, etc. — retrying won't help.
                self._report_status(f"{C.BLUE}{model_name}: {short_reason}, switching model{C.RESET}")
                return "switch", None

        elif isinstance(error, genai_errors.ServerError):
            wait = config.RETRY_BACKOFF_BASE
            self._report_status(f"{C.BLUE}{model_name}: {short_reason}{C.RESET}")
            return "retry", wait

        else:
            wait = config.RETRY_BACKOFF_BASE
            self._report_status(f"{C.BLUE}{model_name}: {short_reason}{C.RESET}")
            return "retry", wait

    # ---------- main entrypoint (non-streaming) ----------
    def generate(self, contents, system_instruction: Optional[str] = None,
                 tools: Optional[list] = None, **kwargs) -> Any:
        """
        Sends a generation request to the current model. On failure it retries,
        and if failures persist it automatically switches to the next model in
        the chain until one succeeds or the chain is exhausted.
        """
        while True:
            self._check_quota_limit()
            model_name = self.current_model_name
            gen_config = self._build_config(system_instruction, tools)
            attempts = 0
            last_error = None

            while attempts < config.RETRIES_PER_MODEL:
                attempts += 1
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=gen_config,
                    )
                    self.request_counts[model_name] += 1
                    self._bump_stat(model_name, "success")
                    return response
                except Exception as e:
                    last_error = e
                    action, wait = self._classify_error(model_name, e)
                    if action == "retry" and attempts < config.RETRIES_PER_MODEL:
                        time.sleep(wait)
                        continue
                    else:
                        break  # give up on this model, fall through to switch logic

            if self._has_next_model():
                self._advance_model(reason=self._short_error_reason(last_error) if last_error else "repeated failure")
                continue
            else:
                raise RuntimeError(f"All models in the chain failed. Last error: {last_error}")

    # ---------- main entrypoint (streaming) ----------
    def generate_stream(self, contents, system_instruction: Optional[str] = None,
                         tools: Optional[list] = None, **kwargs) -> Iterator[str]:
        """
        Same as generate(), but yields text chunks as they arrive instead of
        waiting for the full reply. If a stream fails partway through (or
        before producing anything), the same retry/switch logic applies and a
        fresh stream is started — note this means if a stream fails after
        already yielding some text, those chunks were already shown to the
        user before the retry, which is the expected/acceptable trade-off for
        real-time output.
        """
        while True:
            self._check_quota_limit()
            model_name = self.current_model_name
            gen_config = self._build_config(system_instruction, tools)
            attempts = 0
            last_error = None

            while attempts < config.RETRIES_PER_MODEL:
                attempts += 1
                got_any_output = False
                try:
                    for chunk in self.client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=gen_config,
                    ):
                        chunk_text = getattr(chunk, "text", None)
                        if chunk_text:
                            got_any_output = True
                            yield chunk_text
                    # Completed without raising -> success
                    self.request_counts[model_name] += 1
                    self._bump_stat(model_name, "success")
                    return
                except Exception as e:
                    last_error = e
                    if got_any_output:
                        # We already streamed partial content to the user;
                        # don't silently retry (that would duplicate output).
                        # Surface the error so the caller can decide.
                        self._bump_stat(model_name, "failure")
                        raise
                    action, wait = self._classify_error(model_name, e)
                    if action == "retry" and attempts < config.RETRIES_PER_MODEL:
                        time.sleep(wait)
                        continue
                    else:
                        break

            if self._has_next_model():
                self._advance_model(reason=self._short_error_reason(last_error) if last_error else "repeated failure")
                continue
            else:
                raise RuntimeError(f"All models in the chain failed. Last error: {last_error}")

    def report(self) -> str:
        """Simple text report on usage and switches, to show the user."""
        lines = [f"{C.BOLD}📊 Model usage report:{C.RESET}"]
        for m in self.chain:
            name = m["name"]
            s = self.stats.get(name, {"success": 0, "failure": 0, "switches_away": 0})
            marker = f"{C.GREEN}👉{C.RESET}" if name == self.current_model_name else "  "
            lines.append(
                f"{marker} {C.CYAN}{name}{C.RESET}: "
                f"success={C.GREEN}{s.get('success',0)}{C.RESET} "
                f"failure={C.RED}{s.get('failure',0)}{C.RESET} "
                f"requests_this_session={self.request_counts[name]}"
            )
        if self.switch_log:
            lines.append(f"\n{C.BOLD}🔁 Switches this session:{C.RESET}")
            for s in self.switch_log:
                lines.append(f"  - {s['from']} → {s['to']} ({s['reason']})")
        return "\n".join(lines)
