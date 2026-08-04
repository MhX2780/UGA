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


def fetch_available_models(api_key: str) -> list:
    """
    Queries the Gemini API directly for every model this specific API key
    actually has access to, filtered down to ones that support text
    generation (generateContent) — i.e. usable by this agent's chat/tool-
    calling flow. This is deliberately used instead of a hardcoded model
    list, since which models exist (and which are free) changes often
    enough that a hardcoded "correct" list goes stale — /settings uses this
    to show the real, current, key-specific list.

    Returns a list of dicts: [{"name": "gemini-3.6-flash",
    "display_name": "Gemini 3.6 Flash", "input_token_limit": ...,
    "output_token_limit": ...}, ...]. Raises the underlying exception on
    network/auth failure — callers should catch and show a clear message
    rather than let /settings crash on a bad key or offline connection.
    """
    client = genai.Client(api_key=api_key)
    models = []
    for m in client.models.list():
        actions = m.supported_actions or []
        if "generateContent" not in actions:
            continue  # skip embedding-only, image-generation-only, etc. models
        # Model names come back as "models/gemini-3.6-flash" — strip the
        # "models/" prefix since that's not part of the name used elsewhere
        # in this codebase (MODEL_CHAIN entries, generate_content calls, etc.)
        name = m.name.split("/", 1)[-1] if m.name else None
        if not name:
            continue
        models.append({
            "name": name,
            "display_name": m.display_name or name,
            "input_token_limit": m.input_token_limit,
            "output_token_limit": m.output_token_limit,
        })
    return models


def run_deep_research(api_key: str, prompt: str, model_name: Optional[str] = None) -> str:
    """
    Runs a single-shot Deep Research request against Google AI Studio's Deep
    Research model family (see config._REFERENCE_DEEP_RESEARCH_MODELS /
    config.DEEP_RESEARCH_MODEL). This is a genuinely different capability
    from the normal chat models in MODEL_CHAIN: Deep Research is Google's
    own autonomous multi-step web-research agent — it plans a research
    strategy, issues its own searches, and returns a long-form cited report
    — rather than a plain text-in/text-out completion. Because of that
    different shape (much longer running time, no meaningful "streaming
    partial answer" UX, no failover chain that makes sense for it), it is
    deliberately NOT wired into ModelRouter's normal generate()/
    generate_stream() failover loop, and is instead called directly here as
    its own explicit action — invoked from the CLI via /deepresearch.

    No retry/failover chain is applied: if the Deep Research model itself
    fails (not GA / not enabled for this key / quota), the exception is
    raised as-is for the caller to report clearly, rather than silently
    falling back to a normal chat model — a plain chat model answering in
    its place would silently produce a much shallower, uncited response
    while claiming to be a "Deep Research" result, which is misleading.
    """
    resolved_model = model_name or config.DEEP_RESEARCH_MODEL
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=resolved_model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if text:
        return text
    # Fall back to walking the raw parts if .text came back empty (e.g. the
    # response is only tool/grounding parts with no plain-text part yet).
    try:
        parts = response.candidates[0].content.parts or []
        return "\n".join(p.text for p in parts if getattr(p, "text", None))
    except Exception:
        return "(Deep Research returned no text content.)"


class ModelRouter:
    def __init__(self, api_key: str):
        # api_key here is kept as the "currently active" key for backward
        # compatibility with any external code constructing ModelRouter
        # directly with a single key — but the router also loads the FULL
        # configured key pool (config.load_api_key_pool()) so it can rotate
        # through additional keys on a daily-quota (RPD) exhaustion, which a
        # single key's quota reset (~24h) can't otherwise recover from. If
        # the given api_key isn't in the pool (e.g. passed in directly by a
        # caller rather than loaded from config), it's used as-is and the
        # pool is only consulted for rotation beyond it.
        pool = config.load_api_key_pool()
        if api_key and api_key not in pool:
            pool = [api_key] + pool
        self.key_pool = pool or [api_key]
        self.current_key_index = self.key_pool.index(api_key) if api_key in self.key_pool else 0

        self.client = genai.Client(api_key=self.key_pool[self.current_key_index])
        self.chain = config.MODEL_CHAIN
        self.current_index = 0
        self.request_counts: Dict[str, int] = {m["name"]: 0 for m in self.chain}
        self.switch_log = []  # log of every switch that happened (for transparency)
        self._status_callback = None  # set via set_status_callback; updates a live line instead of print()
        self._load_stats()

    @property
    def has_multiple_keys(self) -> bool:
        return len(self.key_pool) > 1

    def _rotate_to_next_key(self) -> bool:
        """
        Switches to the next API key in the pool (wrapping around to the
        first if at the end) and rebuilds the client to use it. Returns True
        if a different key was actually switched to, False if there's only
        one key configured (nothing to rotate to). Does NOT reset
        current_index — the model chain restarts from the top on the new
        key deliberately, since a fresh key has its own fresh per-model
        quotas regardless of which model the previous key had settled on.
        """
        if not self.has_multiple_keys:
            return False
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.key_pool)
        self.client = genai.Client(api_key=self.key_pool[self.current_key_index])
        self.current_index = 0  # restart the model chain from the top on the new key
        self._report_status(
            f"{C.BLUE}Daily quota exhausted on API key #{old_index + 1} — "
            f"switching to API key #{self.current_key_index + 1} of {len(self.key_pool)}{C.RESET}"
        )
        return True

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

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """
        Distinguishes a DAILY quota exhaustion (RPD — Requests Per Day) from
        a per-minute rate limit (RPM). This matters because they need
        completely different handling: an RPM error is worth a short
        backoff-and-retry (it resets within a minute), but an RPD error
        resets once every 24 hours — waiting even the longest reasonable
        backoff is pointless, and the only way to keep working right now is
        to switch to a different model or a different API key. Google's
        error payload names the specific quota that was hit (visible in the
        'quotaId' field), e.g.:
          - "GenerateRequestsPerDayPerProjectPerModel-FreeTier"  (RPD — daily)
          - "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" (RPM — per-minute)
        so this checks for "PerDay" specifically in that identifier rather
        than guessing from wait-time text (which isn't always present).
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
        return "PerDay" in combined

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
        thinking_config = None
        if config.DEEP_THINKING_ENABLED:
            # thinking_budget=-1 -> "dynamic thinking" (model decides how much
            # to think); a positive int caps it; 0 disables it. Harmless to
            # send to a model that doesn't support thinking — it's simply
            # ignored, so this isn't gated per-model here.
            thinking_config = types.ThinkingConfig(
                thinking_budget=config.DEEP_THINKING_BUDGET,
                include_thoughts=config.DEEP_THINKING_INCLUDE_THOUGHTS,
            )
        return types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            tools=tools if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            if tools else None,
            thinking_config=thinking_config,
        )

    def _classify_error(self, model_name: str, error: Exception):
        """
        Records the failure and decides what to do about it. Returns one of:
          ("retry", wait_seconds)  — retry the SAME model after waiting
                                     (genuine per-minute rate limit — RPM)
          ("switch", None)        — move to the NEXT MODEL in the chain
                                     immediately (zero quota for this model,
                                     or a non-retryable client error)
          ("switch_key", None)    — the DAILY quota (RPD) for this model was
                                     hit on the CURRENT API key. Waiting is
                                     pointless (resets ~24h later), and
                                     unlike a zero-quota error, this isn't
                                     about the model at all — it's the KEY's
                                     daily allowance. If additional API keys
                                     are configured (config pool), the
                                     caller should rotate to the next key and
                                     retry the SAME model chain from the top
                                     on it, rather than just moving to the
                                     next model on the same exhausted key.
        """
        self._bump_stat(model_name, "failure")
        short_reason = self._short_error_reason(error)

        if isinstance(error, genai_errors.ClientError):
            status = getattr(error, "code", None)
            if status == 429 and self._is_zero_quota_error(error):
                self._report_status(f"{C.BLUE}{model_name}: no quota on this plan, switching model{C.RESET}")
                return "switch", None
            elif status == 429 and self._is_daily_quota_error(error):
                self._report_status(f"{C.BLUE}{model_name}: daily quota (RPD) exhausted{C.RESET}")
                return "switch_key", None
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

    # ---------- direct call to a SPECIFIC model (for multi-agent roles) ----------
    def generate_with_model(self, model_name: str, contents, system_instruction: Optional[str] = None,
                             tools: Optional[list] = None) -> Any:
        """
        Sends a request to a SPECIFIC named model (used by the multi-agent
        feature, where each role — classifier/planner/executor/reviewer —
        has its own assigned model rather than using the shared
        chain-switching logic in generate()). Still gets the same retry
        handling for transient errors (rate limits, server errors) as
        generate(), but does NOT permanently advance self.current_index —
        multi-agent roles are independent of whichever model the main
        single-agent chain is currently on.

        If the specific model fails even after retries (e.g. genuinely no
        quota on this plan for it), falls back to running the SAME request
        through the normal generate() chain instead of failing outright —
        so a misconfigured or currently-unavailable role model degrades
        gracefully rather than breaking multi-agent mode entirely.
        """
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
                self.request_counts.setdefault(model_name, 0)
                self.request_counts[model_name] += 1
                self._bump_stat(model_name, "success")
                return response
            except Exception as e:
                last_error = e
                action, wait = self._classify_error(model_name, e)
                if action == "retry" and attempts < config.RETRIES_PER_MODEL:
                    time.sleep(wait)
                    continue
                elif action == "switch_key" and self._rotate_to_next_key():
                    # Retry the SAME role model on the newly-active key —
                    # an RPD exhaustion is about the KEY's daily allowance,
                    # not this specific model, so the model itself is still
                    # worth trying again once a fresh key is active.
                    attempts = 0
                    continue
                else:
                    break

        self._report_status(
            f"{C.BLUE}{model_name} (role model) unavailable, trying fallback chain{C.RESET}"
        )
        return self._generate_fallback_chain(contents, system_instruction, tools, last_error)

    def _generate_fallback_chain(self, contents, system_instruction, tools, original_error) -> Any:
        """
        Tries every model in config.MODEL_CHAIN (in order) as a one-off
        fallback, WITHOUT mutating self.current_index — this is what keeps a
        multi-agent role's fallback from bleeding into (and corrupting)
        the single-agent chain's own "sticky" model selection. Each model
        gets one attempt here (no per-model retry loop — the caller already
        retried the originally-requested model); this is purely "is there
        ANY model in the chain that can answer this one request right now".
        """
        last_error = original_error
        for model_entry in config.MODEL_CHAIN:
            candidate = model_entry["name"]
            try:
                gen_config = self._build_config(system_instruction, tools)
                response = self.client.models.generate_content(
                    model=candidate,
                    contents=contents,
                    config=gen_config,
                )
                self.request_counts.setdefault(candidate, 0)
                self.request_counts[candidate] += 1
                self._bump_stat(candidate, "success")
                return response
            except Exception as e:
                last_error = e
                self._bump_stat(candidate, "failure")
                continue
        raise RuntimeError(f"All models failed for this role-based call. Last error: {last_error}")

    def generate_stream_with_model(self, model_name: str, contents, system_instruction: Optional[str] = None,
                                    tools: Optional[list] = None) -> Iterator[str]:
        """
        Streaming counterpart to generate_with_model() — sends the request to
        a specific named model (for multi-agent roles) and yields text
        chunks as they arrive. Falls back to the normal generate_stream()
        chain if the specific model fails after retries, same reasoning as
        generate_with_model().
        """
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
                self.request_counts.setdefault(model_name, 0)
                self.request_counts[model_name] += 1
                self._bump_stat(model_name, "success")
                return
            except Exception as e:
                last_error = e
                if got_any_output:
                    self._bump_stat(model_name, "failure")
                    raise
                action, wait = self._classify_error(model_name, e)
                if action == "retry" and attempts < config.RETRIES_PER_MODEL:
                    time.sleep(wait)
                    continue
                elif action == "switch_key" and self._rotate_to_next_key():
                    attempts = 0
                    continue
                else:
                    break

        self._report_status(
            f"{C.BLUE}{model_name} (role model) unavailable, trying fallback chain{C.RESET}"
        )
        yield from self._generate_stream_fallback_chain(contents, system_instruction, tools, last_error)

    def _generate_stream_fallback_chain(self, contents, system_instruction, tools, original_error) -> Iterator[str]:
        """Streaming counterpart to _generate_fallback_chain — see its
        docstring for why this avoids self.generate_stream() (which would
        mutate self.current_index and contaminate single-agent state)."""
        last_error = original_error
        for model_entry in config.MODEL_CHAIN:
            candidate = model_entry["name"]
            got_any_output = False
            try:
                gen_config = self._build_config(system_instruction, tools)
                for chunk in self.client.models.generate_content_stream(
                    model=candidate,
                    contents=contents,
                    config=gen_config,
                ):
                    chunk_text = getattr(chunk, "text", None)
                    if chunk_text:
                        got_any_output = True
                        yield chunk_text
                self.request_counts.setdefault(candidate, 0)
                self.request_counts[candidate] += 1
                self._bump_stat(candidate, "success")
                return
            except Exception as e:
                last_error = e
                self._bump_stat(candidate, "failure")
                if got_any_output:
                    # Already streamed partial content under this candidate
                    # — don't silently retry another model and duplicate
                    # output; surface the error same as the main chain does.
                    raise
                continue
        raise RuntimeError(f"All models failed for this role-based streaming call. Last error: {last_error}")

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
            switched_key = False

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
                    elif action == "switch_key" and self._rotate_to_next_key():
                        # Retry the SAME model on the new key rather than
                        # falling through to "advance to next model" below —
                        # a fresh key may well have quota for this exact
                        # model even though the previous key didn't.
                        switched_key = True
                        break
                    else:
                        break  # give up on this model, fall through to switch logic

            if switched_key:
                continue

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
            switched_key = False

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
                    elif action == "switch_key" and self._rotate_to_next_key():
                        switched_key = True
                        break
                    else:
                        break

            if switched_key:
                continue

            if self._has_next_model():
                self._advance_model(reason=self._short_error_reason(last_error) if last_error else "repeated failure")
                continue
            else:
                raise RuntimeError(f"All models in the chain failed. Last error: {last_error}")

    @staticmethod
    def extract_thoughts(response) -> str:
        """
        Pulls out the model's "thought summary" text from a response, when
        Deep Thinking is enabled with include_thoughts=True. The google-genai
        SDK's response.text convenience property deliberately SKIPS parts
        marked thought=True (so normal callers relying on .text never see
        raw thinking output mixed into the answer) — this walks the raw
        candidate parts to surface it separately, for callers (e.g. the CLI)
        that want to show "🧠 Thinking..." content to the user on request.
        Returns "" if there's no thought content (thinking disabled, model
        doesn't support it, or include_thoughts was off).
        """
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return ""
            parts = getattr(candidates[0].content, "parts", None) or []
            thought_parts = [p.text for p in parts if getattr(p, "thought", False) and getattr(p, "text", None)]
            return "\n".join(thought_parts)
        except Exception:
            return ""

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
