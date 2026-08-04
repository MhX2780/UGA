"""
Main Agent class tying together memory + automatic model switching + file tools.

Tool calling is handled manually (rather than relying on the SDK's automatic
function calling) because that combination is more reliable across streaming
and non-streaming calls, and gives us full control to log/notify on each tool
call. The flow is:

  1) Send the conversation so far (non-streaming) so we can inspect whether
     the model wants to call any tools.
  2) If it does, execute them via tools.py, feed the results back, and repeat
     until the model responds with plain text (no more function calls).
  3) Once we know no more tool calls are coming, re-issue that same final
     turn as a *streaming* call so the user sees the reply arrive
     progressively instead of all at once.
"""
from pathlib import Path
from typing import List, Dict, Optional

from google.genai import types

import config
import tools
from memory import MemoryStore, ExecutionLog
from model_router import ModelRouter


SYSTEM_PROMPT_TEMPLATE = """You are an intelligent coding and file-management assistant,
similar to Gemini CLI. You have a wide set of tools available:

- File operations: create_file, read_file, edit_file, delete_file, move_file,
  rename_file, copy_file, create_folder, diff_preview, compare_files.
- Search & discovery: find_file, find_folder, search_in_files, file_stats,
  detect_language, count_files, count_todos, list_files.
- Bulk editing: replace_in_files (find/replace across many files at once).
- Code quality: lint_check (single file), check_file_syntax_all (whole project).
- Git: git_clone, git_fetcher, git_status, git_diff, git_log, git_commit.
  git_fetcher looks up a GitHub repo's metadata (stars, license, description,
  file tree) WITHOUT cloning it — use it first when the user asks about a
  repo you haven't cloned yet, before deciding whether git_clone is needed.
- Execution: run_command, start_background_process, list_background_processes,
  read_background_log, stop_background_process. run_command works on both
  Unix and Windows — common Unix commands (ls, cat, cp, mv, rm, grep, head,
  tail, etc.) are automatically translated to PowerShell equivalents on
  Windows, so use whichever Unix-style command comes naturally.
- Dependencies: list_dependencies, add_dependency (auto-detects npm vs pip).
- Testing: run_tests (auto-detects pytest/unittest/npm test), create_test_file
  (scaffolds a starter test file for a given source file).
- Documentation: generate_readme, extract_docstrings.
- Checkpoints: save_checkpoint/load_checkpoint/list_checkpoints — a full
  project snapshot you can save before a risky change and restore later,
  stronger than undo_last_change (which only reverts one step).
- Networking: check_port_in_use (check before starting a dev server on a port),
  http_request (send GET/POST/etc. requests, e.g. to test a local API).
- Archives: create_zip, extract_zip.
- Environment: env_var_check.
- Code analysis: count_lines_of_code, find_unused_imports (Python only).
- Format conversion: convert_file_format (json<->yaml, json<->csv), minify_file
  (json/css/js).
- Images: Image_Fetch (look at an image already in the workspace and answer a
  question about it, or describe it — use this to verify a generated image,
  read text/diagrams in a screenshot, etc.), Image_Create (generate a new
  image from a text description and save it to the workspace).
  Image_Fetch_Puter and Image_Create_Puter [BETA] are Puter.js fallbacks for
  the same two operations — ONLY call them after Image_Fetch/Image_Create
  has failed AND the user has explicitly agreed to try Puter.js instead
  (their failure messages will tell you a fallback is available; ask the
  user before using it, never call the _Puter variant proactively or as a
  first choice). Be candid that Image_Create_Puter specifically is
  unverified and may simply fail — don't imply it's an equally reliable
  substitute for Gemini's image generation.
- Screen: view_screen takes ONE screenshot of the user's actual screen right
  now and describes/answers a question about it. This is EVENT-DRIVEN ONLY —
  call it once when the user asks about their screen or right after an
  action whose result is visible on screen (e.g. after launching an
  installer). NEVER call it repeatedly in a loop to "watch" the screen over
  time — that wastes requests and hits rate limits fast. If you genuinely
  need to check again later after telling the user to wait, use
  only_if_changed=True so an unchanged screen doesn't cost an extra request.
  view_screen_puter (BETA) does the same single-shot capture but analyzes it
  via Puter.js instead of Gemini — only use it as a fallback or if the user
  asks for Puter explicitly.
- watch_screen is the tool for actual "live"/ongoing screen monitoring: it
  loops for you (up to 10 minutes per call), taking a screenshot every
  interval_seconds, skipping unchanged frames automatically, and returns one
  combined log of everything it saw. Use this instead of manually looping
  view_screen when the user wants you to watch continuously (e.g. "watch me
  work and tell me if I make a mistake", "let me know when the build
  finishes"). It can route analysis through Puter.js instead of Gemini via
  use_puter=True.
- Available_Active_Windows and List_System_Processes are SENSITIVE tools
  gated behind an explicit, user-only permission switch
  (config.SYSTEM_ACCESS_ENABLED). If either returns a permission error, do
  NOT retry it — instead ask the user for permission IN ENGLISH (e.g. "Do
  you allow the AI to access your open windows and running system
  processes?") and tell them to run /settings system access on if they
  agree. Never suggest the user enable this unless the task genuinely needs
  it, and never claim to have enabled it yourself — only the user can.
  Available_Active_Windows lists every open window (optionally with a
  screenshot+description of each). List_System_Processes is a Task-Manager-
  style listing of ALL system processes (PID/CPU/memory), distinct from the
  narrower list_background_processes (which only shows processes this agent
  itself started, and needs no special permission).
- Safety net: undo_last_change (reverts your most recent file change).

Important rules:
- Actually use the tools to carry out any file or command-related request; don't just
  describe what you would do.
- Prefer the specific tool for the job over run_command when one exists (e.g. use
  find_file instead of "run_command('find . -name ...')", use git_status/git_commit
  instead of raw git shell commands, use git_clone instead of "run_command('git clone
  ...')", use replace_in_files instead of manually editing each file one by one).
- For replace_in_files, use dry_run=True first on any broad or risky rename to see
  what would change before actually applying it, and use whole_word=True when
  renaming a variable/identifier (so "name" doesn't also match inside "username").
- After using Image_Create, you can optionally use Image_Fetch on the result if you
  need to verify what was actually generated (e.g. to check it matches the request)
  before telling the user it's done.
- Before making a large edit to an important file, prefer using diff_preview to
  preview the change first.
- After writing or editing code files, consider running lint_check on them to catch
  syntax errors or issues immediately. For broader changes, check_file_syntax_all can
  sanity-check the whole project at once.
- run_command automatically detects long-running server/dev commands (npm run dev,
  flask run, uvicorn, vite, python -m http.server, etc.) and starts them in the
  BACKGROUND instead of waiting — it returns right away with a PID. Use
  read_background_log(pid) to check on it and stop_background_process(pid) to stop it.
  Never assume a server command "failed" just because run_command returned quickly with
  a PID — that's the expected, successful behavior for servers.
- When running commands, prefer non-interactive flags (e.g. "npm install --yes") so
  the command doesn't hang waiting for input.
- Be precise and concise in your text replies, and let the tools handle execution.
- If a tool returns an error, explain what happened to the user and suggest a fix.

{memory_context}

{execution_log_context}
"""

# Maps tool names to their actual functions, for manual dispatch.
TOOL_MAP = {func.__name__: func for func in tools.ALL_TOOLS}

# Safety cap on how many tool-call rounds we'll do in a single turn, in case
# the model gets stuck calling tools in a loop.
MAX_TOOL_ROUNDS = 15

_IMAGE_MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}


def _guess_image_mime_type(path: Path) -> Optional[str]:
    """Returns the MIME type for a supported image extension, or None if the
    file's extension isn't a supported image format."""
    return _IMAGE_MIME_TYPES.get(path.suffix.lower())


class GeminiAgent:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or config.GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "No GEMINI_API_KEY was provided. Pass it explicitly or save it via the CLI."
            )
        self.memory = MemoryStore()
        self.execution_log = ExecutionLog()
        self.router = ModelRouter(api_key=api_key)
        self.history: List[types.Content] = []
        self._load_history_from_memory()

    def _load_history_from_memory(self):
        recent = self.memory.load_recent_session()
        for rec in recent:
            role = "user" if rec["role"] == "user" else "model"
            if rec["role"] in ("user", "model"):
                self.history.append(
                    types.Content(role=role, parts=[types.Part(text=rec["content"])])
                )

    def _build_system_prompt(self) -> str:
        memory_context = self.memory.memory_as_context_string()
        execution_log_context = self.execution_log.as_context_string()
        return SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context or "(No saved information about the user yet)",
            execution_log_context=execution_log_context or "(No actions taken yet this session)",
        )

    def _run_tool_calls(self, response, model_name: Optional[str] = None) -> object:
        """
        Executes any function calls in `response` in a loop, feeding results
        back to the model, until it returns a response with no more function
        calls. Returns that final response (not yet streamed/read as text).

        Critically, the system prompt is rebuilt FRESH before every single
        round-trip to the model (not passed in once and reused) — this is
        what keeps the injected execution_log_context accurate as tools
        actually run. Reusing a stale system prompt built before any tool
        had executed meant every subsequent round (and especially the model
        seen right after an automatic model-switch mid-task) still saw
        "(No actions taken yet this session)" even after several tools had
        already completed successfully, making it look like nothing had
        been done yet — which is exactly why a model resuming after a
        switch would redo work a previous model already finished.

        Args:
            response: the initial response (already known to have function_calls)
            model_name: if given, every round-trip in this loop goes to this
                SPECIFIC model via generate_with_model (used by the
                multi-agent executor role, which has its own assigned
                model) instead of the normal auto-switching chain via
                generate(). Single-agent mode leaves this as None.
        """
        rounds = 0
        while getattr(response, "function_calls", None) and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            # Save the model's turn (which requested the function call(s))
            self.history.append(response.candidates[0].content)

            for fc in response.function_calls:
                func_name = fc.name
                func_args = fc.args or {}

                if func_name in TOOL_MAP:
                    try:
                        tool_result = TOOL_MAP[func_name](**func_args)
                        # A tool returning a string starting with an error
                        # glyph (our own convention across tools.py) still
                        # counts as a "success" at the Python level (no
                        # exception raised), but we want the execution log
                        # to reflect the actual outcome the model saw.
                        call_succeeded = not str(tool_result).lstrip().startswith(("❌", "🚫"))
                    except Exception as e:
                        tool_result = f"❌ Error executing {func_name}: {e}"
                        call_succeeded = False
                else:
                    tool_result = f"❌ Error: Tool {func_name} not found."
                    call_succeeded = False

                self.execution_log.record(func_name, func_args, tool_result, call_succeeded)

                self.history.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": tool_result},
                            )
                        ],
                    )
                )

            # Ask the model to continue now that it has the tool result(s).
            # Rebuilt fresh (not reusing the caller's original prompt) so it
            # reflects everything recorded in execution_log above, including
            # what just happened in this very round.
            fresh_system_prompt = self._build_system_prompt()
            if model_name:
                response = self.router.generate_with_model(
                    model_name=model_name,
                    contents=self.history,
                    system_instruction=fresh_system_prompt,
                    tools=tools.ALL_TOOLS,
                )
            else:
                response = self.router.generate(
                    contents=self.history,
                    system_instruction=fresh_system_prompt,
                    tools=tools.ALL_TOOLS,
                )

        return response

    def _run_tool_calls_with_model(self, response, model_name: str) -> object:
        """Convenience wrapper: same as _run_tool_calls(response, model_name=...),
        used by multi_agent.py so its call sites read a bit more explicitly."""
        return self._run_tool_calls(response, model_name=model_name)

    # ---------------- Puter.js tool-calling loop (BETA) ----------------
    def _run_puter_tool_rounds(self, model_name: str, messages: list, openai_tools: list) -> tuple:
        """
        BETA. Shared core of the Puter tool-calling loop: executes any
        tool_calls the model returns, feeding results back, until it
        returns a turn with no more tool calls. Returns (messages,
        final_message) — the updated OpenAI-format messages list and the
        final response message object (whose .content is the answer, once
        no tool_calls remain).

        This part is always non-streaming even when the overall turn will
        be streamed (see _send_via_puter_with_tools_stream below): a tool
        call isn't actionable until its full JSON arguments string has
        arrived, so there's nothing useful to show the user chunk-by-chunk
        while the model is still deciding whether/how to call a tool.
        Streaming only pays off once we reach the final plain-text answer,
        which is exactly what the two methods below split between them —
        matching the same non-streaming-tool-rounds-then-stream-the-reply
        pattern agent.py already uses for Gemini in send_stream().
        """
        import providers
        import json

        rounds = 0
        response = providers.puter_chat_with_tools(model_name, messages, tools=openai_tools)

        while rounds < MAX_TOOL_ROUNDS:
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return messages, message

            rounds += 1
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments or "{}")
                except (ValueError, TypeError):
                    func_args = {}

                if func_name in TOOL_MAP:
                    try:
                        tool_result = TOOL_MAP[func_name](**func_args)
                        call_succeeded = not str(tool_result).lstrip().startswith(("❌", "🚫"))
                    except Exception as e:
                        tool_result = f"❌ Error executing {func_name}: {e}"
                        call_succeeded = False
                else:
                    # A model hallucinating a tool that doesn't exist is one
                    # of the specific BETA risks noted in config.py — surface
                    # it to the model as an error rather than crashing, so
                    # it has a chance to self-correct on the next round.
                    tool_result = f"❌ Error: Tool {func_name} not found."
                    call_succeeded = False

                self.execution_log.record(func_name, func_args, tool_result, call_succeeded)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result),
                })

            # Rebuild the system prompt fresh (same reasoning as
            # _run_tool_calls: keep execution_log_context accurate) and
            # refresh it in-place at index 0 rather than re-appending, since
            # OpenAI-format conversations keep a single system message.
            messages[0] = {"role": "system", "content": self._build_system_prompt()}
            response = providers.puter_chat_with_tools(model_name, messages, tools=openai_tools)

        # Hit MAX_TOOL_ROUNDS without a final plain-text answer.
        return messages, response.choices[0].message

    def _build_puter_user_message(self, user_message: str, image_paths: Optional[List[str]] = None) -> dict:
        """
        Builds a single OpenAI-format user message, optionally with image
        attachments encoded as inline data: URIs — the same content-part
        convention already used by providers.puter_vision_describe() for
        Image_Fetch_Puter, and the standard OpenAI chat-completions image
        input format generally. If image_paths is given, `content` becomes
        a list of {"type": "text"/"image_url", ...} parts instead of a
        plain string (per the OpenAI spec — a message can't mix a bare
        string with image parts).

        Unreadable/unsupported images are skipped individually (with a
        printed-to-status warning is NOT done here — this is a pure
        builder function) rather than failing the whole message, matching
        send_stream()'s existing behavior for the Gemini path.
        """
        import base64

        if not image_paths:
            return {"role": "user", "content": user_message}

        content_parts = [{"type": "text", "text": user_message}]
        for img_path in image_paths:
            img_file = Path(img_path)
            mime_type = _guess_image_mime_type(img_file)
            if mime_type is None:
                continue  # unsupported/unknown type — skip rather than fail the whole message
            try:
                data = img_file.read_bytes()
            except OSError:
                continue
            b64_data = base64.b64encode(data).decode("ascii")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            })
        return {"role": "user", "content": content_parts}

    def _send_via_puter_with_tools(self, model_name: str, user_message: str,
                                    image_paths: Optional[List[str]] = None) -> str:
        """
        BETA, non-streaming. See _run_puter_tool_rounds() for the loop
        itself; this just sets up the initial messages and unpacks the
        final answer. Kept for callers that want a plain string back
        (e.g. send()) rather than a chunk generator — use
        send_stream(puter_model=...) instead for progressive output.

        Args:
            image_paths: optional list of filesystem paths to attach to
                the user's message — see _build_puter_user_message(). Only
                takes effect on models that actually support vision input;
                a non-vision Puter model will likely just ignore the image
                parts or error, same caveat as everywhere else in this
                BETA path.
        """
        import tool_schemas

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt},
                    self._build_puter_user_message(user_message, image_paths)]
        openai_tools = tool_schemas.build_openai_tools_schema(tools.ALL_TOOLS)

        _, final_message = self._run_puter_tool_rounds(model_name, messages, openai_tools)
        return final_message.content or "(No text reply)"

    def _send_via_puter_with_tools_stream(self, model_name: str, user_message: str,
                                           image_paths: Optional[List[str]] = None):
        """
        BETA, streaming. Same overall shape as send_stream()'s Gemini path:
        any tool-call rounds happen first (non-streaming, via
        _run_puter_tool_rounds — see its docstring for why), and once the
        model has nothing left to call, that final answer is re-requested
        as a stream so the user sees it arrive progressively instead of
        all at once. Yields plain text chunks.

        Args:
            image_paths: optional list of filesystem paths to attach —
                see _build_puter_user_message()'s docstring for the
                encoding and its vision-model caveat.
        """
        import providers
        import tool_schemas

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt},
                    self._build_puter_user_message(user_message, image_paths)]
        openai_tools = tool_schemas.build_openai_tools_schema(tools.ALL_TOOLS)

        # First, do a normal (non-streaming) round to see if the model
        # wants to call any tools at all — same reasoning as Gemini's
        # send_stream(): avoids a wasted extra round-trip for the common
        # "no tools needed" case, since we'd otherwise have to stream it
        # once just to inspect for tool_calls and then stream it again.
        response = providers.puter_chat_with_tools(model_name, messages, tools=openai_tools)
        message = response.choices[0].message

        if getattr(message, "tool_calls", None):
            messages, final_message = self._run_puter_tool_rounds(model_name, messages, openai_tools)
            if getattr(final_message, "tool_calls", None):
                # Hit MAX_TOOL_ROUNDS — fall back to whatever text we have
                # rather than looping forever or streaming nothing.
                yield final_message.content or "(No text reply — hit max tool-call rounds)"
                return
            # Refresh the system message once more before the final
            # streamed call, same reasoning as _run_puter_tool_rounds does
            # between rounds — keeps execution_log_context accurate for
            # the answer the user is about to see.
            messages[0] = {"role": "system", "content": self._build_system_prompt()}
            for event in providers.puter_chat_with_tools_stream(model_name, messages, tools=openai_tools):
                if event["type"] == "text":
                    yield event["text"]
                # A "tool_calls" event here (the model changing its mind
                # and requesting yet another tool call instead of finally
                # answering) is deliberately not executed — MAX_TOOL_ROUNDS
                # non-streaming rounds already ran above, so at this point
                # we commit to showing whatever text came back rather than
                # opening another non-streaming detour mid-stream.
        else:
            # No tools needed — this first response IS the final answer,
            # so re-issuing it as a stream would double the API call for
            # something as simple as "hi". Just yield its text as one chunk,
            # matching Gemini's send_stream() behavior for the same case.
            yield message.content or "(No text reply)"


    def _trim_history(self):
        """
        Trims self.history down to config.MAX_HISTORY_MESSAGES entries when
        it grows too large — but unlike a naive slice, this respects
        function_call/function_response pairing. Each tool call adds at
        least two Content entries (the model's turn requesting the call,
        then our "tool" role turn with the result); cutting the history
        list at an arbitrary boundary can leave a dangling function_response
        with no preceding function_call (or vice versa), which the Gemini
        API can reject or mishandle on the next turn. To avoid that, we only
        ever trim starting from a "user" role turn — the natural start of a
        complete conversational exchange — so a trimmed history always
        begins with a clean, self-contained turn.
        """
        if len(self.history) <= config.MAX_HISTORY_MESSAGES:
            return

        # Walk backwards from the count-based cutoff to find the nearest
        # preceding "user" turn, so the kept history starts cleanly there
        # rather than mid-way through a tool-call exchange.
        cutoff = len(self.history) - config.MAX_HISTORY_MESSAGES
        for i in range(cutoff, len(self.history)):
            if self.history[i].role == "user":
                self.history = self.history[i:]
                return
        # No "user" turn found in the range being trimmed (unlikely, but
        # possible if a single exchange used more tool calls than
        # MAX_HISTORY_MESSAGES) — fall back to keeping everything rather
        # than risk cutting a function_call/function_response pair apart.

    # ---------------- main entrypoint: send a message (non-streaming) ----------------
    def send(self, user_message: str, puter_model: Optional[str] = None,
              image_paths: Optional[List[str]] = None) -> str:
        """
        Args:
            user_message: the text of the message.
            puter_model: BETA. If given (a Puter model name/id), this turn
                is routed entirely through Puter instead of the normal
                Gemini chain, via _send_via_puter_with_tools() — only takes
                effect when config.PUTER_CHAT_ENABLED and
                config.PUTER_TOOL_CALLING_ENABLED are both on. Not wired up
                to any automatic "currently active chat model" selection
                yet (PUTER_CHAT_ENABLED alone doesn't pick a model today —
                see /settings puter tools' help text) — a caller (e.g. a
                future CLI command) must pass the specific Puter model
                name explicitly for now.
            image_paths: optional list of filesystem paths to attach —
                works for both the Gemini and Puter (BETA) paths; see
                _build_puter_user_message()'s docstring for the Puter
                encoding and its vision-model caveat (not every Puter
                model understands images even when the request is
                formatted correctly).
        """
        if puter_model and config.PUTER_CHAT_ENABLED and config.PUTER_TOOL_CALLING_ENABLED:
            self.memory.log_message("user", user_message)
            reply_text = self._send_via_puter_with_tools(puter_model, user_message, image_paths=image_paths)
            self.memory.log_message("model", reply_text, meta={"model": puter_model, "provider": "puter"})
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
            self.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))
            self._trim_history()
            return reply_text

        self.memory.log_message("user", user_message)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        system_prompt = self._build_system_prompt()

        response = self.router.generate(
            contents=self.history,
            system_instruction=system_prompt,
            tools=tools.ALL_TOOLS,
        )
        response = self._run_tool_calls(response)
        reply_text = response.text or "(No text reply)"

        self.memory.log_message(
            "model", reply_text, meta={"model": self.router.current_model_name}
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))

        self._trim_history()

        return reply_text

    # ---------------- streaming entrypoint: send a message, yield chunks ----------------
    def send_stream(self, user_message: str, image_paths: Optional[List[str]] = None,
                     _skip_multi_agent: bool = False, puter_model: Optional[str] = None):
        """
        Same as send(), but the final text reply is streamed chunk by chunk.
        Tool calls (if any) still happen as non-streaming round-trips first
        (needed to inspect function_calls), then once the model is ready to
        give its final plain-text answer, that last turn is streamed so the
        user sees it appear progressively instead of all at once.

        Args:
            user_message: the text of the message
            image_paths: optional list of absolute filesystem paths to image
                files to attach to this message (e.g. from the CLI's /image
                command). These are read directly from disk — unlike
                Image_Fetch (which the model calls itself for images already
                inside the workspace), this is for the user handing images
                to the model as part of their own message.
            _skip_multi_agent: internal flag used by MultiAgentOrchestrator
                itself when it hands a "simple" classified request back to
                this method, to avoid re-classifying and looping forever.
                Not meant to be passed by normal callers.
            puter_model: BETA. If given, routes this entire turn through
                Puter (streamed) instead of Gemini — see send()'s docstring
                for the same parameter for requirements/caveats.
                image_paths (if also given) are attached to the Puter
                message too — see _build_puter_user_message()'s docstring
                for the encoding and its vision-model caveat: not every
                Puter model understands image input even when the request
                is formatted correctly, so results may vary more here than
                on the well-tested Gemini path.

        Usage:
            for chunk in agent.send_stream("hello"):
                print(chunk, end="", flush=True)
        """
        if puter_model and config.PUTER_CHAT_ENABLED and config.PUTER_TOOL_CALLING_ENABLED:
            self.memory.log_message("user", user_message)
            full_text_parts = []
            for chunk_text in self._send_via_puter_with_tools_stream(puter_model, user_message, image_paths=image_paths):
                full_text_parts.append(chunk_text)
                yield chunk_text
            reply_text = "".join(full_text_parts) or "(No text reply)"
            self.memory.log_message("model", reply_text, meta={"model": puter_model, "provider": "puter"})
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
            self.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))
            self._trim_history()
            return

        if config.MULTI_AGENT_ENABLED and not _skip_multi_agent and not image_paths:
            # Multi-agent mode is on: hand off to the orchestrator, which
            # yields MultiAgentEvent objects (not plain text chunks) — the
            # CLI checks config.MULTI_AGENT_ENABLED itself and calls
            # run_multi_agent_turn() directly in that case rather than
            # send_stream(), so reaching here with multi-agent enabled
            # would only happen via a direct send_stream() call bypassing
            # the CLI's own dispatch — still handled safely by just running
            # single-agent mode instead of erroring.
            pass  # fall through to normal single-agent flow below

        self.memory.log_message("user", user_message)

        user_parts = [types.Part(text=user_message)]
        if image_paths:
            for img_path in image_paths:
                img_file = Path(img_path)
                mime_type = _guess_image_mime_type(img_file)
                if mime_type is None:
                    continue  # unsupported/unknown type — skip rather than fail the whole message
                user_parts.append(
                    types.Part.from_bytes(data=img_file.read_bytes(), mime_type=mime_type)
                )
        self.history.append(types.Content(role="user", parts=user_parts))

        system_prompt = self._build_system_prompt()

        # First, check (non-streaming) whether the model wants to call any
        # tools. Most turns involving file operations need at least one tool
        # round-trip, which requires inspecting function_calls — that can't
        # be done mid-stream. Once we're in a round with no tool calls
        # pending, we switch to a streaming call for that turn so the user
        # sees the final answer arrive progressively.
        response = self.router.generate(
            contents=self.history,
            system_instruction=system_prompt,
            tools=tools.ALL_TOOLS,
        )

        full_text_parts = []

        if getattr(response, "function_calls", None):
            # Run all tool-call rounds (non-streaming) until the model has
            # nothing left to call, then stream just the final turn.
            response = self._run_tool_calls(response)
            if getattr(response, "function_calls", None):
                # Extremely unlikely (would mean we hit MAX_TOOL_ROUNDS) —
                # fall back to whatever text we have rather than looping forever.
                full_text_parts.append(response.text or "")
                yield full_text_parts[-1]
            else:
                # Rebuilt fresh here too — same reasoning as inside
                # _run_tool_calls: by this point several tools may have
                # already run, and the final streamed answer should be
                # generated with a system prompt that actually reflects
                # that (fresh execution_log_context), not the one built
                # before any of this turn's tool calls happened.
                final_system_prompt = self._build_system_prompt()
                for chunk_text in self.router.generate_stream(
                    contents=self.history,
                    system_instruction=final_system_prompt,
                    tools=tools.ALL_TOOLS,
                ):
                    full_text_parts.append(chunk_text)
                    yield chunk_text
        else:
            # No tools needed at all — this first response IS the final
            # answer, so re-issuing it as a stream would double the API call
            # for something as simple as "hi". Just yield its text as one
            # chunk instead.
            full_text_parts.append(response.text or "")
            yield full_text_parts[-1]

        reply_text = "".join(full_text_parts) or "(No text reply)"

        self.memory.log_message(
            "model", reply_text, meta={"model": self.router.current_model_name}
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))

        self._trim_history()

    # ---------------- manual long-term memory management ----------------
    def remember(self, key: str, value: str, category: str = "general"):
        self.memory.remember(key, value, category)

    def recall_all(self) -> Dict:
        return self.memory.recall_all()

    def usage_report(self) -> str:
        return self.router.report()

    # ---------------- execution log ----------------
    def recent_actions(self, limit: int = 15) -> List[Dict]:
        """Returns the most recent tool-call actions and their outcomes."""
        return self.execution_log.recent(limit)

    def clear_execution_log(self):
        """Wipes the execution log — useful when starting a clearly new task
        so old actions don't clutter context for no reason."""
        self.execution_log.clear()

    # ---------------- multi-agent mode ----------------
    def run_multi_agent_turn(self, user_message: str):
        """
        Runs a user turn through the multi-agent pipeline (classifier ->
        either the normal single-agent flow for simple requests, or
        planner -> executor(s) -> reviewer for complex ones). Yields
        multi_agent.MultiAgentEvent objects for the CLI to render live —
        see multi_agent.py's module docstring for the event kinds.

        Only used when config.MULTI_AGENT_ENABLED is True (toggled via
        /settings in the CLI); the CLI checks that flag itself and calls
        this instead of send_stream() when it's on.
        """
        from multi_agent import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(self)
        yield from orchestrator.run_turn(user_message)
