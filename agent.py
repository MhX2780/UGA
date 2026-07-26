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
- Git: git_clone, git_status, git_diff, git_log, git_commit.
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

    def _run_tool_calls(self, response, system_prompt) -> object:
        """
        Executes any function calls in `response` in a loop, feeding results
        back to the model, until it returns a response with no more function
        calls. Returns that final response (not yet streamed/read as text).
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
            response = self.router.generate(
                contents=self.history,
                system_instruction=system_prompt,
                tools=tools.ALL_TOOLS,
            )

        return response

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
    def send(self, user_message: str) -> str:
        self.memory.log_message("user", user_message)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        system_prompt = self._build_system_prompt()

        response = self.router.generate(
            contents=self.history,
            system_instruction=system_prompt,
            tools=tools.ALL_TOOLS,
        )
        response = self._run_tool_calls(response, system_prompt)

        reply_text = response.text or "(No text reply)"

        self.memory.log_message(
            "model", reply_text, meta={"model": self.router.current_model_name}
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))

        self._trim_history()

        return reply_text

    # ---------------- streaming entrypoint: send a message, yield chunks ----------------
    def send_stream(self, user_message: str, image_paths: Optional[List[str]] = None):
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

        Usage:
            for chunk in agent.send_stream("hello"):
                print(chunk, end="", flush=True)
        """
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
            response = self._run_tool_calls(response, system_prompt)
            if getattr(response, "function_calls", None):
                # Extremely unlikely (would mean we hit MAX_TOOL_ROUNDS) —
                # fall back to whatever text we have rather than looping forever.
                full_text_parts.append(response.text or "")
                yield full_text_parts[-1]
            else:
                for chunk_text in self.router.generate_stream(
                    contents=self.history,
                    system_instruction=system_prompt,
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
