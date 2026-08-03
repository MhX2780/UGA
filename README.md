# Gemini Agent CLI

A terminal-based AI coding agent powered by the Gemini API — similar in spirit to
Gemini CLI or Claude Code, built from scratch with a focus on **persistent
memory**, **automatic model failover**, and a large toolbox for real file/code
work, all in a single-file-per-concern Python codebase with no heavyweight
framework dependencies.
```
╭ 🤖 ────────────────────────────────────────────────────────────────────────╮
│ Gemini Agent · CLI                                                         │
│ Persistent memory · automatic model switching · live file activity         │
╰─────────────────────────────────────────────────────────────────────────────╯
```

## Features

- **Persistent memory** — remembers facts about you and recent actions across
  restarts (JSONL-based, no database required).
- **Automatic model failover** — if a model hits a rate limit or has no quota
  on your plan, it automatically retries and switches to the next model in a
  configurable chain, with clear on-screen status.
- **52 tools** covering file operations, search, git, testing, dependencies,
  checkpoints, images, HTTP requests, format conversion, and more (full list
  below).
- **Cross-platform shell commands** — common Unix commands (`ls`, `grep`,
  `cp`, `rm`, `cat`, `head`, `tail`, ...) are automatically translated to
  PowerShell equivalents on Windows, so the agent works the same way whether
  you're on Linux, macOS, Termux, or Windows.
- **Background process handling** — dev servers (`npm run dev`, `flask run`,
  etc.) are automatically detected and run in the background instead of
  hanging the agent forever.
- **Live status line** — a single, in-place-updating line shows what the
  agent is doing (thinking, running a command, editing a file) without
  spamming your terminal.
- **Sandboxed workspace** — all file operations are confined to a `workspace/`
  directory; the agent can't touch files outside it.
- **Image support** — attach images to your messages (`/image`), or let the
  agent look at (`Image_Fetch`) or generate (`Image_Create`) images itself.
- **Undo & checkpoints** — step-by-step undo for any file change, plus full
  project snapshots you can save and restore.

## Requirements

- Python 3.9+
- A free [Google AI Studio](https://aistudio.google.com/) API key

## Installation

```bash
git clone <this-repo-url>
cd gemini_agent
pip install -r requirements.txt
python3 cli.py
```

The first run will ask for your Gemini API key and save it locally next to
the program (`.gemini_api_key`, permissions restricted to your user) — you
won't be asked again on future runs.

## Usage

```bash
python3 cli.py
```

Just type naturally — e.g. "create a Flask app with a health check endpoint
and run it", "find all TODO comments in this project", "commit these changes
with a message about the bugfix". The agent will use its tools to actually
do the work, not just describe it.

### Slash commands

| Command | What it does |
|---|---|
| `/help` | Show available commands |
| `/clear` | Clear the screen |
| `/image` | Attach one or more images to your next message |
| `/force_review` | Force the agent to read and understand every project file before responding |
| `/remember k=v` | Save a persistent fact (e.g. `/remember name=Ahmed`) |
| `/memory` | Show everything saved in long-term memory |
| `/forget <key>` | Delete a fact from memory |
| `/undo` | Revert the last file change |
| `/tree` | Show the workspace as a directory tree |
| `/ps` | List background processes (dev servers, etc.) |
| `/log` | Show recent actions taken this session |
| `/clearlog` | Clear the execution log |
| `/stats` | Model usage report and automatic switches |
| `/workspace` | Show the workspace path |
| `/resetkey` | Delete the saved API key |
| `/exit`, `/quit` | Quit |

Typing a lone `/` (or an unrecognized `/command`) shows a list of
suggestions — no tab-completion dependency required.

## Configuration

Environment variables (all optional):

| Variable | Purpose |
|---|---|
| `AGENT_WORKSPACE` | Path to the sandboxed workspace directory (default: `./workspace`) |
| `AGENT_DATA_DIR` | Where persistent data (API key, memory, logs) is stored (default: next to the script/executable) |
| `AGENT_ENABLE_TAB_COMPLETE=1` | Opt in to readline-based Tab completion (disabled by default — see note below) |
| `NO_COLOR` | Disable ANSI colors in the terminal |

To change the model priority/failover order or per-model request caps, edit
`MODEL_CHAIN` in `config.py`.

> **Note on Tab completion:** it's off by default because on some
> platforms (Termux on Android in particular) importing `readline` at all
> has been observed to corrupt typed keystrokes. Slash commands remain fully
> discoverable without it (typing `/` shows a suggestions box). Set
> `AGENT_ENABLE_TAB_COMPLETE=1` if you're on a platform where readline is
> known to behave well.

## Tools

The agent has 52 tools it can call on its own to get things done:

**File operations** — `create_file`, `read_file`, `edit_file`, `delete_file`,
`move_file`, `rename_file`, `copy_file`, `create_folder`, `diff_preview`,
`compare_files`

**Search & discovery** — `find_file`, `find_folder`, `search_in_files`,
`file_stats`, `detect_language`, `count_files`, `count_todos`, `list_files`,
`count_lines_of_code`

**Bulk editing** — `replace_in_files` (find/replace across many files at
once, with `dry_run`, `whole_word`, and `case_insensitive` options)

**Code quality** — `lint_check`, `check_file_syntax_all`, `find_unused_imports`

**Git** — `git_clone`, `git_status`, `git_diff`, `git_log`, `git_commit`

**Execution** — `run_command`, `start_background_process`,
`list_background_processes`, `read_background_log`, `stop_background_process`

**Dependencies** — `list_dependencies`, `add_dependency`

**Testing** — `run_tests`, `create_test_file`

**Documentation** — `generate_readme`, `extract_docstrings`

**Checkpoints** — `save_checkpoint`, `load_checkpoint`, `list_checkpoints`
(full project snapshots — stronger than `undo_last_change`, which only
reverts one step)

**Networking** — `check_port_in_use`, `http_request`

**Archives** — `create_zip`, `extract_zip`

**Format conversion** — `convert_file_format` (JSON ⇄ YAML ⇄ CSV),
`minify_file` (JSON/CSS/JS)

**Environment** — `env_var_check`

**Images** — `Image_Fetch` (look at an image already in the workspace),
`Image_Create` (generate a new image and save it)

**Safety net** — `undo_last_change`

## Building a standalone executable

A [PyInstaller](https://pyinstaller.org/) spec file is included:

```bash
pip install pyinstaller
pyinstaller gemini_agent.spec
```

This produces a single executable in `dist/` (`gemini_agent` on Linux/macOS,
`gemini_agent.exe` on Windows) that bundles Python and all dependencies — no
separate Python installation needed to run it. Persistent data (API key,
memory) is stored next to the executable itself, not in a temporary
extraction directory, so it survives between runs.

## Architecture

| File | Responsibility |
|---|---|
| `cli.py` | Terminal UI — input loop, slash commands, colored output, live status line, Markdown rendering |
| `agent.py` | Orchestrates a conversation turn: builds the system prompt, runs the manual tool-calling loop, manages history |
| `model_router.py` | Sends requests to the Gemini API with automatic retry/failover across the model chain |
| `tools.py` | All 52 tool implementations, sandboxed to the workspace directory |
| `memory.py` | Persistent long-term memory, session log, and the execution log (recent-actions summary injected into every prompt) |
| `colors.py` | ANSI color helpers and box-drawing for the terminal UI |
| `markdown_render.py` | Lightweight Markdown-to-ANSI renderer (bold, code blocks, lists, headers) |
| `config.py` | All configuration: model chain, paths, timeouts, limits |

Tool-calling is handled **manually** rather than relying on the Gemini SDK's
built-in automatic function calling — this gives full control over logging,
retries, and streaming, and avoids a class of bugs where the SDK and the
app's own code could both try to invoke the same tool.

## Safety notes

- All file tools are confined to the `workspace/` directory; attempts to
  escape it (e.g. `../../etc/passwd`) are rejected.
- `run_command` has a blocklist for obviously destructive commands (`rm -rf
  /`, fork bombs, `shutdown`, etc.) and a timeout on every invocation.
- Long-running commands (dev servers) are detected and run in the background
  automatically instead of blocking.
- `git_clone` only accepts HTTPS URLs (no SSH/git://) and disables
  interactive credential prompts, so it can never hang waiting for input.
- The API key is stored in a local file with owner-only permissions (`600`),
  not in shell environment variables.

This is a best-effort sandboxing layer for a personal coding assistant, not a
hardened multi-tenant security boundary — don't point it at anything you
don't trust the model to have full read/write access to within the workspace.

## License

No license specified — add one appropriate for your use case.
