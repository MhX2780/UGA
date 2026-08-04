"""
Converts the plain-Python tool functions in tools.ALL_TOOLS into OpenAI-style
function-calling JSON schemas — the format Puter.js's OpenAI-compatible REST
endpoint expects (and, incidentally, the same format OpenAI/most other
providers use).

Why this exists: for Gemini, the google-genai SDK builds function
declarations automatically from a plain Python function's type hints and
docstring (that's why tools.ALL_TOOLS is just a list of functions, with no
hand-written schema anywhere). Puter is reached through the `openai` SDK
instead, which has no equivalent auto-inference — it expects an explicit
`tools=[{"type": "function", "function": {"name":..., "parameters": {...}}}]`
list to be built by the caller. Rather than hand-writing and maintaining a
second schema for all 52+ tools, this module derives it from the exact same
functions/docstrings Gemini already uses, via `inspect`, so there is still
only ONE place (tools.py) that defines what a tool does and takes.

BETA scope note: this is used only by the Puter tool-calling path
(config.PUTER_TOOL_CALLING_ENABLED), which is itself off by default — see
that setting's docstring in config.py for why.
"""
import inspect
import re
from typing import Callable, List, Optional, get_type_hints

# Maps Python type hints to JSON Schema types. Anything not found here falls
# back to "string" — a safe default for a docstring-derived schema (the
# model just gets a slightly less strict hint, not a broken one).
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    List[str]: "array",
}

_ARGS_LINE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def _parse_docstring_arg_descriptions(func: Callable) -> dict:
    """
    Extracts {param_name: description} from a Google-style docstring's
    'Args:' section (the style already used throughout tools.py), e.g.:

        Args:
            path: relative file path (e.g. "src/main.py")
            content: file content as text

    Returns {} if the function has no docstring or no Args: section —
    callers should treat missing descriptions as acceptable, not fatal,
    since the schema is still usable (just less descriptive) without them.
    """
    doc = inspect.getdoc(func)
    if not doc:
        return {}
    descriptions = {}
    in_args_section = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue
        if in_args_section:
            # A blank line or a new top-level section (e.g. "Returns:")
            # ends the Args: block.
            if not stripped or stripped.lower().rstrip(":") in ("returns", "raises", "note", "notes"):
                if not stripped:
                    continue
                break
            match = _ARGS_LINE_RE.match(line)
            if match:
                descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


def _json_type_for_annotation(annotation) -> str:
    """Best-effort mapping from a Python type annotation to a JSON Schema
    type string. Defaults to 'string' for anything unrecognized (Optional[],
    custom classes, etc.) rather than raising — a slightly loose schema is
    far better than a converter that crashes on one odd tool signature."""
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    origin = getattr(annotation, "__origin__", None)
    if origin in (list,):
        return "array"
    if origin in (dict,):
        return "object"
    # typing.Optional[X] / typing.Union[X, None] — use X's mapped type.
    args = getattr(annotation, "__args__", None)
    if args:
        for arg in args:
            if arg is not type(None) and arg in _TYPE_MAP:
                return _TYPE_MAP[arg]
    return "string"


def tool_to_openai_schema(func: Callable) -> dict:
    """
    Builds one OpenAI-style tool schema dict for a single tool function,
    e.g.:
        {
          "type": "function",
          "function": {
            "name": "create_file",
            "description": "Creates a new file with the given content...",
            "parameters": {
              "type": "object",
              "properties": {
                "path": {"type": "string", "description": "relative file path..."},
                "content": {"type": "string", "description": "file content as text"}
              },
              "required": ["path", "content"]
            }
          }
        }
    """
    name = func.__name__
    full_doc = inspect.getdoc(func) or ""
    # Use only the text before "Args:"/"Returns:" as the top-level
    # description — keeps it a concise summary rather than dumping the
    # entire docstring (including per-arg detail, which belongs in
    # `parameters` instead) into the `description` field.
    description = full_doc.split("\n\nArgs:")[0].split("\nArgs:")[0].strip()
    description = " ".join(description.split())  # collapse whitespace/newlines

    arg_descriptions = _parse_docstring_arg_descriptions(func)
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        prop = {"type": _json_type_for_annotation(annotation)}
        if param_name in arg_descriptions:
            prop["description"] = arg_descriptions[param_name]
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Calls {name}.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_openai_tools_schema(tool_functions: List[Callable]) -> list:
    """
    Converts a list of tool functions (e.g. tools.ALL_TOOLS) into the full
    OpenAI-style `tools=[...]` list expected by Puter's REST endpoint.
    Skips (rather than crashes on) any single tool whose schema can't be
    built, so one malformed/unusual signature can't take down the entire
    toolbox for every other tool.
    """
    schemas = []
    for func in tool_functions:
        try:
            schemas.append(tool_to_openai_schema(func))
        except Exception:
            # Best-effort: a tool that fails to introspect cleanly is
            # simply left out of the Puter toolbox rather than raised,
            # since this whole path is already BETA/best-effort.
            continue
    return schemas
