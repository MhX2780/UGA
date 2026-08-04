"""
Multi-provider abstraction: lets multi-agent ROLES (classifier, planner,
reviewer) use a model from a DIFFERENT AI provider than Gemini — e.g.
"Gemini for understanding the request, Claude for planning, Gemini for
execution (tool-calling), DeepSeek/GPT for a final sanity check".

Scope note: this module only covers PLAIN TEXT generation (no tool/function
calling). The executor role — which actually calls tools.ALL_TOOLS to touch
files/run commands — stays on Gemini via ModelRouter regardless of provider
settings, because:
  1) Tool-calling schemas/response formats differ significantly across
     providers (Gemini's function_call parts vs. Anthropic's tool_use blocks
     vs. OpenAI's tool_calls), and unifying that is a much larger, riskier
     rewrite than this feature needs to justify right now.
  2) The other three roles (classify simple-vs-complex, write a plan, review
     a finished result) are all pure text-in/text-out tasks — no tools
     needed — so they're a clean, low-risk fit for a provider-agnostic layer.

Providers are OPTIONAL dependencies: this module only imports the
anthropic/openai SDKs lazily, inside each provider's own function, so the
whole app keeps working with only google-genai installed if the user never
configures a non-Gemini provider.

API keys for other providers are stored the same way as the Gemini key
(local file, owner-only permissions) — see config.py's
save_provider_api_key/load_provider_api_key.
"""
from typing import Iterator, Optional

import config


# Maps a provider id to (display name, env-var-style key name, key filename).
PROVIDERS = {
    "gemini": {"label": "Google Gemini", "key_file": ".gemini_api_key"},
    "anthropic": {"label": "Anthropic Claude", "key_file": ".anthropic_api_key"},
    "openai": {"label": "OpenAI ChatGPT", "key_file": ".openai_api_key"},
    "puter": {"label": "Puter.js (free, user-pays)", "key_file": ".puter_token"},
}

# Puter's OpenAI-compatible endpoint — an auth token from
# puter.com/dashboard#account works as the "API key" here. This gives free
# (to this app — the signed-in Puter user pays for their own usage) access
# to 500+ models including Claude, GPT, Gemini, DeepSeek, and more, all
# through one endpoint/token instead of needing separate keys for each.
PUTER_BASE_URL = "https://api.puter.com/puterai/openai/v1/"


def provider_for_model(model_name: str) -> str:
    """
    Guesses which provider a model name belongs to, from common naming
    conventions. Used so a role's assigned model string (e.g.
    "claude-sonnet-4-5" or "gpt-4o") automatically routes to the right
    provider without needing a separate "role -> provider" setting on top
    of "role -> model".

    Note: Puter model names overlap with Anthropic/OpenAI's own naming
    (Puter re-exposes the same underlying models), so this function can't
    distinguish "gpt-4o via OpenAI directly" from "gpt-4o via Puter" by name
    alone — see puter_model() below, which is used explicitly instead of
    this auto-detection whenever the caller specifically wants Puter.
    """
    lowered = model_name.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gpt") or lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
        return "openai"
    return "gemini"  # default/fallback — covers gemini-* and gemma-* names


def has_provider_key(provider: str) -> bool:
    """Whether an API key is configured for the given provider."""
    if provider == "gemini":
        return bool(config.GEMINI_API_KEY or config.load_saved_api_key())
    if provider == "puter":
        return bool(config.load_puter_token())
    key_file = config.BASE_DIR / PROVIDERS[provider]["key_file"]
    return key_file.exists() and key_file.read_text(encoding="utf-8").strip() != ""


def load_provider_api_key(provider: str) -> str:
    if provider == "gemini":
        return config.GEMINI_API_KEY or config.load_saved_api_key()
    if provider == "puter":
        return config.load_puter_token()
    key_file = config.BASE_DIR / PROVIDERS[provider]["key_file"]
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


def save_provider_api_key(provider: str, key: str):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    if provider == "puter":
        config.save_puter_token(key)
        return
    key_file = config.BASE_DIR / PROVIDERS[provider]["key_file"]
    key_file.write_text(key.strip(), encoding="utf-8")
    try:
        import os
        os.chmod(key_file, 0o600)
    except OSError:
        pass


# ---------------- Anthropic (Claude) ----------------

def _anthropic_generate(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    import anthropic
    api_key = load_provider_api_key("anthropic")
    if not api_key:
        raise RuntimeError("No Anthropic API key configured. Set one via /settings provider anthropic <key>.")
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {"model": model_name, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
    if system_instruction:
        kwargs["system"] = system_instruction
    response = client.messages.create(**kwargs)
    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(text_parts)


def _anthropic_generate_stream(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> Iterator[str]:
    import anthropic
    api_key = load_provider_api_key("anthropic")
    if not api_key:
        raise RuntimeError("No Anthropic API key configured. Set one via /settings provider anthropic <key>.")
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {"model": model_name, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
    if system_instruction:
        kwargs["system"] = system_instruction
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


# ---------------- OpenAI (ChatGPT) ----------------

def _openai_generate(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    import openai
    api_key = load_provider_api_key("openai")
    if not api_key:
        raise RuntimeError("No OpenAI API key configured. Set one via /settings provider openai <key>.")
    client = openai.OpenAI(api_key=api_key)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model_name, messages=messages)
    return response.choices[0].message.content or ""


def _openai_generate_stream(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> Iterator[str]:
    import openai
    api_key = load_provider_api_key("openai")
    if not api_key:
        raise RuntimeError("No OpenAI API key configured. Set one via /settings provider openai <key>.")
    client = openai.OpenAI(api_key=api_key)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    stream = client.chat.completions.create(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


# ---------------- Puter.js (free, user-pays; 500+ models via one token) ----------------

def _enforce_puter_free_only(model_name: str):
    """
    If the "Use Free Models Only (Puter.js)" setting is on, blocks any Puter
    call to a model that doesn't look free (id doesn't end with/contain
    "free"). Raises instead of silently swapping models, so the caller finds
    out immediately rather than getting a surprise model in the response.
    """
    from config import PUTER_FREE_ONLY
    if PUTER_FREE_ONLY and "free" not in model_name.lower():
        raise RuntimeError(
            f"'{model_name}' is blocked by the \"Use Free Models Only (Puter.js)\" "
            f"setting (model id doesn't contain \"free\"). Pick a free model via "
            f"/free, or turn the setting off with /settings puter free off."
        )


def _puter_reasoning_kwargs() -> dict:
    """
    Returns the extra kwargs to attach to a Puter chat.completions call when
    "Deep Thinking" is enabled for Puter (config.PUTER_DEEP_THINKING_ENABLED)
    — i.e. {"reasoning_effort": "low"|"medium"|"high"}, the OpenAI-style
    field that reasoning-capable models re-exposed through Puter (o1/o3,
    deepseek-reasoner, Claude extended-thinking models, etc.) use to control
    how much hidden reasoning they do before answering. Returns {} when the
    setting is off, so this can be splatted into kwargs unconditionally by
    every Puter call site below without an extra if-branch at each one.
    Sending reasoning_effort to a model that doesn't support it is
    harmless — Puter/the underlying provider ignores fields it doesn't
    recognize rather than erroring, same as OpenAI's own API does for
    non-reasoning models.
    """
    if not config.PUTER_DEEP_THINKING_ENABLED:
        return {}
    return {"reasoning_effort": config.PUTER_DEEP_THINKING_EFFORT}


def _puter_generate(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    import openai
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model_name, messages=messages, **_puter_reasoning_kwargs()
    )
    return response.choices[0].message.content or ""


# ---------------- Puter.js tool calling (BETA) ----------------
# See config.PUTER_TOOL_CALLING_ENABLED's docstring for the full rationale
# and the two known sources of risk (untested endpoint parity, inconsistent
# model support). These functions are only ever called when that setting is
# explicitly turned on — never by the plain-text _puter_generate() above.

def puter_chat_with_tools(model_name: str, messages: list, tools: Optional[list] = None):
    """
    BETA. Sends a raw OpenAI-style `messages` list (which may already
    include prior assistant tool_calls and "tool"-role results, unlike
    _puter_generate's simple single-prompt shape) to Puter, optionally with
    a `tools` schema (see tool_schemas.build_openai_tools_schema). Returns
    the raw response object as-is — the caller (agent.py's Puter tool-call
    loop) is responsible for inspecting response.choices[0].message for
    both `.content` and `.tool_calls`, since a tool-calling turn may return
    either, and the loop logic (execute tool -> append result -> ask again)
    lives there, not in this thin transport function.

    Not currently used for the plain-text-only path (PUTER_CHAT_ENABLED
    without PUTER_TOOL_CALLING_ENABLED) — that continues to use the simpler
    _puter_generate() above unchanged.
    """
    import openai
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    kwargs = {"model": model_name, "messages": messages, **_puter_reasoning_kwargs()}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def puter_chat_with_tools_stream(model_name: str, messages: list, tools: Optional[list] = None):
    """
    BETA. Streaming counterpart to puter_chat_with_tools(). Yields a
    sequence of events as dicts so the caller (agent.py) doesn't need to
    know OpenAI's raw chunk-delta shape:

        {"type": "text", "text": "..."}                       - a text chunk to display
        {"type": "tool_calls", "tool_calls": [...]}            - the complete, reassembled
                                                                  tool call list, yielded
                                                                  once the stream ends if
                                                                  any tool calls were made

    Why reassembly is needed: in OpenAI-style streaming, a single tool call
    arrives split across many chunks — e.g. the function *name* might be in
    one chunk and its JSON *arguments* dribble in character-by-character
    across several more, all correlated only by a per-call `index`. Unlike
    plain text (where each chunk is independently printable), a tool call
    isn't usable until every fragment for that index has been concatenated
    and the arguments string is complete valid JSON — so tool_calls are
    intentionally NOT yielded incrementally, only once, fully assembled,
    after the stream finishes.
    """
    import openai
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    kwargs = {"model": model_name, "messages": messages, "stream": True, **_puter_reasoning_kwargs()}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    stream = client.chat.completions.create(**kwargs)

    # Accumulates partial tool-call fragments keyed by their `index` (their
    # only stable identifier while streaming — `id` may only appear on the
    # first fragment for that index).
    tool_call_fragments = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if getattr(delta, "content", None):
            yield {"type": "text", "text": delta.content}

        for tc_delta in (getattr(delta, "tool_calls", None) or []):
            idx = tc_delta.index
            entry = tool_call_fragments.setdefault(
                idx, {"id": None, "name": "", "arguments": ""}
            )
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    entry["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    entry["arguments"] += tc_delta.function.arguments

    if tool_call_fragments:
        assembled = [
            {
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            }
            for _, entry in sorted(tool_call_fragments.items())
        ]
        yield {"type": "tool_calls", "tool_calls": assembled}


def _puter_generate_stream(model_name: str, prompt: str, system_instruction: Optional[str] = None) -> Iterator[str]:
    import openai
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    stream = client.chat.completions.create(
        model=model_name, messages=messages, stream=True, **_puter_reasoning_kwargs()
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


PUTER_MODELS_ENDPOINT = "https://api.puter.com/puterai/chat/models/details"


def puter_list_models() -> list:
    """
    Fetches the list of models currently available through Puter for the
    configured token. Used by /settings provider puter models and /puterJS
    in the CLI so the user can see actual current model IDs (e.g. "gpt-4o",
    "claude-sonnet-4-5", "deepseek-chat") rather than guessing names.

    Important: Puter does NOT implement an OpenAI-compatible /v1/models
    endpoint — calling client.models.list() against PUTER_BASE_URL (as an
    earlier version of this function did) hits a route that simply doesn't
    exist on Puter's backend and returns 404 Not Found. Chat completions
    (chat.completions.create, used elsewhere in this module) ARE OpenAI-
    compatible and work fine; listing models is not, and needs Puter's own
    endpoint instead. So this function talks to Puter directly over
    `requests` rather than going through the `openai` SDK/base_url.
    """
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError("No Puter auth token configured.")
    import requests
    response = requests.get(
        PUTER_MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    # Puter's response shape has varied across versions/endpoints seen in
    # the wild (a bare list, {"models": [...]}, or {"data": [...]}), and
    # each model entry may be a plain string id or a dict with an "id"/
    # "name" field. Normalize defensively rather than assuming one shape.
    if isinstance(data, dict):
        entries = data.get("models") or data.get("data") or []
    elif isinstance(data, list):
        entries = data
    else:
        entries = []

    model_ids = []
    for entry in entries:
        if isinstance(entry, str):
            model_ids.append(entry)
        elif isinstance(entry, dict):
            model_id = entry.get("id") or entry.get("name")
            if model_id:
                model_ids.append(model_id)
    return model_ids


def puter_list_free_models() -> list:
    """
    Same as puter_list_models(), but filtered to only model IDs ending in
    "free" (case-insensitive), e.g. names like "...:free" or "...-free" that
    some providers exposed through Puter use to mark a no-cost/limited tier.
    Used by the /free command so the user can quickly see and pick from only
    the free-tier models without scrolling through the full 500+ list.
    """
    all_models = puter_list_models()
    return [m for m in all_models if m.lower().rstrip().endswith("free")]


# ---------------- unified entrypoints ----------------

def generate_text(model_name: str, prompt: str, system_instruction: Optional[str] = None,
                   provider: Optional[str] = None) -> str:
    """
    Sends a plain text-in/text-out request to whichever provider `model_name`
    belongs to (auto-detected via provider_for_model, or pass `provider`
    explicitly — needed for Puter, since Puter re-exposes models under the
    same names as their original providers, e.g. "gpt-4o", so name-based
    auto-detection alone can't tell "OpenAI's gpt-4o" and "Puter's gpt-4o"
    apart). For Gemini models, this is a thin wrapper that still goes through
    the caller's existing ModelRouter for full retry/failover behavior —
    callers needing that should prefer router.generate_with_model() directly
    for Gemini models and only fall back to this function for non-Gemini
    providers.
    """
    resolved_provider = provider or provider_for_model(model_name)
    if resolved_provider == "puter":
        return _puter_generate(model_name, prompt, system_instruction)
    if resolved_provider == "anthropic":
        return _anthropic_generate(model_name, prompt, system_instruction)
    if resolved_provider == "openai":
        return _openai_generate(model_name, prompt, system_instruction)
    raise ValueError(
        f"generate_text() called with a Gemini model ({model_name}) — use "
        f"ModelRouter.generate_with_model() instead for Gemini, which has "
        f"proper retry/failover handling this function doesn't."
    )


def generate_text_stream(model_name: str, prompt: str, system_instruction: Optional[str] = None,
                          provider: Optional[str] = None) -> Iterator[str]:
    """Streaming counterpart to generate_text() — see its docstring."""
    resolved_provider = provider or provider_for_model(model_name)
    if resolved_provider == "puter":
        yield from _puter_generate_stream(model_name, prompt, system_instruction)
    elif resolved_provider == "anthropic":
        yield from _anthropic_generate_stream(model_name, prompt, system_instruction)
    elif resolved_provider == "openai":
        yield from _openai_generate_stream(model_name, prompt, system_instruction)
    else:
        raise ValueError(
            f"generate_text_stream() called with a Gemini model ({model_name}) — "
            f"use ModelRouter.generate_stream_with_model() instead."
        )


# ---------------- Puter.js image tools (BETA fallback for Gemini) ----------------
# See config.PUTER_IMAGE_TOOLS_ENABLED's docstring for the full rationale
# and which of these two is on more solid ground (vision) vs. genuinely
# unverified (generation). Both are only ever invoked by tools.py's
# Image_Fetch_Puter / Image_Create_Puter, themselves only reachable after
# the model has asked the user for permission — see those functions'
# docstrings in tools.py for the full flow.

def puter_vision_describe(model_name: str, image_bytes: bytes, mime_type: str, question: str) -> str:
    """
    BETA. Sends an image + question to a Puter-hosted vision-capable model
    via chat.completions with an image content part — the same
    documented, REST-compatible endpoint already used for text and tool
    calling elsewhere in this module (unlike puter_image_generate below,
    this one is NOT the uncertain half of the image BETA).

    Uses the OpenAI chat-completions "image_url" content-part convention
    (a data: URI is passed as the "url" field) since that's the format the
    `openai` SDK / OpenAI-compatible endpoints expect for image input.
    """
    import openai
    import base64
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    b64_data = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64_data}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
    )
    return response.choices[0].message.content or ""


def puter_image_generate(model_name: str, prompt: str) -> bytes:
    """
    BETA — genuinely unverified. Attempts image generation through Puter
    via the OpenAI SDK's standard images.generate() call against Puter's
    base_url. Puter's OWN documentation only demonstrates image generation
    (puter.ai.txt2img()) via the browser JavaScript SDK, returning an
    HTMLImageElement — no official example shows a server-side/REST route
    for it analogous to OpenAI's /v1/images/generations. This function may
    simply fail with a 404 or similar; that's expected and handled by the
    caller (tools.Image_Create_Puter), not treated as a bug in this
    function. Kept as a best-effort attempt (Puter advertises broad
    OpenAI-compatibility) rather than omitted entirely, so the capability
    is at least tried and clearly reported on rather than silently assumed
    impossible.

    Returns the raw image bytes (decoded from the response's b64_json)
    on success. Raises on any failure — the caller is responsible for
    turning that into a clear ❌ message for the model/user rather than
    letting a raw SDK exception surface.
    """
    import openai
    import base64
    _enforce_puter_free_only(model_name)
    token = load_provider_api_key("puter")
    if not token:
        raise RuntimeError(
            "No Puter auth token configured. Get one free at "
            "puter.com/dashboard#account ('Create token') and set it via "
            "/settings provider puter <token>, or use /puterJS to sign in "
            "through the browser instead."
        )
    client = openai.OpenAI(api_key=token, base_url=PUTER_BASE_URL)
    response = client.images.generate(model=model_name, prompt=prompt, response_format="b64_json")
    b64_data = response.data[0].b64_json
    if not b64_data:
        raise RuntimeError("Puter's image endpoint returned no image data.")
    return base64.b64decode(b64_data)
