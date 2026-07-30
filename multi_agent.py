"""
Multi-agent orchestration: automatically decides whether a user's request is
simple (handled by a single model, same as the classic single-agent flow) or
complex enough to benefit from a structured team of role-specialized models:

  1) classifier — fast/cheap model decides simple-vs-complex
  2) planner    — breaks a complex task into a numbered list of concrete steps
  3) executor   — carries out the plan step by step, actually calling tools
  4) reviewer   — checks the final result and writes the summary shown to the user

Each role's model is configurable (see config.MULTI_AGENT_ROLES / /settings
in the CLI) — by default all four roles use different Gemini models chosen
for their strengths (a bigger model for planning/reasoning, a fast one for
execution, a cheap one for the quick classification and review passes), but
nothing here is tied to Gemini specifically at the orchestration level: each
role only needs a model name string and goes through ModelRouter.

The plan is surfaced to the user as it's carried out, e.g.:

    Plan 1 of 3

    Plan 1:
     Created main.py
     Deleted old.py
    Plan 2:
     Running command...
     git clone ...
    Plan 3:
     Code check
     Zip workspace

This module deliberately does NOT talk to the terminal directly (no print()
calls) — it yields structured events that cli.py renders, so the UI and
orchestration logic stay decoupled the same way the rest of this app does.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Union

from google.genai import types

import config
import tools


CLASSIFIER_PROMPT = """You are a request classifier for a coding agent. Decide whether the
user's request is SIMPLE or COMPLEX.

SIMPLE: a direct question, a small one-step file edit, a quick lookup, casual
conversation, or anything answerable/doable in a single response without
needing a multi-step plan.

COMPLEX: anything requiring multiple distinct steps that depend on each
other's results — e.g. "set up a project, install dependencies, write code,
and test it", "clone this repo, check the code, and push a fix", "build a
Flask app with three endpoints and run it".

Respond with ONLY one word: SIMPLE or COMPLEX. Nothing else.

User's request: {request}"""

PLANNER_PROMPT = """You are the planning stage of a multi-agent coding assistant. Break the
user's request into a short numbered list of concrete, sequential steps. Each
step should be a single coherent unit of work (e.g. "Create the Flask app
file with three endpoints", "Install Flask via pip", "Run the app and verify
it starts").

Rules:
- 2 to 6 steps. Do not over-split simple sub-parts into separate steps.
- Each step must be something the executor can actually carry out using file/
  command tools — not vague advice.
- Do not write any code yourself here — just the plan.
- Respond with ONLY a JSON array of short step descriptions, e.g.:
  ["Create app.py with the Flask routes", "Install Flask", "Run app.py and confirm it starts"]

{memory_context}

User's request: {request}"""

EXECUTOR_PROMPT_TEMPLATE = """You are the execution stage of a multi-agent coding assistant, currently
working on ONE step of a larger plan. Use your tools to actually carry out
this step. Be concise in your text reply — the tool calls are what matter.

Full plan (for context — you are only responsible for the CURRENT step):
{full_plan}

Current step ({step_number} of {total_steps}): {current_step}

{memory_context}

{execution_log_context}
"""

REVIEWER_PROMPT_TEMPLATE = """You are the review stage of a multi-agent coding assistant. The plan below
was just carried out step by step. Review what was actually done (see the
execution log) and write a concise final summary for the user: what was
accomplished, and flag anything that looks incomplete, failed, or worth
double-checking. Do not call any tools — just summarize.

Plan that was carried out:
{full_plan}

{execution_log_context}

User's original request: {request}"""


@dataclass
class PlanStep:
    number: int
    description: str
    status: str = "pending"  # pending | running | done | failed
    actions: List[str] = field(default_factory=list)  # short summaries of tool calls made during this step


@dataclass
class MultiAgentEvent:
    """
    A structured event yielded by run_multi_agent_turn() as the team works,
    for cli.py to render. `kind` determines what the other fields mean:
      - "classified": {"complexity": "simple"|"complex"}
      - "plan_ready": {"steps": [str, ...]}
      - "step_start": {"step_number": int, "total_steps": int, "description": str}
      - "step_action": {"step_number": int, "action_summary": str}
      - "step_done": {"step_number": int}
      - "text_chunk": {"text": str}  (final streamed reply text, simple-path or reviewer summary)
    """
    kind: str
    data: dict


def _extract_json_array(text: str) -> Optional[list]:
    """
    Best-effort extraction of a JSON array from a model's text response,
    tolerating common wrapping like markdown code fences around the JSON.
    Returns None if no valid array could be parsed.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    # Fallback: try to find the first [...] substring in case there's stray
    # prose around the array despite the prompt asking for JSON only.
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return None


class MultiAgentOrchestrator:
    """
    Runs the classifier -> (planner -> executor(s) -> reviewer) flow for a
    single user turn. Holds a reference to the parent GeminiAgent so it can
    reuse its router, tool-calling loop, memory, and execution log rather
    than duplicating that machinery.
    """

    def __init__(self, parent_agent):
        self.agent = parent_agent  # the owning GeminiAgent instance

    def _role_model(self, role: str) -> str:
        """Looks up which model is currently assigned to a role, via the
        live config (so /settings changes take effect without restart)."""
        return config.MULTI_AGENT_ROLES.get(role) or config.MODEL_CHAIN[0]["name"]

    def _classify(self, user_message: str) -> str:
        """Returns 'simple' or 'complex'."""
        prompt = CLASSIFIER_PROMPT.format(request=user_message)
        response = self.agent.router.generate_with_model(
            model_name=self._role_model("classifier"),
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )
        answer = (response.text or "").strip().upper()
        return "complex" if "COMPLEX" in answer else "simple"

    def _plan(self, user_message: str) -> List[str]:
        """Returns a list of step description strings."""
        memory_context = self.agent.memory.memory_as_context_string()
        prompt = PLANNER_PROMPT.format(
            request=user_message,
            memory_context=memory_context or "(No saved information about the user yet)",
        )
        response = self.agent.router.generate_with_model(
            model_name=self._role_model("planner"),
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )
        steps = _extract_json_array(response.text or "")
        if not steps:
            # Planner didn't return usable JSON — fall back to a single
            # step covering the whole request rather than crashing the turn.
            steps = [user_message]
        return steps

    def _execute_step(self, step: PlanStep, total_steps: int, full_plan_str: str, user_message: str) -> Iterator[MultiAgentEvent]:
        """
        Runs one plan step to completion (including any tool calls it
        needs), yielding step_action events as tools are used. Reuses the
        agent's own tool-calling loop (_run_tool_calls) so undo/execution-log/
        activity-notification behavior stays identical to single-agent mode.
        """
        memory_context = self.agent.memory.memory_as_context_string()
        execution_log_context = self.agent.execution_log.as_context_string()
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            full_plan=full_plan_str,
            step_number=step.number,
            total_steps=total_steps,
            current_step=step.description,
            memory_context=memory_context or "(No saved information about the user yet)",
            execution_log_context=execution_log_context or "(No actions taken yet this session)",
        )

        self.agent.history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        response = self.agent.router.generate_with_model(
            model_name=self._role_model("executor"),
            contents=self.agent.history,
            system_instruction=None,
            tools=tools.ALL_TOOLS,
        )

        # Track how many execution_log entries existed before this step so
        # we can report only the NEW ones as step_action events.
        before_count = len(self.agent.execution_log.recent(limit=10_000))
        response = self.agent._run_tool_calls_with_model(response, self._role_model("executor"))
        after_entries = self.agent.execution_log.recent(limit=10_000)
        new_entries = after_entries[before_count:]

        for entry in new_entries:
            icon = "✓" if entry["success"] else "✗"
            args_str = ", ".join(f"{k}={v}" for k, v in entry.get("args", {}).items())
            yield MultiAgentEvent("step_action", {
                "step_number": step.number,
                "action_summary": f"{icon} {entry['tool']}({args_str})",
            })

        reply_text = response.text or ""
        if reply_text.strip():
            self.agent.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))

    def _review(self, full_plan_str: str, user_message: str) -> Iterator[str]:
        """Yields the final reviewer summary as streamed text chunks."""
        execution_log_context = self.agent.execution_log.as_context_string()
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            full_plan=full_plan_str,
            execution_log_context=execution_log_context or "(No actions were recorded)",
            request=user_message,
        )
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        for chunk in self.agent.router.generate_stream_with_model(
            model_name=self._role_model("reviewer"),
            contents=contents,
        ):
            yield chunk

    def run_turn(self, user_message: str) -> Iterator[MultiAgentEvent]:
        """
        Main entrypoint: classifies the request, then either hands off to
        the normal single-agent flow (simple case) or runs the full
        plan -> execute -> review pipeline (complex case), yielding
        MultiAgentEvent objects throughout for cli.py to render live.
        """
        complexity = self._classify(user_message)
        yield MultiAgentEvent("classified", {"complexity": complexity})

        if complexity == "simple":
            # Simple path: identical to normal single-agent send_stream, just
            # routed through here so the CLI's call site doesn't need two
            # separate code paths for "multi-agent mode but simple request".
            for chunk in self.agent.send_stream(user_message, _skip_multi_agent=True):
                yield MultiAgentEvent("text_chunk", {"text": chunk})
            return

        # Complex path: plan -> execute each step -> review
        step_descriptions = self._plan(user_message)
        steps = [PlanStep(number=i + 1, description=desc) for i, desc in enumerate(step_descriptions)]
        full_plan_str = "\n".join(f"{s.number}. {s.description}" for s in steps)

        yield MultiAgentEvent("plan_ready", {"steps": step_descriptions})

        self.agent.memory.log_message("user", user_message)
        self.agent.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        for step in steps:
            yield MultiAgentEvent("step_start", {
                "step_number": step.number,
                "total_steps": len(steps),
                "description": step.description,
            })
            step.status = "running"
            try:
                for event in self._execute_step(step, full_plan_str, user_message):
                    yield event
                step.status = "done"
            except Exception as e:
                step.status = "failed"
                yield MultiAgentEvent("step_action", {
                    "step_number": step.number,
                    "action_summary": f"✗ Step failed: {e}",
                })
            yield MultiAgentEvent("step_done", {"step_number": step.number})

        full_text_parts = []
        for chunk in self._review(full_plan_str, user_message):
            full_text_parts.append(chunk)
            yield MultiAgentEvent("text_chunk", {"text": chunk})

        reply_text = "".join(full_text_parts) or "(No summary produced)"
        self.agent.memory.log_message("model", reply_text, meta={"model": self._role_model("reviewer")})
        self.agent.history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))
        self.agent._trim_history()
