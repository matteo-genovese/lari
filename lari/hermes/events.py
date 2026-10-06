"""Semantic Hermes events and pure, incremental parsing of SSE text lines."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace


class ApprovalNotAvailable(Exception):
    """The voice channel cannot answer a command approval request."""


@dataclass(frozen=True)
class HermesTextDelta:
    text: str


@dataclass(frozen=True)
class HermesStatus:
    name: str
    payload: object


@dataclass(frozen=True)
class HermesApprovalRequest:
    payload: object


@dataclass(frozen=True)
class HermesTurnCompleted:
    text: str
    run_id: str | None = None


class HermesReply(str):
    """String-compatible Hermes reply carrying the response session id."""

    def __new__(
        cls,
        text: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        reply = super().__new__(cls, text)
        reply.session_id = session_id
        reply.run_id = run_id
        return reply


HermesEvent = HermesTextDelta | HermesStatus | HermesApprovalRequest | HermesTurnCompleted


@dataclass(frozen=True)
class HermesSSEState:
    """Explicit parser state; inputs are never mutated or shared between turns."""
    event_name: str | None = None
    data_lines: tuple[str, ...] = ()
    content: tuple[str, ...] = ()
    run_id: str | None = None
    done: bool = False
    saw_stop_finish: bool = False


def _find_run_id(value: object) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("run_id")
        if isinstance(candidate, str) and candidate:
            return candidate
        for nested in value.values():
            found = _find_run_id(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_run_id(nested)
            if found:
                return found
    return None


def _parse_event(state: HermesSSEState) -> tuple[HermesSSEState, list[HermesEvent]]:
    event_name = state.event_name
    data = "\n".join(state.data_lines)
    state = replace(state, event_name=None, data_lines=())
    if not data:
        return state, []
    if data == "[DONE]":
        return replace(state, done=True), []

    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("stream Hermes non valido: JSON SSE corrotto") from exc

    event_run_id = _find_run_id(event)
    if event_run_id:
        state = replace(state, run_id=event_run_id)

    if event_name == "approval.request":
        return state, [HermesApprovalRequest(event)]
    if event_name in {"hermes.tool.progress", "hermes.status"}:
        return state, [HermesStatus(event_name, event)]

    choices = event.get("choices") if isinstance(event, dict) else None
    if not isinstance(choices, list) or not choices:
        return state, []
    choice = choices[0]
    if not isinstance(choice, dict):
        return state, []
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        if not isinstance(finish_reason, str) or finish_reason.lower() != "stop":
            if finish_reason == "error":
                raise RuntimeError("Hermes stream terminato con errore")
            raise RuntimeError(
                f"Hermes stream terminato con finish_reason={finish_reason!r}"
            )
        state = replace(state, saw_stop_finish=True)

    delta = choice.get("delta")
    delta_content = delta.get("content") if isinstance(delta, dict) else None
    if isinstance(delta_content, str):
        state = replace(state, content=state.content + (delta_content,))
        return state, [HermesTextDelta(delta_content)]
    return state, []


def parse_hermes_lines(
    lines: Iterable[str],
    state: HermesSSEState | None = None,
    *,
    final: bool = True,
) -> tuple[HermesSSEState, list[HermesEvent]]:
    """Parse decoded lines without I/O; use ``final=False`` for partial chunks.

    Pass the returned immutable state into the next call. At EOF, ``final=True``
    parses any pending frame and validates both terminal markers before emitting
    completion. Session ids belong to HTTP headers and are attached by the client.
    """
    if state is None:
        state = HermesSSEState()
    events: list[HermesEvent] = []
    for line in lines:
        if line == "":
            state, parsed = _parse_event(state)
            events.extend(parsed)
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            state = replace(state, event_name=value)
        elif field == "data":
            state = replace(state, data_lines=state.data_lines + (value,))

    if final:
        if state.data_lines:
            state, parsed = _parse_event(state)
            events.extend(parsed)
        if not state.done:
            raise RuntimeError("stream Hermes incompleto: manca [DONE]")
        if not state.saw_stop_finish:
            raise RuntimeError("stream Hermes incompleto: manca finish_reason='stop'")
        events.append(HermesTurnCompleted("".join(state.content).strip(), state.run_id))
    return state, events
