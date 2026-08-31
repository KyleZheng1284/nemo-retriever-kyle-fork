# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized, opt-in progress events for agentic retrieval.

This module deliberately observes only fixed lifecycle metadata. It never
reads agent messages, retrieval records, ATIF trajectories, or document data.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal, Mapping, TypeAlias, TypedDict

logger = logging.getLogger(__name__)

JSONScalar: TypeAlias = str | int | float | bool | None


class AgenticProgressEvent(TypedDict):
    """One privacy-safe lifecycle event emitted by agentic retrieval."""

    schema_version: Literal["1"]
    run_id: str
    sequence: int
    query_id: str
    query_index: int
    operation: Literal["query", "retrieval", "llm", "selection"]
    phase: Literal["start", "end"]
    message: str
    attributes: dict[str, JSONScalar]


AgenticProgressSink: TypeAlias = Callable[[AgenticProgressEvent], None]
Operation = Literal["query", "retrieval", "llm", "selection"]
Outcome = Literal["success", "error", "limit"]


@dataclass
class _QueryState:
    caller_query_id: str
    query_index: int
    retrieval_rounds: int = 0
    closed: bool = False
    llm_steps: dict[str, int] = field(default_factory=dict)


class _AgenticProgressSession:
    """Serialize one run's privacy-safe lifecycle callback."""

    def __init__(self, sink: AgenticProgressSink, query_ids: list[str]) -> None:
        self._sink: AgenticProgressSink | None = sink
        self._run_id = uuid.uuid4().hex
        self._sequence = 0
        self._lock = threading.Lock()
        self._queries = {
            str(position): _QueryState(caller_query_id=query_id, query_index=position)
            for position, query_id in enumerate(query_ids)
        }
        for graph_query_id in self._queries:
            self._emit(
                graph_query_id,
                operation="query",
                phase="start",
                message="Agentic retrieval started.",
                attributes={},
            )

    @property
    def enabled(self) -> bool:
        """Return whether the sink has remained healthy for this run."""
        return self._sink is not None

    def start_retrieval(
        self,
        graph_query_id: str,
        *,
        kind: Literal["initial", "follow_up"],
        requested: int,
    ) -> int:
        """Emit a retrieval start using the target before private over-fetch."""
        state = self._queries.get(str(graph_query_id))
        if state is None or not self.enabled:
            return 0
        with self._lock:
            state.retrieval_rounds += 1
            round_number = state.retrieval_rounds
        message = (
            "Running the initial retrieval."
            if kind == "initial"
            else f"Running follow-up retrieval round {round_number - 1}."
        )
        self._emit(
            graph_query_id,
            operation="retrieval",
            phase="start",
            message=message,
            attributes={"round": round_number, "kind": kind, "requested": int(requested)},
        )
        return round_number

    def finish_retrieval(
        self,
        graph_query_id: str,
        *,
        round_number: int,
        kind: Literal["initial", "follow_up"],
        requested: int,
        returned: int,
        outcome: Literal["success", "error"],
    ) -> None:
        """Emit the safe scalar result of one retrieval attempt."""
        if outcome == "success":
            result_word = "result" if returned == 1 else "results"
            message = f"Retrieval returned {returned} {result_word}."
        else:
            message = "Retrieval failed."
        self._emit(
            graph_query_id,
            operation="retrieval",
            phase="end",
            message=message,
            attributes={
                "round": int(round_number),
                "kind": kind,
                "requested": int(requested),
                "returned": int(returned),
                "outcome": outcome,
            },
        )

    def start_llm(self, graph_query_id: str, stage: Literal["react", "selection"]) -> int:
        """Allocate and emit a one-based LLM attempt for a query stage."""
        state = self._queries.get(str(graph_query_id))
        if state is None or not self.enabled:
            return 0
        with self._lock:
            step = state.llm_steps.get(stage, 0) + 1
            state.llm_steps[stage] = step
        self._emit(
            graph_query_id,
            operation="llm",
            phase="start",
            message=f"Running {_stage_label(stage)} LLM step {step}.",
            attributes={"stage": stage, "step": step},
        )
        return step

    def finish_llm(
        self,
        graph_query_id: str,
        *,
        stage: Literal["react", "selection"],
        step: int,
        outcome: Outcome,
    ) -> None:
        """Emit the outcome of one LLM attempt without provider telemetry."""
        label = _stage_label(stage)
        if outcome == "success":
            message = f"Completed {label} LLM step {step}."
        elif outcome == "limit":
            message = f"{label} LLM step {step} reached its limit."
        else:
            message = f"{label} LLM step {step} failed."
        self._emit(
            graph_query_id,
            operation="llm",
            phase="end",
            message=message,
            attributes={"stage": stage, "step": int(step), "outcome": outcome},
        )

    def start_selection(self, graph_query_id: str) -> None:
        """Emit the selection-agent gate only when it is actually entered."""
        self._emit(
            graph_query_id,
            operation="selection",
            phase="start",
            message="Running the selection pass.",
            attributes={},
        )

    def finish_selection(self, graph_query_id: str, *, selected: int, outcome: Outcome) -> None:
        """Emit the safe scalar result of the selection pass."""
        if outcome == "success":
            result_word = "result" if selected == 1 else "results"
            message = f"Selection pass supplied {selected} {result_word}."
        elif outcome == "limit":
            message = "Selection pass reached its limit."
        else:
            message = "Selection pass failed."
        self._emit(
            graph_query_id,
            operation="selection",
            phase="end",
            message=message,
            attributes={"selected": int(selected), "outcome": outcome},
        )

    def finish_query(
        self,
        graph_query_id: str,
        *,
        result_source: str,
        selected: int,
        outcome: Outcome,
    ) -> None:
        """Close one query with a deterministic, content-free summary."""
        state = self._queries.get(str(graph_query_id))
        if state is None or not self.enabled:
            return
        with self._lock:
            if state.closed:
                return
            state.closed = True
        result_source = _safe_result_source(result_source)
        message = (
            _query_summary(state.retrieval_rounds, result_source, int(selected))
            if outcome == "success"
            else "Agentic retrieval failed."
        )
        self._emit(
            graph_query_id,
            operation="query",
            phase="end",
            message=message,
            attributes={
                "retrieval_rounds": state.retrieval_rounds,
                "result_source": result_source,
                "selected": int(selected),
                "outcome": outcome,
            },
        )

    def _emit(
        self,
        graph_query_id: str,
        *,
        operation: Operation,
        phase: Literal["start", "end"],
        message: str,
        attributes: Mapping[str, JSONScalar],
    ) -> None:
        state = self._queries.get(str(graph_query_id))
        if state is None:
            return
        with self._lock:
            sink = self._sink
            if sink is None:
                return
            self._sequence += 1
            event: AgenticProgressEvent = {
                "schema_version": "1",
                "run_id": self._run_id,
                "sequence": self._sequence,
                "query_id": state.caller_query_id,
                "query_index": state.query_index,
                "operation": operation,
                "phase": phase,
                "message": message,
                "attributes": dict(attributes),
            }
            try:
                sink(event)
            except Exception:
                self._sink = None
                logger.warning("Agentic progress callback failed; progress is disabled for this run.")


_ACTIVE_SESSION: ContextVar[_AgenticProgressSession | None] = ContextVar("agentic_progress_session", default=None)


def get_progress_session() -> _AgenticProgressSession | None:
    """Return the ambient progress session while its callback remains healthy."""
    session = _ACTIVE_SESSION.get()
    return session if session is not None and session.enabled else None


@contextmanager
def bind_progress_session(session: _AgenticProgressSession) -> Iterator[None]:
    """Bind one progress session around canonical graph execution."""
    token = _ACTIVE_SESSION.set(session)
    try:
        yield
    finally:
        _ACTIVE_SESSION.reset(token)


def _stage_label(stage: Literal["react", "selection"]) -> str:
    return "ReAct" if stage == "react" else "Selection"


def _safe_result_source(value: str) -> str:
    return value if value in {"final_results", "selection_agent", "rrf"} else ""


def _query_summary(retrieval_rounds: int, result_source: str, selected: int) -> str:
    if retrieval_rounds == 0:
        retrieval_sentence = "The agent completed no retrieval rounds."
    elif retrieval_rounds == 1:
        retrieval_sentence = "The agent completed the initial retrieval."
    else:
        follow_ups = retrieval_rounds - 1
        suffix = "round" if follow_ups == 1 else "rounds"
        retrieval_sentence = f"The agent completed the initial retrieval and {follow_ups} follow-up {suffix}."

    result_word = "result" if selected == 1 else "results"
    if result_source == "final_results":
        result_sentence = f"The ReAct agent supplied {selected} {result_word}."
    elif result_source == "selection_agent":
        result_sentence = f"The selection pass supplied {selected} {result_word}."
    elif result_source == "rrf":
        result_sentence = f"Reciprocal-rank fusion supplied {selected} {result_word}."
    else:
        result_sentence = f"The retrieval pipeline supplied {selected} {result_word}."
    return f"{retrieval_sentence} {result_sentence}"


__all__ = ["AgenticProgressEvent", "AgenticProgressSink"]
