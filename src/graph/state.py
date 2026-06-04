"""Shared graph state.

[LangGraph state primer]
A LangGraph state is just a ``TypedDict``. The novel piece is the
``Annotated[..., reducer]`` syntax: when a node returns ``{"messages": [m]}``,
LangGraph calls the reducer (``add_messages``) to merge it into the existing
list rather than overwriting. ``add_messages`` is the canonical reducer for
chat-style state -- it dedupes by message id and appends new ones.

[Why we also track ``next``]
The supervisor node fills in ``next`` (which specialist to call). The
conditional edge then routes on that string. We keep it in state so the
streamed UI can show "the supervisor decided X" before the specialist runs.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


AgentName = Literal[
    "researcher",
    "portfolio_analyst",
    "risk_officer",
    "trader",
    "FINISH",
]


class InvestState(TypedDict):
    """Shared state passed between every node in the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: AgentName
