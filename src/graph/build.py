"""Assemble the multi-agent ``StateGraph``.

[Graph topology]
- START -> supervisor
- supervisor -> (researcher | portfolio_analyst | risk_officer | trader | END)
  via a conditional edge on ``state["next"]``.
- Each specialist -> supervisor (so the supervisor re-evaluates after every
  step until it decides FINISH).

[Why we keep only the specialist's FINAL message in shared state]
``create_react_agent`` returns the full transcript including ToolMessages for
each call. Letting all of that flow back into the parent state would make the
supervisor's context window explode within a few turns. So we keep just the
specialist's last AIMessage -- the supervisor only needs the conclusion, not
the trace. The full trace is still available via ``graph.stream(...)`` if you
need to surface it (e.g. for the CLI's verbose log).
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings, load_settings
from src.graph.agents import build_specialists
from src.graph.state import InvestState
from src.graph.supervisor import SPECIALISTS, make_supervisor_node


def _wrap_specialist(name: str, agent) -> Callable[[InvestState], dict]:
    """Wrap a compiled ReAct sub-agent so it returns ONLY its final message,
    tagged with the agent's name so the UI can show who said what.
    """

    def node(state: InvestState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        tagged = AIMessage(
            content=last.content if hasattr(last, "content") else str(last),
            name=name,
        )
        return {"messages": [tagged]}

    return node


def build_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Assemble and compile the multi-agent graph.

    Args:
        settings: If ``None``, :func:`src.config.load_settings` reads ``.env``
            (CLI mode). The Streamlit UI builds settings explicitly from the
            user-entered key.

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke({"messages": [...]})``.
    """
    if settings is None:
        settings = load_settings()
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    specialists = build_specialists(settings.openai_model)
    supervisor = make_supervisor_node(settings.openai_model)

    g: StateGraph = StateGraph(InvestState)
    g.add_node("supervisor", supervisor)
    for name, agent in specialists.items():
        g.add_node(name, _wrap_specialist(name, agent))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {**{n: n for n in SPECIALISTS}, "FINISH": END},
    )
    for n in SPECIALISTS:
        g.add_edge(n, "supervisor")

    return g.compile()
