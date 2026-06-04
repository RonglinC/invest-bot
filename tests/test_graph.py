"""Offline tests for the LangGraph multi-agent stack.

[Scope of these tests]
We test the **wiring** -- can the graph be built? do node names match? does
the conditional edge route on ``state['next']``? -- without making any LLM
calls. Live agent behavior is intentionally NOT covered here because:

1. Calling OpenAI from tests is slow, flaky, and costs money.
2. ``create_react_agent`` is upstream code we already trust.
3. The interesting LangGraph-specific glue (state, routing, edges) is the
   exact surface area these tests cover.

[How we test routing without an LLM]
We monkey-patch ``make_supervisor_node`` to return a hard-coded plan -- a
list of agent names to dispatch one-by-one -- so we can assert that the
graph really does visit each specialist in order. Each specialist node is
also stubbed to record that it ran and append a fake reply.
"""

from __future__ import annotations

import importlib
from typing import Iterator

import pytest

import src.state as state_mod
import src.graph.build as graph_build
import src.graph.supervisor as graph_supervisor


@pytest.fixture(autouse=True)
def isolated_portfolio(tmp_path, monkeypatch):
    """Each test runs with its own temp portfolio.json. Also stub the API key
    so ``build_graph`` doesn't reject ``sk-test-...`` via the env-var ValueError.
    """
    state_mod.set_portfolio_path(tmp_path / "portfolio.json")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    yield
    importlib.reload(state_mod)


def _fake_supervisor_factory(plan: Iterator[str]):
    """Return a ``make_supervisor_node`` replacement that walks a fixed plan."""
    from langchain_core.messages import AIMessage

    def factory(model: str):
        def node(state):
            try:
                nxt = next(plan)
            except StopIteration:
                nxt = "FINISH"
            return {
                "next": nxt,
                "messages": [AIMessage(content=f"[stub-supervisor] -> {nxt}", name="supervisor")],
            }
        return node

    return factory


def _fake_specialists():
    """Build specialist stubs that record invocations and return a labeled reply."""
    from langchain_core.messages import AIMessage
    from langgraph.graph import StateGraph, END

    from src.graph.state import InvestState

    called: list[str] = []

    def make_stub(name: str):
        # We need each "specialist" to be a thing whose `.invoke({"messages": ...})`
        # returns `{"messages": [final AIMessage]}` -- exactly what create_react_agent
        # produces. The simplest such thing is a one-node compiled graph.
        sub = StateGraph(InvestState)

        def reply(state):
            called.append(name)
            return {"messages": [AIMessage(content=f"{name} did its thing", name=name)]}

        sub.add_node(name, reply)
        sub.set_entry_point(name)
        sub.add_edge(name, END)
        return sub.compile()

    specialists = {
        "researcher": make_stub("researcher"),
        "portfolio_analyst": make_stub("portfolio_analyst"),
        "risk_officer": make_stub("risk_officer"),
        "trader": make_stub("trader"),
    }
    return specialists, called


def test_graph_compiles_with_expected_nodes(monkeypatch):
    """Compiling the graph exposes one supervisor + four specialists + start/end."""
    monkeypatch.setattr(graph_supervisor, "make_supervisor_node", _fake_supervisor_factory(iter([])))
    monkeypatch.setattr(graph_build, "make_supervisor_node", _fake_supervisor_factory(iter([])))
    specialists, _ = _fake_specialists()
    monkeypatch.setattr(graph_build, "build_specialists", lambda model: specialists)

    from src.config import build_settings
    from src.graph import build_graph

    graph = build_graph(build_settings("sk-test-dummy"))
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "__end__",
        "supervisor",
        "researcher",
        "portfolio_analyst",
        "risk_officer",
        "trader",
    }
    assert expected.issubset(nodes), f"missing: {expected - nodes}"


def test_supervisor_routes_to_each_specialist(monkeypatch):
    """A plan of [researcher, portfolio_analyst, risk_officer, trader] visits all four."""
    plan = iter(["researcher", "portfolio_analyst", "risk_officer", "trader", "FINISH"])
    monkeypatch.setattr(graph_build, "make_supervisor_node", _fake_supervisor_factory(plan))
    specialists, called = _fake_specialists()
    monkeypatch.setattr(graph_build, "build_specialists", lambda model: specialists)

    from langchain_core.messages import HumanMessage

    from src.config import build_settings
    from src.graph import build_graph

    graph = build_graph(build_settings("sk-test-dummy"))
    final = graph.invoke(
        {"messages": [HumanMessage(content="hello team")]},
        config={"recursion_limit": 25},
    )
    assert called == ["researcher", "portfolio_analyst", "risk_officer", "trader"]
    # Final state has supervisor narration + 4 specialist replies + the user msg.
    contents = [getattr(m, "content", "") for m in final["messages"]]
    assert any("researcher did its thing" in c for c in contents)
    assert any("trader did its thing" in c for c in contents)


def test_supervisor_finish_skips_specialists(monkeypatch):
    """If the supervisor immediately FINISHes, no specialist should run."""
    monkeypatch.setattr(graph_build, "make_supervisor_node", _fake_supervisor_factory(iter(["FINISH"])))
    specialists, called = _fake_specialists()
    monkeypatch.setattr(graph_build, "build_specialists", lambda model: specialists)

    from langchain_core.messages import HumanMessage

    from src.config import build_settings
    from src.graph import build_graph

    graph = build_graph(build_settings("sk-test-dummy"))
    graph.invoke(
        {"messages": [HumanMessage(content="nothing to do")]},
        config={"recursion_limit": 25},
    )
    assert called == []


def test_preview_market_order_dry_run_does_not_mutate_account():
    """Risk Officer's primary tool must be side-effect-free."""
    from src.tools import portfolio, trading

    before = portfolio.get_account()
    # Cheap symbol-less call: we don't need yfinance to succeed -- we just want to
    # verify that, no matter the outcome, the ledger isn't touched. Use a
    # nonsense symbol so the quote lookup fails fast and we exit before any
    # state mutation could occur.
    result = trading.preview_market_order(symbol="ZZZZZZ_NOT_A_TICKER", side="buy", qty=1)
    after = portfolio.get_account()

    assert result["would_fill"] is False  # bad symbol -> rejected
    assert before == after  # account untouched
