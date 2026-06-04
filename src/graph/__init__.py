r"""LangGraph multi-agent stack for invest-bot.

This package is the "Multi-agent" counterpart to the single-agent ``src.agent``.
The architecture is the canonical LangGraph supervisor pattern::

                            START
                              |
                              v
                       +-------------+
              +------->| supervisor  |<-------+
              |        +------+------+        |
              |               |               |
              |   +---------+-+-+---------+   |
              |   v         v   v         v   |
              | researcher  |   |       trader|
              |     portfolio_  risk_         |
              |     analyst     officer       |
              +--------/                \-----+
                                |
                                v
                              FINISH

The shared **state** is a list of chat messages. Each specialist is itself a
ReAct sub-agent (``langgraph.prebuilt.create_react_agent``) wired to a focused
subset of the tools defined in ``src.tools.*`` (the same tools used by the
single-agent mode, so business logic stays in one place).

Public API
----------
- :func:`build_graph` -- assemble + compile the StateGraph. Pass a
  :class:`src.config.Settings`, or omit it to read from ``.env``.
"""

from src.graph.build import build_graph

__all__ = ["build_graph"]
