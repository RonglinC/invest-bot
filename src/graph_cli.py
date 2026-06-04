"""CLI entry point for the LangGraph **multi-agent** mode.

[How to run]
    source .venv/bin/activate
    python -m src.graph_cli                 # interactive REPL
    python -m src.graph_cli --mermaid        # print the graph as Mermaid diagram and exit

[Difference from src.main]
``src.main`` uses the single-agent ``openai-agents`` stack (one LLM, all tools).
This file drives the supervisor+specialists graph from ``src.graph``. The two
modes share the same underlying tools (``src.tools.*``), risk gate, and ledger
(``src.state``), so they are interchangeable from a user perspective -- just
different orchestrations.

[Verbose streaming]
We use ``graph.stream(...)`` which emits one event per node finishing. For
each event we print the node name (e.g. ``supervisor``, ``researcher``) plus
the new message(s) that node added to state. This makes the multi-agent dance
visible in the terminal, which is the main pedagogical win of multi-agent.
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.graph import build_graph
from src.tools.portfolio import get_account


console = Console()

AGENT_STYLES = {
    "supervisor": "yellow",
    "researcher": "cyan",
    "portfolio_analyst": "blue",
    "risk_officer": "red",
    "trader": "magenta",
}


def _print_banner() -> None:
    acc = get_account()
    console.print(Panel.fit(
        f"[bold cyan]invest-bot[/bold cyan]  [dim](multi-agent / LangGraph)[/dim]\n"
        f"cash: ${acc['cash']:,.2f}    positions: ${acc['positions_value']:,.2f}    "
        f"equity: ${acc['equity']:,.2f}    return: {acc['total_pnl_pct']:+.2f}%\n"
        f"[dim]Team: supervisor, researcher, portfolio_analyst, risk_officer, trader. "
        f"Type 'exit' to quit.[/dim]",
        border_style="cyan",
    ))


def _render_event(event: dict) -> None:
    """LangGraph ``stream`` yields one dict per finished node. Each dict has a
    single key (the node name) whose value is the state delta returned by that
    node. We print the node's name + any new messages it produced.
    """
    for node_name, delta in event.items():
        new_messages = delta.get("messages", []) if isinstance(delta, dict) else []
        if not new_messages:
            continue
        style = AGENT_STYLES.get(node_name, "white")
        for msg in new_messages:
            text = getattr(msg, "content", "") or ""
            if not text:
                continue
            console.print(f"[bold {style}]{node_name}>[/bold {style}] {text}")


def chat() -> None:
    _print_banner()
    graph = build_graph()

    history: list = []

    while True:
        try:
            user = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not user or user.lower() in {"exit", "quit", "q"}:
            console.print("[dim]bye[/dim]")
            return

        history.append(HumanMessage(content=user))

        console.print(Rule(style="dim"))
        try:
            final_state = None
            for event in graph.stream(
                {"messages": history},
                config={"recursion_limit": 25},
                stream_mode="updates",
            ):
                _render_event(event)
                final_state = event
            # Capture the specialist's final reply into our local history so
            # the next turn keeps multi-turn context.
            if final_state:
                for delta in final_state.values():
                    if isinstance(delta, dict):
                        for m in delta.get("messages", []):
                            if isinstance(m, AIMessage):
                                history.append(m)
        except Exception as e:
            console.print(f"[red]error: {type(e).__name__}: {e}[/red]")
        console.print(Rule(style="dim"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="invest-bot multi-agent CLI (LangGraph)",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print the compiled graph as a Mermaid diagram and exit (no LLM call needed).",
    )
    args = parser.parse_args()

    if args.mermaid:
        # Use a dummy key purely so build_settings is happy; no API call is made.
        from src.config import build_settings
        graph = build_graph(build_settings("sk-mermaid-only-no-call"))
        print(graph.get_graph().draw_mermaid())
        sys.exit(0)

    chat()


if __name__ == "__main__":
    main()
