"""CLI entry point: chat with the agent in your terminal.

[How to run]
    source .venv/bin/activate
    python -m src.main

[How to end the chat]
Type exit / quit / empty line, or hit Ctrl-D.

[What is rich]
A library that gives terminal output colors and formatting. We use it for
the "you> / bot>" prompts and the account summary banner. Plain print(...)
would also work — rich is just nicer.
"""

from __future__ import annotations

from agents import Runner
from rich.console import Console
from rich.panel import Panel

from src.agent import build_agent
from src.tools.portfolio import get_account


console = Console()


def print_banner() -> None:
    acc = get_account()
    console.print(Panel.fit(
        f"[bold cyan]invest-bot[/bold cyan]  (paper account)\n"
        f"cash: ${acc['cash']:,.2f}    positions: ${acc['positions_value']:,.2f}    "
        f"equity: ${acc['equity']:,.2f}    return: {acc['total_pnl_pct']:+.2f}%\n"
        f"[dim]Type 'exit' to quit. First message may take 5-10 seconds while the model warms up.[/dim]",
        border_style="cyan",
    ))


def main() -> None:
    print_banner()
    agent = build_agent()

    while True:
        try:
            user = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not user or user.lower() in {"exit", "quit", "q"}:
            console.print("[dim]bye[/dim]")
            return

        try:
            result = Runner.run_sync(agent, user)
        except Exception as e:
            console.print(f"[red]error: {type(e).__name__}: {e}[/red]")
            continue

        console.print(f"[bold magenta]bot>[/bold magenta] {result.final_output}\n")


if __name__ == "__main__":
    main()
