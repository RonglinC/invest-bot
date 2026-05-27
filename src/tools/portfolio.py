"""Account and position tools: let the agent see cash, holdings, and order history.

[What this file does]
Three read-only functions, all backed by the local data/portfolio.json file.
Computing "total equity" requires the current price of each holding, so
get_account calls market_data.get_quote internally.
"""

from __future__ import annotations

from typing import Any

from src.state import load_portfolio
from src.tools.market_data import get_quote


def get_account() -> dict[str, Any]:
    """Summarize the account: cash + position market value = equity, plus total P/L vs starting equity."""
    state = load_portfolio()
    cash = float(state["cash"])
    starting = float(state["starting_equity"])

    positions_value = 0.0
    for symbol, pos in state["positions"].items():
        quote = get_quote(symbol)
        if "error" in quote:
            continue
        positions_value += float(pos["qty"]) * float(quote["price"])

    equity = cash + positions_value
    pnl = equity - starting
    pnl_pct = (pnl / starting) * 100 if starting else 0.0

    return {
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "equity": round(equity, 2),
        "starting_equity": round(starting, 2),
        "total_pnl": round(pnl, 2),
        "total_pnl_pct": round(pnl_pct, 2),
    }


def get_positions() -> list[dict[str, Any]]:
    """List all holdings, enriched with current price, market value, and unrealized P/L."""
    state = load_portfolio()
    out = []
    for symbol, pos in state["positions"].items():
        qty = float(pos["qty"])
        avg_entry = float(pos["avg_entry_price"])
        quote = get_quote(symbol)
        current_price = float(quote.get("price", avg_entry))
        market_value = qty * current_price
        cost_basis = qty * avg_entry
        unrealized_pl = market_value - cost_basis
        unrealized_pl_pct = (unrealized_pl / cost_basis) * 100 if cost_basis else 0.0
        out.append({
            "symbol": symbol,
            "qty": qty,
            "avg_entry_price": round(avg_entry, 4),
            "current_price": round(current_price, 4),
            "market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_pl_pct": round(unrealized_pl_pct, 2),
        })
    return out


def get_orders(limit: int = 20) -> list[dict[str, Any]]:
    """Recent fills (most recent first, default 20)."""
    state = load_portfolio()
    orders = list(reversed(state["orders"]))[:limit]
    return orders
