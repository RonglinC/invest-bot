"""The local "virtual account" — a single JSON file as our database.

[What this file does]
We don't connect to any real broker. The entire "account" is a JSON file:
{
  "cash": 100000.00,
  "starting_equity": 100000.00,
  "positions": {"AAPL": {"qty": 10, "avg_entry_price": 189.50}},
  "orders": [...]
}

[Two usage modes]
1. CLI (local): all reads/writes go to data/portfolio.json
2. Web UI (deployed): each browser session uses its own temp JSON file,
   so two users never collide.

The second mode is implemented via thread-local storage: Streamlit runs
each session in its own thread, so "which file to use" is per-thread.

[Beginner concepts cheat sheet]
- threading.local(): a special object whose attributes are visible only
  in the current thread.
- with open(...) as f: automatically closes the file (safer than manual close).
- deepcopy: copies a nested dict thoroughly, so mutating the copy doesn't
  affect the original.
"""

from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PORTFOLIO_FILE = DEFAULT_DATA_DIR / "portfolio.json"
STARTING_CASH = 100_000.00


_tls = threading.local()


def set_portfolio_path(path: Path) -> None:
    """Override the storage path for the current thread.
    Streamlit calls this once per session right at the top of app.py.
    """
    _tls.path = Path(path)


def get_portfolio_path() -> Path:
    return getattr(_tls, "path", DEFAULT_PORTFOLIO_FILE)


def _empty_portfolio() -> dict[str, Any]:
    return {
        "cash": STARTING_CASH,
        "starting_equity": STARTING_CASH,
        "positions": {},
        "orders": [],
    }


def load_portfolio() -> dict[str, Any]:
    """Read the JSON. If missing, create it with the initial $100k state."""
    path = get_portfolio_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_portfolio(_empty_portfolio())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio(state: dict[str, Any]) -> None:
    path = get_portfolio_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def reset_portfolio() -> dict[str, Any]:
    """Wipe and reinitialize the account. Used by tests and the "Reset" button."""
    fresh = _empty_portfolio()
    save_portfolio(fresh)
    return fresh


def new_order_id() -> str:
    return "ord_" + uuid.uuid4().hex[:8]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def todays_order_count(state: dict[str, Any]) -> int:
    """Count how many orders were placed today (used by risk.py)."""
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for o in state["orders"] if o["ts"].startswith(today))


def current_position_qty(state: dict[str, Any], symbol: str) -> float:
    """How many shares of `symbol` are currently held? 0 if none."""
    pos = state["positions"].get(symbol.upper())
    return float(pos["qty"]) if pos else 0.0


def apply_fill(
    state: dict[str, Any],
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
) -> dict[str, Any]:
    """Apply a fill: update cash, update positions, append an order. Return the new order.

    Buy:  cash decreases by qty*price; position increases (weighted-avg cost).
    Sell: cash increases by qty*price; position decreases (removed when zero).
    """
    state = deepcopy(state)
    symbol = symbol.upper()
    side = side.lower()
    cost = qty * fill_price

    if side == "buy":
        state["cash"] -= cost
        pos = state["positions"].get(symbol)
        if pos:
            old_qty = float(pos["qty"])
            old_avg = float(pos["avg_entry_price"])
            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * fill_price) / new_qty
            state["positions"][symbol] = {
                "qty": new_qty,
                "avg_entry_price": round(new_avg, 4),
            }
        else:
            state["positions"][symbol] = {
                "qty": qty,
                "avg_entry_price": round(fill_price, 4),
            }
    elif side == "sell":
        state["cash"] += cost
        pos = state["positions"].get(symbol)
        if not pos:
            raise ValueError(f"Cannot sell {symbol}: no position held")
        remaining = float(pos["qty"]) - qty
        if remaining < -1e-9:
            raise ValueError(f"Tried to sell {qty} shares of {symbol}, but only hold {pos['qty']}")
        if abs(remaining) < 1e-9:
            del state["positions"][symbol]
        else:
            state["positions"][symbol] = {
                "qty": remaining,
                "avg_entry_price": pos["avg_entry_price"],
            }
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    order = {
        "id": new_order_id(),
        "ts": now_iso(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "fill_price": round(fill_price, 4),
        "status": "filled",
    }
    state["orders"].append(order)
    save_portfolio(state)
    return order
