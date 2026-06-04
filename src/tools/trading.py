"""Trading tools: put these on the agent so the LLM can "act" on the virtual account.

[Flow]
1. Fetch the realtime quote (we use it as the fill price — a simplified market order).
2. Compute the account snapshot (needed by risk: equity and cash).
3. Run risk.check_order. On reject, raise (so the agent explains to the user).
4. Call state.apply_fill to mutate portfolio.json. Return the new order.

[Why raise instead of return error]
openai-agents catches exceptions raised by tools and feeds them back to the LLM.
That gives the LLM structured error info to compose a "what went wrong + try this"
explanation for the user.
"""

from __future__ import annotations

from typing import Any

from src.state import (
    apply_fill,
    current_position_qty,
    load_portfolio,
    todays_order_count,
)
from src.tools.market_data import get_quote
from src.tools.portfolio import get_account
from src.tools.risk import check_order


def preview_market_order(symbol: str, side: str, qty: float) -> dict[str, Any]:
    """Dry-run a market order: run the same quote + risk checks, but DO NOT touch the account.

    Returns:
        {
          "would_fill": bool,
          "reason": "<short explanation, OK on success, rejection text on failure>",
          "symbol": "AAPL",
          "side": "buy",
          "qty": 5,
          "estimated_fill_price": 189.12,
          "estimated_value": 945.60
        }

    Use this from the Risk Officer agent (or any callsite) when you want to know
    "would this order be allowed?" without actually placing it.
    """
    symbol = symbol.upper().strip()
    side = side.lower().strip()

    quote = get_quote(symbol)
    if "error" in quote:
        return {
            "would_fill": False,
            "reason": f"Cannot fetch quote for {symbol}: {quote['error']}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
        }
    last_price = float(quote["price"])

    account = get_account()
    state = load_portfolio()
    ok, reason = check_order(
        symbol=symbol,
        side=side,
        qty=qty,
        last_price=last_price,
        account=account,
        todays_order_count=todays_order_count(state),
        current_position_qty=current_position_qty(state, symbol),
    )
    return {
        "would_fill": ok,
        "reason": reason,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "estimated_fill_price": round(last_price, 4),
        "estimated_value": round(qty * last_price, 2),
    }


def place_market_order(symbol: str, side: str, qty: float) -> dict[str, Any]:
    """Place a market order. `side` is 'buy' or 'sell'; `qty` is number of shares.

    Returns on success:
        {"id": "ord_...", "ts": "...", "symbol": "AAPL", "side": "buy",
         "qty": 5, "fill_price": 189.12, "status": "filled"}
    """
    symbol = symbol.upper().strip()
    side = side.lower().strip()

    quote = get_quote(symbol)
    if "error" in quote:
        raise ValueError(f"Cannot fetch quote for {symbol}: {quote['error']}")
    last_price = float(quote["price"])

    account = get_account()
    state = load_portfolio()
    ok, reason = check_order(
        symbol=symbol,
        side=side,
        qty=qty,
        last_price=last_price,
        account=account,
        todays_order_count=todays_order_count(state),
        current_position_qty=current_position_qty(state, symbol),
    )
    if not ok:
        raise ValueError(f"Rejected by risk: {reason}")

    order = apply_fill(state, symbol, side, qty, last_price)
    return order


def cancel_order(order_id: str) -> dict[str, Any]:
    """In our simulated environment market orders fill instantly, so there's
    nothing to cancel. The tool exists to show how to cleanly reject an op.
    """
    raise ValueError(
        f"Cannot cancel market order {order_id}: simulated market orders are "
        "instant-fill. To 'cancel', place an offsetting order."
    )
