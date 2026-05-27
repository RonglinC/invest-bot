"""Risk gate: hard pre-trade checks that the LLM cannot bypass.

[Why this layer]
LLMs occasionally do something dumb — try to go all in on one ticker, fire
50 orders per minute, or sell stock they don't own. trading.place_market_order
is required to call check_order(...) first and raise on rejection. The agent
then sees the error and explains it to the user instead of going around it.

[Design principles]
- Rules are hard-coded, not just "please don't..." prompted. LLMs ignore those.
- Return (ok, reason) instead of raising, so callers decide how to handle.
- Thresholds live in a dataclass so they're easy to tweak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct_of_equity: float = 0.05
    max_orders_per_day: int = 10
    allow_short: bool = False
    allowed_sides: tuple = ("buy", "sell")
    min_qty: float = 1.0


DEFAULT_CONFIG = RiskConfig()


def check_order(
    symbol: str,
    side: str,
    qty: float,
    last_price: float,
    account: dict[str, Any],
    todays_order_count: int,
    current_position_qty: float,
    cfg: RiskConfig = DEFAULT_CONFIG,
) -> tuple[bool, str]:
    """Returns (allowed, reason)."""
    side = side.lower().strip()
    symbol = symbol.upper().strip()

    if side not in cfg.allowed_sides:
        return False, f"side must be one of {cfg.allowed_sides}, got {side!r}"

    if qty < cfg.min_qty:
        return False, f"Quantity too small ({qty}); minimum is {cfg.min_qty} share(s)"

    if last_price <= 0:
        return False, f"Invalid reference price ({last_price})"

    order_value = qty * last_price
    equity = float(account.get("equity", 0))
    max_value = equity * cfg.max_position_pct_of_equity

    if side == "buy":
        if order_value > max_value:
            return False, (
                f"Order value ${order_value:.2f} exceeds the per-trade cap of "
                f"{cfg.max_position_pct_of_equity * 100:.0f}% of equity "
                f"(max ${max_value:.2f}). Try a smaller quantity."
            )
        cash = float(account.get("cash", 0))
        if order_value > cash:
            return False, f"Not enough cash: need ${order_value:.2f}, account has ${cash:.2f}"

    if side == "sell":
        if not cfg.allow_short and qty > current_position_qty:
            return False, (
                f"Naked shorts disabled: tried to sell {qty} shares of {symbol}, "
                f"but only hold {current_position_qty}"
            )

    if todays_order_count >= cfg.max_orders_per_day:
        return False, f"Already placed {todays_order_count} orders today, daily cap is {cfg.max_orders_per_day}"

    return True, "OK"
