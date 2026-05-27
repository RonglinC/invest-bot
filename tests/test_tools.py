"""Offline unit tests for state and risk logic. No network, no OpenAI required.

[pytest cheat sheet]
- A function whose name starts with test_ IS a test.
- `assert <condition>`: fails if the condition is falsy.
- Run the suite with: `pytest -v`

[Why not test market_data or agent here]
- market_data needs the internet; would fail offline.
- agent costs OpenAI tokens every run.
We cover the most critical "the ledger cannot lie" and "risk cannot be bypassed"
logic with pure-Python tests.
"""

from __future__ import annotations

import importlib

import pytest

import src.state as state_mod
import src.tools.risk as risk_mod


@pytest.fixture(autouse=True)
def isolated_portfolio(tmp_path):
    """Each test runs with its own temp portfolio.json so they don't pollute each other."""
    state_mod.set_portfolio_path(tmp_path / "portfolio.json")
    yield
    importlib.reload(state_mod)


def test_fresh_portfolio_starts_with_100k():
    s = state_mod.load_portfolio()
    assert s["cash"] == 100_000.00
    assert s["positions"] == {}
    assert s["orders"] == []


def test_buy_decreases_cash_and_adds_position():
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=10, fill_price=200.0)
    s = state_mod.load_portfolio()
    assert s["cash"] == 100_000.00 - 2_000.00
    assert s["positions"]["AAPL"]["qty"] == 10
    assert s["positions"]["AAPL"]["avg_entry_price"] == 200.0


def test_second_buy_updates_weighted_avg():
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=10, fill_price=200.0)
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=10, fill_price=300.0)
    s = state_mod.load_portfolio()
    assert s["positions"]["AAPL"]["qty"] == 20
    assert s["positions"]["AAPL"]["avg_entry_price"] == 250.0


def test_sell_increases_cash_and_reduces_position():
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=10, fill_price=200.0)
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "sell", qty=4, fill_price=250.0)
    s = state_mod.load_portfolio()
    assert s["positions"]["AAPL"]["qty"] == 6
    assert s["cash"] == 100_000.00 - 2_000.00 + 4 * 250.0


def test_sell_all_removes_position():
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=10, fill_price=200.0)
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "sell", qty=10, fill_price=210.0)
    s = state_mod.load_portfolio()
    assert "AAPL" not in s["positions"]


def test_cannot_sell_more_than_owned():
    s = state_mod.load_portfolio()
    state_mod.apply_fill(s, "AAPL", "buy", qty=5, fill_price=100.0)
    s = state_mod.load_portfolio()
    with pytest.raises(ValueError):
        state_mod.apply_fill(s, "AAPL", "sell", qty=10, fill_price=100.0)


def test_risk_rejects_oversized_buy():
    account = {"equity": 100_000.0, "cash": 100_000.0}
    ok, reason = risk_mod.check_order(
        symbol="AAPL", side="buy", qty=100, last_price=200.0,
        account=account, todays_order_count=0, current_position_qty=0,
    )
    assert not ok
    assert "exceeds" in reason.lower() or "cap" in reason.lower()


def test_risk_rejects_short_when_disabled():
    account = {"equity": 100_000.0, "cash": 100_000.0}
    ok, reason = risk_mod.check_order(
        symbol="AAPL", side="sell", qty=10, last_price=100.0,
        account=account, todays_order_count=0, current_position_qty=0,
    )
    assert not ok
    assert "short" in reason.lower()


def test_risk_rejects_overfrequent_orders():
    account = {"equity": 100_000.0, "cash": 100_000.0}
    ok, reason = risk_mod.check_order(
        symbol="AAPL", side="buy", qty=1, last_price=100.0,
        account=account, todays_order_count=10, current_position_qty=0,
    )
    assert not ok
    assert "today" in reason.lower() or "daily" in reason.lower()


def test_risk_allows_reasonable_buy():
    account = {"equity": 100_000.0, "cash": 100_000.0}
    ok, reason = risk_mod.check_order(
        symbol="AAPL", side="buy", qty=10, last_price=200.0,  # $2000 = 2% of equity
        account=account, todays_order_count=0, current_position_qty=0,
    )
    assert ok, reason
