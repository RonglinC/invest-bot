"""Streamlit Web UI for invest-bot.

[How to run locally]
    source .venv/bin/activate
    streamlit run app.py

The browser will open http://localhost:8501 automatically.

[How to deploy online]
See README.md > "Deploy online". TL;DR: push to GitHub -> connect at
share.streamlit.io -> get a public URL.

[Streamlit mental model — read before touching this code]
Streamlit is NOT a normal web framework. Every time the user clicks a
button or types a message, the entire script reruns top to bottom.
- Want to keep state? Use st.session_state (per browser tab, lost on refresh)
- Want to avoid redoing expensive work? Use @st.cache_resource / @st.cache_data
- Render order = code execution order, top to bottom
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="invest-bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Each browser session gets its own virtual account ----------
# Key requirement: we must set the storage path BEFORE any tool calls,
# so subsequent state I/O lands on this session's temp file.
import src.state as state_mod

if "portfolio_path" not in st.session_state:
    pid = uuid.uuid4().hex[:12]
    st.session_state["portfolio_path"] = Path(tempfile.gettempdir()) / f"invest-bot-{pid}.json"

state_mod.set_portfolio_path(st.session_state["portfolio_path"])


# ---------- Sidebar: key, model, account summary, reset ----------
from src.config import DEFAULT_MODEL, build_settings
from src.tools.portfolio import get_account, get_positions

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("OpenAI")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        help="Get yours at https://platform.openai.com/api-keys. Not stored anywhere.",
    )
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
        help="gpt-4o-mini is the cheapest and fastest; great for learning.",
    )

    st.subheader("Agent mode")
    agent_mode = st.radio(
        "Which orchestration?",
        options=["Single agent (openai-agents)", "Multi-agent (LangGraph)"],
        index=1,
        help=(
            "Single agent = one LLM with all tools.\n\n"
            "Multi-agent = a supervisor routes between researcher, "
            "portfolio analyst, risk officer, and trader -- each with its own "
            "focused tool subset."
        ),
    )
    is_multi = agent_mode.startswith("Multi")

    st.divider()

    st.subheader("💰 My account")
    try:
        acc = get_account()
        st.metric("Equity", f"${acc['equity']:,.2f}", f"{acc['total_pnl_pct']:+.2f}%")
        col1, col2 = st.columns(2)
        col1.metric("Cash", f"${acc['cash']:,.2f}")
        col2.metric("Positions", f"${acc['positions_value']:,.2f}")

        positions = get_positions()
        if positions:
            st.caption("Holdings")
            st.dataframe(
                positions,
                column_config={
                    "symbol": "Symbol",
                    "qty": "Qty",
                    "avg_entry_price": st.column_config.NumberColumn("Cost", format="$%.2f"),
                    "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "unrealized_pl": st.column_config.NumberColumn("P/L", format="$%.2f"),
                    "unrealized_pl_pct": st.column_config.NumberColumn("P/L %", format="%.2f%%"),
                },
                hide_index=True,
                use_container_width=True,
            )
    except Exception as e:
        st.warning(f"Failed to read account: {e}")

    st.divider()
    if st.button("🔄 Reset account (start over)", use_container_width=True):
        state_mod.reset_portfolio()
        st.session_state.pop("messages", None)
        st.session_state.pop("graph_messages", None)
        st.rerun()

    st.caption(
        "100% simulated — no real money. "
        f"Session ID: `{st.session_state['portfolio_path'].stem}`"
    )


# ---------- Main area: chat ----------
st.title("📈 invest-bot")
st.caption("A tool-using LLM agent. It can fetch quotes, read news, view positions, and place simulated orders.")

if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to start.", icon="ℹ️")
    st.stop()

# ---- Build whichever orchestrator the user chose ----
# @st.cache_resource keys on (key, model, is_multi) so switching modes returns
# a freshly built object (the user expects an immediate switch).
from agents import Runner
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_agent
from src.graph import build_graph


@st.cache_resource(show_spinner=False)
def get_single_agent(_key: str, _model: str):
    return build_agent(build_settings(_key, _model))


@st.cache_resource(show_spinner=False)
def get_multi_graph(_key: str, _model: str):
    return build_graph(build_settings(_key, _model))


try:
    if is_multi:
        graph = get_multi_graph(api_key, model)
    else:
        agent = get_single_agent(api_key, model)
except ValueError as e:
    st.error(f"Config error: {e}")
    st.stop()


# ---- Message history ----
# We store messages in a uniform shape: {"role": "user"|"assistant", "content", "name"?}.
# The optional "name" lets us label *which* agent spoke (e.g. researcher, risk_officer).
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm invest-bot. Ask about your account, quotes, news, or tell me "
                "to place a simulated order."
            ),
        }
    ]

# Keep an in-memory LangGraph message list parallel to the UI list, so
# multi-agent conversations can carry context across turns.
if "graph_messages" not in st.session_state:
    st.session_state["graph_messages"] = []


def _render_message(msg: dict) -> None:
    """Render one stored message, with an agent-name badge when present."""
    with st.chat_message(msg["role"]):
        name = msg.get("name")
        if name:
            badge_colors = {
                "supervisor": "#eab308",
                "researcher": "#06b6d4",
                "portfolio_analyst": "#3b82f6",
                "risk_officer": "#ef4444",
                "trader": "#a855f7",
            }
            color = badge_colors.get(name, "#6b7280")
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:6px;font-size:0.75em;font-weight:600'>{name}</span>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])


for msg in st.session_state["messages"]:
    _render_message(msg)


# ---- Input box ----
if user_input := st.chat_input("Talk to the bot… (e.g. 'What's AAPL right now?')"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if is_multi:
        # Multi-agent: stream node-by-node and render each agent's reply with a badge.
        st.session_state["graph_messages"].append(HumanMessage(content=user_input))
        try:
            with st.spinner("Supervisor + specialists working…"):
                final_messages: list = []
                for event in graph.stream(
                    {"messages": st.session_state["graph_messages"]},
                    config={"recursion_limit": 25},
                    stream_mode="updates",
                ):
                    for node_name, delta in event.items():
                        if not isinstance(delta, dict):
                            continue
                        for m in delta.get("messages", []):
                            text = getattr(m, "content", "") or ""
                            if not text:
                                continue
                            stored = {"role": "assistant", "content": text, "name": node_name}
                            st.session_state["messages"].append(stored)
                            _render_message(stored)
                            if isinstance(m, AIMessage):
                                final_messages.append(m)
                st.session_state["graph_messages"].extend(final_messages)
        except Exception as e:
            err = f"❌ Error: `{type(e).__name__}`: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": err})
            _render_message({"role": "assistant", "content": err})
    else:
        # Single-agent: classic one-shot call to openai-agents Runner.
        with st.chat_message("assistant"):
            with st.spinner("Agent thinking + calling tools…"):
                try:
                    result = Runner.run_sync(agent, user_input)
                    answer = result.final_output
                except Exception as e:
                    answer = f"❌ Error: `{type(e).__name__}`: {e}"
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.rerun()  # Trigger one more rerun so the sidebar account summary refreshes
