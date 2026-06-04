"""Supervisor node: routes the conversation to one specialist, or to FINISH.

[Design choice: structured output instead of free-form tool call]
We use :meth:`langchain_openai.ChatOpenAI.with_structured_output` so the LLM is
*forced* to return one of ``{"researcher", "portfolio_analyst", "risk_officer",
"trader", "FINISH"}`` plus a short reason string. This is more reliable than
parsing free-form text -- the OpenAI SDK uses the function-calling pathway
under the hood, which gives us a Pydantic object back, never a malformed JSON.

[Why a separate prompt]
Routing is a different skill than executing. The supervisor sees the full
conversation (user + all prior specialists' replies) and answers a single
question: "who acts next?". That's a small, well-scoped LLM task.
"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.graph.state import InvestState


SPECIALISTS: list[str] = ["researcher", "portfolio_analyst", "risk_officer", "trader"]


class Route(BaseModel):
    """Supervisor's structured-output schema."""

    next: Literal[
        "researcher",
        "portfolio_analyst",
        "risk_officer",
        "trader",
        "FINISH",
    ] = Field(
        description=(
            "Which specialist should act next, or FINISH when the user's "
            "request has been fully answered."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the routing decision (shown to the user).",
    )


SUPERVISOR_PROMPT = """You are the SUPERVISOR of an invest-bot team. You decide
WHO acts next. You do NOT answer the user yourself.

[Your team]
- researcher: market quotes, bars, news. Best for "what's X?", "how did Y do?".
- portfolio_analyst: cash, positions, P/L, order history. Best for "what do I
  own?", "how am I doing?".
- risk_officer: dry-runs proposed trades through the risk gate; explains pass
  / reject reasoning. Best BEFORE a trade.
- trader: actually places the (paper) order. ONLY route here AFTER the user
  has CONFIRMED explicitly (e.g. "yes", "confirm", "proceed", "确认", "下单").

[Routing rules]
1. Read the full conversation -- both user messages and prior specialist
   replies.
2. Pick the SINGLE specialist whose tools / knowledge fit the next missing
   piece. Routing two specialists in parallel is not supported -- one at a
   time.
3. If the latest specialist already answered the user's question fully,
   return FINISH.
4. If the user said something like "buy 5 AAPL", route to risk_officer FIRST
   (never directly to trader).
5. Only route to trader when the most recent user message contains explicit
   confirmation ("yes", "confirm", "proceed", "ok", "去做", "确认", "下单").
   If you are unsure, FINISH and let the user explicitly say yes.
6. Avoid loops. Do not pick the same specialist twice in a row unless new user
   input has arrived since their last turn.
7. Your `reason` will be shown verbatim to the user; keep it under 20 words.
"""


def make_supervisor_node(model: str) -> Callable[[InvestState], dict]:
    """Build the supervisor callable. Bound to a specific model name.

    Returns a function with signature ``(state) -> {"next": str, "messages": [..]}``
    that LangGraph adds as a node.
    """
    llm = ChatOpenAI(model=model, temperature=0)
    router = llm.with_structured_output(Route)

    def supervisor_node(state: InvestState) -> dict:
        decision: Route = router.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]],
        )
        narration = AIMessage(
            content=f"[supervisor -> {decision.next}] {decision.reason}",
            name="supervisor",
        )
        return {"next": decision.next, "messages": [narration]}

    return supervisor_node
