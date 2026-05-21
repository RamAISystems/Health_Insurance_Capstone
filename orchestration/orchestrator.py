# updated orchestrator.py
"""
Health Insurance AI Copilot — LangGraph Sequential Chain Orchestrator.
Updated with:

- Retrieval grading
- Groundedness evaluation
- Confidence scoring
- One corrective retry loop
- Safe hallucination prevention
"""

import os
import sys
import json
from typing import TypedDict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from rich.console import Console

from config import (
    LLM_MODEL,
    CLASSIFIER_LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)

from orchestration.tools import (
    policy_search,
    relational_search,
    plan_comparison_search,
    prior_auth_search,
)

from orchestration.tracing import trace_log

load_dotenv()
console = Console()


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_context: str
    answer: str
    chat_history: List[tuple]
    steps_log: List[str]

    retrieval_grade: str
    grounded: bool
    confidence_score: float
    retry_count: int


# ══════════════════════════════════════════════════════════════
# LLM HELPERS
# ══════════════════════════════════════════════════════════════


def _classifier_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=CLASSIFIER_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )



def _synthesis_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# ══════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """You are a query intent classifier for a Health Insurance AI assistant.

Classify the user query into EXACTLY ONE of these categories:

SIMPLE_LOOKUP
POLICY_QUESTION
MULTI_HOP
COMPARISON

Respond with ONLY the category name.
"""


# ══════════════════════════════════════════════════════════════
# SYNTHESIS PROMPT
# ══════════════════════════════════════════════════════════════

_SYNTHESIS_TEMPLATE = """{system_prompt}

─── RETRIEVED CONTEXT ────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

─── CONVERSATION HISTORY ─────────────────────────────────────
{history}
──────────────────────────────────────────────────────────────

Using ONLY the retrieved context above, answer the user's question.
Always cite source file and page number.
If context is insufficient, say so clearly.
Do not hallucinate.
"""


# ══════════════════════════════════════════════════════════════
# EVALUATION PROMPT
# ══════════════════════════════════════════════════════════════

_EVALUATION_PROMPT = """
You are an answer quality evaluator for a healthcare insurance AI system.

Evaluate whether the answer:

1. Is grounded in retrieved context
2. Contains citations
3. Avoids hallucinations
4. Answers the user question completely
5. Avoids unsupported claims

Return ONLY valid JSON:

{
  "grounded": true,
  "confidence_score": 0.92,
  "reason": "short explanation"
}
"""


# ══════════════════════════════════════════════════════════════
# NODE 1 — CLASSIFY INTENT
# ══════════════════════════════════════════════════════════════


def classify_intent(state: AgentState) -> AgentState:
    llm = _classifier_llm()

    query = state["query"]
    log = list(state.get("steps_log", []))

    response = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=query),
    ])

    raw = response.content.strip().upper()

    valid = {
        "SIMPLE_LOOKUP",
        "POLICY_QUESTION",
        "MULTI_HOP",
        "COMPARISON",
    }

    intent = raw if raw in valid else "POLICY_QUESTION"

    log.append(f"🔍 Intent classified → {intent}")

    return {
        **state,
        "intent": intent,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# NODE 2 — RETRIEVE
# ══════════════════════════════════════════════════════════════


def retrieve(state: AgentState) -> AgentState:
    query = state["query"]
    intent = state["intent"]

    log = list(state.get("steps_log", []))
    parts = []

    internal_logs = []
    token = trace_log.set(internal_logs)

    if intent == "SIMPLE_LOOKUP":
        log.append("📊 SIMPLE_LOOKUP retrieval")

        graph_ctx = relational_search.invoke({"query": query})
        if graph_ctx and "No structured" not in graph_ctx:
            parts.append(f"[GRAPH]\n{graph_ctx}")

        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY]\n{policy_ctx}")

    elif intent == "POLICY_QUESTION":
        log.append("📄 POLICY_QUESTION retrieval")

        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY]\n{policy_ctx}")

    elif intent == "MULTI_HOP":
        log.append("🔗 MULTI_HOP retrieval")

        graph_ctx = relational_search.invoke({"query": query})
        if graph_ctx and "No structured" not in graph_ctx:
            parts.append(f"[GRAPH]\n{graph_ctx}")

        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY]\n{policy_ctx}")

        auth_ctx = prior_auth_search.invoke({"query": query})
        if auth_ctx and "No prior authorization" not in auth_ctx:
            parts.append(f"[PRIOR AUTH]\n{auth_ctx}")

    elif intent == "COMPARISON":
        for tier in ("Bronze", "Silver", "Gold"):
            log.append(f"⚖️ Retrieving {tier} plan")

            tier_ctx = plan_comparison_search.invoke(
                {
                    "query": query,
                    "tier": tier,
                }
            )

            if tier_ctx and f"No {tier}" not in tier_ctx:
                parts.append(f"[{tier.upper()}]\n{tier_ctx}")

    separator = "\n\n" + "─" * 60 + "\n\n"

    full_context = (
        separator.join(parts)
        if parts
        else "No relevant context found."
    )

    for item in internal_logs:
        log.append(item)

    log.append(f"✅ Retrieved {len(parts)} context sections")

    trace_log.reset(token)

    return {
        **state,
        "retrieved_context": full_context,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# NODE 3 — RETRIEVAL GRADING
# ══════════════════════════════════════════════════════════════


def grade_retrieval(state: AgentState) -> AgentState:
    context = state.get("retrieved_context", "")

    log = list(state.get("steps_log", []))

    if (
        not context
        or context == "No relevant context found."
        or len(context) < 300
    ):
        grade = "WEAK"
        log.append("⚠️ Retrieval grading → WEAK")
    else:
        grade = "GOOD"
        log.append("✅ Retrieval grading → GOOD")

    return {
        **state,
        "retrieval_grade": grade,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# NODE 4 — SYNTHESIZE
# ══════════════════════════════════════════════════════════════


def synthesize(state: AgentState) -> AgentState:
    llm = _synthesis_llm()

    query = state["query"]
    log = list(state.get("steps_log", []))

    history_str = "\n".join(
        f"{role.upper()}: {msg}"
        for role, msg in (state.get("chat_history") or [])
    ) or "None"

    system_content = _SYNTHESIS_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=state["retrieved_context"],
        history=history_str,
    )

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ])

    answer = response.content

    if "Source:" not in answer:
        answer += (
            "\n\n⚠️ Warning: Limited citation coverage detected."
        )

    log.append("💬 Answer synthesized")

    return {
        **state,
        "answer": answer,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# NODE 5 — EVALUATE ANSWER
# ══════════════════════════════════════════════════════════════


def evaluate_answer(state: AgentState) -> AgentState:
    llm = _classifier_llm()

    answer = state.get("answer", "")
    context = state.get("retrieved_context", "")

    log = list(state.get("steps_log", []))

    evaluation_input = f"""
QUESTION:
{state['query']}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}
"""

    response = llm.invoke([
        SystemMessage(content=_EVALUATION_PROMPT),
        HumanMessage(content=evaluation_input),
    ])

    try:
        data = json.loads(response.content)

        grounded = bool(data.get("grounded", False))
        confidence = float(data.get("confidence_score", 0.0))

    except Exception:
        grounded = False
        confidence = 0.0

    if grounded:
        log.append(
            f"✅ Groundedness PASSED ({confidence:.2f})"
        )
    else:
        log.append(
            f"⚠️ Groundedness FAILED ({confidence:.2f})"
        )

    return {
        **state,
        "grounded": grounded,
        "confidence_score": confidence,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# RETRY ROUTER
# ══════════════════════════════════════════════════════════════


def should_retry(state: AgentState):
    grounded = state.get("grounded", False)
    retry_count = state.get("retry_count", 0)

    if grounded:
        return "finish"

    if retry_count >= 1:
        return "finish"

    return "retry"


# ══════════════════════════════════════════════════════════════
# RETRY NODE
# ══════════════════════════════════════════════════════════════


def retry_retrieval(state: AgentState) -> AgentState:
    log = list(state.get("steps_log", []))

    retry_count = state.get("retry_count", 0) + 1

    improved_query = (
        state["query"]
        + " detailed insurance coverage prior authorization exclusions"
    )

    log.append("🔁 Corrective retry triggered")
    log.append(f"🔁 Retry query → {improved_query}")

    return {
        **state,
        "query": improved_query,
        "retry_count": retry_count,
        "steps_log": log,
    }


# ══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_retrieval", grade_retrieval)
    builder.add_node("synthesize", synthesize)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("retry_retrieval", retry_retrieval)

    builder.set_entry_point("classify_intent")

    builder.add_edge("classify_intent", "retrieve")
    builder.add_edge("retrieve", "grade_retrieval")
    builder.add_edge("grade_retrieval", "synthesize")
    builder.add_edge("synthesize", "evaluate_answer")

    builder.add_conditional_edges(
        "evaluate_answer",
        should_retry,
        {
            "retry": "retry_retrieval",
            "finish": END,
        },
    )

    builder.add_edge("retry_retrieval", "retrieve")

    return builder.compile()


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR CLASS
# ══════════════════════════════════════════════════════════════




class Orchestrator:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found.")

        self.graph = build_graph()

        self.chat_history = []

        # Required for /trace and /graph endpoints
        self.last_detailed_result = {}

    # ───────────────────────────────────────────────────────────
    # SIMPLE ASK
    # ───────────────────────────────────────────────────────────

    def ask(self, query: str, verbose: bool = False):
        initial_state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_context": "",
            "answer": "",
            "chat_history": self.chat_history.copy(),
            "steps_log": [],
            "retrieval_grade": "",
            "grounded": False,
            "confidence_score": 0.0,
            "retry_count": 0,
        }

        result = self.graph.invoke(initial_state)

        if verbose:
            console.print("\n[bold dim]📋 Orchestrator trace:[/bold dim]")

            for step in result["steps_log"]:
                console.print(f"  [dim]{step}[/dim]")

        answer = result["answer"]

        self.chat_history.append(("human", query))
        self.chat_history.append(("ai", answer))

        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return answer

    # ───────────────────────────────────────────────────────────
    # DETAILED ASK
    # ───────────────────────────────────────────────────────────

    def ask_detailed(self, query: str) -> dict:
        """
        Returns full orchestrator result for:
        - API layer
        - developer console
        - graph view
        - trace endpoints
        """

        initial_state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_context": "",
            "answer": "",
            "chat_history": self.chat_history.copy(),
            "steps_log": [],
            "retrieval_grade": "",
            "grounded": False,
            "confidence_score": 0.0,
            "retry_count": 0,
        }

        result = self.graph.invoke(initial_state)

        answer = result["answer"]

        self.chat_history.append(("human", query))
        self.chat_history.append(("ai", answer))

        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # Store latest detailed result
        self.last_detailed_result = {
            "query": query,
            "answer": answer,
            "intent": result.get("intent", ""),
            "steps_log": result.get("steps_log", []),
            "retrieved_context": result.get(
                "retrieved_context",
                "",
            ),
            "retrieval_grade": result.get(
                "retrieval_grade",
                "",
            ),
            "grounded": result.get(
                "grounded",
                False,
            ),
            "confidence_score": result.get(
                "confidence_score",
                0.0,
            ),
            "retry_count": result.get(
                "retry_count",
                0,
            ),
        }

        return self.last_detailed_result

    # ───────────────────────────────────────────────────────────
    # STREAMING SUPPORT
    # ───────────────────────────────────────────────────────────

    def stream_detailed(self, query: str):
        """
        Streams intermediate graph execution states.
        Useful for live developer console.
        """

        initial_state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_context": "",
            "answer": "",
            "chat_history": self.chat_history.copy(),
            "steps_log": [],
            "retrieval_grade": "",
            "grounded": False,
            "confidence_score": 0.0,
            "retry_count": 0,
        }

        for event in self.graph.stream(initial_state):
            for node, state in event.items():
                yield {
                    "node": node,
                    "state": state,
                }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orch = Orchestrator()

    q = "What is the copay for Metformin on the Silver plan?"

    console.print(f"[bold]Q:[/bold] {q}\n")

    answer = orch.ask(q, verbose=True)

    console.print(answer)