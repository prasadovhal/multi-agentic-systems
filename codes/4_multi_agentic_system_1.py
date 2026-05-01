from imports_for_all_systems import *

########################################################################

# Tools

def retrieval_tool(query):
    return "\n".join(retrieve(query, k=5))

#########################################
## state

from typing import TypedDict

class State(TypedDict):
    query: str
    plan: str
    coverage: str
    waiting: str
    finance: str
    final: str

## planner

def planner(state):
    plan = llm(f"Break into sub-problems:\n{state['query']}")
    return {"plan": plan}

## 3.3 Specialized Agents

### coverage agent

def coverage_agent(state):
    context = retrieval_tool("surgery coverage insurance")
    res = llm(f"Extract coverage rules:\n{context}")
    return {"coverage": res}

### waiting agent

def waiting_agent(state):
    context = retrieval_tool("waiting period pre-existing")
    res = llm(f"Explain waiting period:\n{context}")
    return {"waiting": res}

### finance agent

def finance_agent(state):
    context = retrieval_tool("reimbursement insurance claim")
    res = llm(f"Explain reimbursement:\n{context}")
    return {"finance": res}

########################################################
## Decision Agent (Conflict Resolution)

def decision_agent(state):
    prompt = f"""
    Resolve conflicts:

    Coverage: {state['coverage']}
    Waiting: {state['waiting']}
    Finance: {state['finance']}

    Give final decision.
    """

    return {"final": llm(prompt)}

#####################################################################

# langraph

from langgraph.graph import StateGraph

builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("coverage", coverage_agent)
builder.add_node("waiting", waiting_agent)
builder.add_node("finance", finance_agent)
builder.add_node("decision", decision_agent)

builder.set_entry_point("planner")

builder.add_edge("planner", "coverage")
builder.add_edge("planner", "waiting")
builder.add_edge("planner", "finance")

builder.add_edge("coverage", "decision")
builder.add_edge("waiting", "decision")
builder.add_edge("finance", "decision")

graph = builder.compile()

#############################################

# Structured Output (Guardrails)
from pydantic import BaseModel

class FinalAnswer(BaseModel):
    answer: str
    confidence: float
    reasoning: str


# Enforce in LLM

import json

def structured_llm(prompt):
    response = llm(prompt)

    try:
        return FinalAnswer(**json.loads(response))
    except:
        return FinalAnswer(
            answer="Invalid output",
            confidence=0.0,
            reasoning="Parsing failed"
        )
    
##############################################################
## LLM-as-Judge (Evaluation Core)

def judge_answer(query, answer, context):
    prompt = f"""
    Evaluate answer quality:

    Query: {query}
    Answer: {answer}
    Context: {context}

    Score (0-1) for:
    - correctness
    - completeness
    - groundedness

    Return JSON:
    {{
        "score": float,
        "reason": "..."
    }}
    """
    return llm(prompt)

##############################################################

# Hallucination Detection

def hallucination_check(answer, context):
    prompt = f"""
    Check if answer is grounded in context.

    Context:
    {context}

    Answer:
    {answer}

    Return:
    {{
        "hallucination": true/false,
        "reason": "..."
    }}
    """
    return llm(prompt)

# metrics tracking

metrics = {
    "total": 0,
    "passed": 0,
    "avg_score": 0
}

def update_metrics(score, passed):
    metrics["total"] += 1
    metrics["passed"] += int(passed)
    metrics["avg_score"] = (
        (metrics["avg_score"] * (metrics["total"] - 1) + score)
        / metrics["total"]
    )

###############################################################

# Observability (Tracing)

import time

def trace(agent_name, func, state):
    start = time.time()

    result = func(state)

    print({
        "agent": agent_name,
        "input": state,
        "output": result,
        "latency": time.time() - start
    })

    return result

#########################################################

# Evaluation Pipeline

def evaluate_pipeline(query, result, context):
    judge = judge_answer(query, result, context)
    hallucination = hallucination_check(result, context)

    import json
    judge = json.loads(judge)
    hallucination = json.loads(hallucination)

    passed = (
        judge["score"] > 0.7 and
        hallucination["hallucination"] == False
    )

    update_metrics(judge["score"], passed)

    return {
        "passed": passed,
        "score": judge["score"],
        "hallucination": hallucination
    }

#########################################################

# CI/CD Gating Logic

def production_run(query):
    result = graph.invoke({"query": query})
    final_answer = result["final"]

    context = retrieval_tool(query)

    eval_result = evaluate_pipeline(
        query,
        final_answer,
        context
    )

    if not eval_result["passed"]:
        return {
            "status": "FAILED",
            "reason": eval_result
        }

    return {
        "status": "SUCCESS",
        "answer": final_answer
    }

##########################################################

# Batch Evaluation (Offline Testing)

test_set = [
    "Is surgery covered?",
    "Waiting period for pre-existing?",
    "Reimbursement rules?"
]

for q in test_set:
    print(production_run(q))

##############################################################

# Drift Detection (Simple Version)

def detect_drift():
    if metrics["avg_score"] < 0.6:
        print("Model drift detected!")
