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

# Run

result = graph.invoke({
    "query": "If surgery is excluded but approved as exception and waiting period not completed, what happens to reimbursement?"
})

print(result["final"])