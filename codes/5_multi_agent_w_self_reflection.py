from imports_for_all_systems import *

#############################################################################

# tools

def retrieval_tool(query):
    return "\n".join(retrieve(query, k=5))

#############################################################################
# SMART RETRIEVAL (DYNAMIC TOOL USE)

def smart_retrieval_agent(query):
    decision = llm(f"""
    Decide if retrieval is needed.

    Query: {query}

    Answer only YES or NO.
    """)

    if "YES" in decision.upper():
        return retrieval_tool(query)
    return ""

#############################################################################
# STATE

from typing import TypedDict

class DebateState(TypedDict):
    query: str
    context: str
    answer_a: str
    critique: str
    refined: str
    final: str
    confidence: float
    iterations: int

#############################################################################
# AGENTS

# 1. CONTEXT BUILDER (SMART RETRIEVAL)
def context_agent(state):
    context = smart_retrieval_agent(state["query"])
    return {"context": context}


# 2. PROPOSER
def proposer(state):
    prompt = f"""
    Answer the query using context:

    {state.get('context','')}

    Query: {state['query']}
    """

    return {"answer_a": llm(prompt)}


# 3. CRITIC
def critic(state):
    prompt = f"""
    Critically evaluate this answer:

    {state['answer_a']}

    Find:
    - errors
    - missing info
    - wrong assumptions
    """

    return {"critique": llm(prompt)}


# 4. REFLECTION (IMPROVEMENT)
def reflection(state):
    prompt = f"""
    Improve answer using critique:

    Original:
    {state['answer_a']}

    Critique:
    {state['critique']}

    Provide improved answer.
    """

    return {
        "refined": llm(prompt),
        "iterations": state.get("iterations", 0) + 1
    }


# 5. JUDGE (FINAL SELECTION)
def judge(state):
    prompt = f"""
    Compare answers:

    A: {state['answer_a']}
    B: {state['refined']}

    Return best answer only.
    """

    return {"final": llm(prompt)}


import json

# 6. CONFIDENCE SCORER
def confidence_agent(state):
    prompt = f"""
    Evaluate answer quality:

    {state['final']}

    Return JSON:
    {{
        "confidence": 0-1
    }}
    """

    try:
        result = json.loads(llm(prompt))
        return {"confidence": result["confidence"]}
    except:
        return {"confidence": 0.0}


#############################################################################
# CONTROL LOGIC (ITERATION LOOP)

def should_continue(state):
    # stop if enough iterations
    if state.get("iterations", 0) >= 2:
        return "judge"

    return "critic"


#############################################################################
# GRAPH

from langgraph.graph import StateGraph

builder = StateGraph(DebateState)

builder.add_node("context", context_agent)
builder.add_node("proposer", proposer)
builder.add_node("critic", critic)
builder.add_node("reflection", reflection)
builder.add_node("judge", judge)
builder.add_node("confidence", confidence_agent)

builder.set_entry_point("context")

builder.add_edge("context", "proposer")
builder.add_edge("proposer", "critic")

builder.add_conditional_edges("critic", should_continue)

builder.add_edge("reflection", "critic")   # iterative loop
builder.add_edge("judge", "confidence")

graph = builder.compile()

#############################################################################
# RUN

query = "If I have a pre-existing condition and surgery, what about waiting period AND reimbursement?"

result = graph.invoke({
    "query": query,
    "iterations": 0
})

print("\n FINAL ANSWER \n")
print(result["final"])

print("\n CONFIDENCE \n")
print(result["confidence"])

#############################################################################
# ERROR HANDLING

if "don't know" in result["final"].lower():
    print("\n Fallback to human review triggered")