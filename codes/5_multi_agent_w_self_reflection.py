from imports_for_all_systems import *

########################################################################

# Tools
def retrieval_tool(query):
    return "\n".join(retrieve(query, k=5))


############################################################## 

# Agents

## Agent A (Proposer)
def proposer(state):
    context = retrieval_tool(state["query"])

    prompt = f"""
    Answer the query using context:

    {context}

    Query: {state['query']}
    """

    return {"answer_a": llm(prompt)}


## Agent B (Critic)
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


## Reflection Agent

def reflection(state):
    prompt = f"""
    Improve answer using critique:

    Original:
    {state['answer_a']}

    Critique:
    {state['critique']}

    Provide improved answer.
    """

    return {"refined": llm(prompt)}


## Judge Agent

def judge(state):
    prompt = f"""
    Compare answers:

    A: {state['answer_a']}
    B: {state['refined']}

    Which is better and why?

    Return final answer.
    """

    return {"final": llm(prompt)}

####################################################

from typing import TypedDict

class DebateState(TypedDict):
    query: str
    answer_a: str
    critique: str
    refined: str
    final: str


######################################################

# langgraph

from langgraph.graph import StateGraph

builder = StateGraph(DebateState)

builder.add_node("proposer", proposer)
builder.add_node("critic", critic)
builder.add_node("reflection", reflection)
builder.add_node("judge", judge)

builder.set_entry_point("proposer")

builder.add_edge("proposer", "critic")
builder.add_edge("critic", "reflection")
builder.add_edge("reflection", "judge")

graph = builder.compile()


#########################################################

# run
result = graph.invoke({
    "query": "Is surgery covered under insurance with pre-existing condition?"
})

print(result["final"])

#################################################

# Add Tool Usage (Dynamic)

def smart_retrieval_agent(query):
    prompt = f"""
    Decide if retrieval is needed for:
    {query}

    If yes → say RETRIEVE
    else → answer directly
    """

    decision = llm(prompt)

    if "RETRIEVE" in decision:
        return retrieval_tool(query)
    return ""

# Add Iterative Debate (Advanced)

for _ in range(2):
    critique = critic(state)
    state.update(critique)

    refined = reflection(state)
    state.update(refined)


# Add Confidence Scoring

def judge_with_score(state):
    prompt = f"""
    Evaluate final answer:

    {state['refined']}

    Return:
    {{
        "answer": "...",
        "confidence": 0-1
    }}
    """
    return llm(prompt)


# add error handling

if "don't know" in result["final"].lower():
    return "Fallback to human review"