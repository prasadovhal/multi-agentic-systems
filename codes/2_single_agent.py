from imports_for_all_systems import *

########################################################################

# Tools

def retrieval_tool(query):
    return "\n".join(retrieve(query, k=5))

################################################################################

## Single Agent

def single_agent(query):
    context = retrieval_tool(query)

    prompt = f"""
    You are an insurance expert.

    Steps:
    1. Identify relevant policies
    2. Combine rules
    3. Answer logically

    Context:
    {context}

    Question:
    {query}
    """

    return llm(prompt)

##########################################################################

# Run

print(single_agent(
    "If I have pre-existing disease and need surgery, what about waiting period and reimbursement?"
))


########################################################################

"""
❌ Limitation

Fails for:

“If surgery is excluded BUT approved under exception AND waiting period not completed, what happens to claim + reimbursement?”

👉 Needs:

exception handling
conflict resolution
multi-domain reasoning

"""


print(single_agent(
    "If surgery is excluded BUT approved under exception AND waiting period not completed, what happens to claim + reimbursement?"
))
