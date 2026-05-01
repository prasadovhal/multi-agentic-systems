from imports_for_all_systems import *

########################################################################
## RAG

query = "Is surgery covered and what is reimbursement?"

context = "\n".join(retrieve(query))

response = llm(f"""
Answer based only on context:
{context}

Question: {query}
""")

print(response)

########################################################################

"""

❌ RAG Limitation

Fails for:

“If I have a pre-existing condition and surgery, what about waiting period AND reimbursement?”

👉 Needs reasoning across:

coverage
waiting
reimbursement

"""