"""

- Hierarchical supervisor + dynamic task allocation
- Peer-to-peer agent calls
- Conflict resolution (scoring + voting + arbitration)
- Tree of Thoughts (multi-path + pruning)
- Replanning mid-execution
- Memory system (episodic + semantic + procedural, versioning, scoring, pruning, persistence)
- Observability (trace logs + replay)
- Evaluation (benchmark + regression testing)

"""

##########################################################
# IMPORTS & SETUP

import os, json, time, uuid, pickle
import numpy as np
import faiss
import ollama
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer

##########################################################
# LLM

def llm(prompt: str):
    return ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]

##########################################################
# MEMORY SYSTEM (SEMANTIC + EPISODIC + PROCEDURAL)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class MemoryStore:
    def __init__(self, path="memory.pkl"):
        self.path = path
        self.semantic_texts = []
        self.semantic_vectors = []
        self.episodic = []
        self.procedural = []
        self.version = 0

        self.index = faiss.IndexFlatL2(384)

        if os.path.exists(path):
            self.load()

    def add_semantic(self, text, score=1.0):
        emb = embed_model.encode([text])
        self.index.add(np.array(emb))
        self.semantic_texts.append((text, score, self.version))

    def retrieve(self, query, k=3):
        if len(self.semantic_texts) == 0:
            return []
        q_emb = embed_model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)
        return [self.semantic_texts[i][0] for i in I[0]]

    def add_episode(self, record):
        self.episodic.append(record)

    def add_procedure(self, rule):
        self.procedural.append(rule)

    def prune(self):
        # remove low score memories
        self.semantic_texts = [m for m in self.semantic_texts if m[1] > 0.3]

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        with open(self.path, "rb") as f:
            self.__dict__.update(pickle.load(f))

memory = MemoryStore()

##########################################################
# TRACE SYSTEM

TRACE_LOG = []

def trace(agent, input_data, output_data):
    TRACE_LOG.append({
        "id": str(uuid.uuid4()),
        "agent": agent,
        "input": input_data,
        "output": output_data,
        "time": time.time()
    })

def replay_trace():
    for step in TRACE_LOG:
        print(step["agent"], "→", step["output"])

##########################################################
# TOOLS

def retrieve_tool(query):
    return "\n".join(memory.retrieve(query))

##########################################################
# STATE

class State(TypedDict):
    query: str
    thoughts: List[str]
    candidates: List[str]
    final: str
    iteration: int

##########################################################
# AGENTS

# SUPERVISOR (Dynamic Task Allocation)
def supervisor(state):
    decision = llm(f"""
    Decide tasks for query:
    {state['query']}

    Options:
    - RETRIEVE
    - REASON
    - FINAL
    """)
    trace("supervisor", state["query"], decision)
    return decision


# PEER AGENTS

def retriever_agent(state):
    context = retrieve_tool(state["query"])
    trace("retriever", state["query"], context)
    return context

def reasoning_agent(state, context):
    thought = llm(f"""
    Think step by step:

    Context:
    {context}

    Query:
    {state['query']}
    """)
    trace("reasoning", context, thought)
    return thought


##########################################################
# TREE OF THOUGHTS

def generate_candidates(state, context):
    outputs = []
    for _ in range(3):  # multi-path
        outputs.append(llm(f"""
        Provide reasoning path:

        {context}

        Query: {state['query']}
        """))
    return outputs


def score_candidates(candidates):
    scored = []
    for c in candidates:
        score = llm(f"Score this reasoning 0-1:\n{c}")
        try:
            score = float(score.strip())
        except:
            score = 0.5
        scored.append((c, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def prune_candidates(scored):
    return [c for c, s in scored[:2]]  # keep top 2


##########################################################
# CONFLICT RESOLUTION

def vote(candidates):
    votes = []
    for c in candidates:
        v = llm(f"Is this correct? yes/no\n{c}")
        votes.append(1 if "yes" in v.lower() else 0)
    return candidates[votes.index(max(votes))]


def arbitration(candidates):
    return llm(f"""
    Choose best answer:
    {candidates}
    """)


##########################################################
# MAIN EXECUTION LOOP (REPLANNING)

def run_agent(query):

    state = {
        "query": query,
        "thoughts": [],
        "candidates": [],
        "final": "",
        "iteration": 0
    }

    while state["iteration"] < 3:

        decision = supervisor(state)

        if "RETRIEVE" in decision:
            context = retriever_agent(state)

        else:
            context = ""

        # Tree of Thoughts
        candidates = generate_candidates(state, context)

        # Scoring + pruning
        scored = score_candidates(candidates)
        pruned = prune_candidates(scored)

        # Conflict resolution
        voted = vote(pruned)
        final = arbitration(pruned)

        state["candidates"] = pruned
        state["final"] = final

        # Memory update
        memory.add_semantic(final, score=0.8)
        memory.add_episode({"query": query, "answer": final})

        # Replanning condition
        if "insufficient" in final.lower():
            state["iteration"] += 1
            continue

        break

    memory.prune()
    memory.save()

    return state


##########################################################
# EVALUATION (BENCHMARK + REGRESSION)

benchmark = [
    {"q": "Is surgery covered?", "expected": "depends"},
    {"q": "Waiting period?", "expected": "2-4 years"}
]

def evaluate():
    scores = []
    for item in benchmark:
        result = run_agent(item["q"])
        score = llm(f"""
        Compare:
        Expected: {item['expected']}
        Got: {result['final']}

        Score 0-1
        """)
        try:
            score = float(score)
        except:
            score = 0.5
        scores.append(score)

    print("Avg Score:", sum(scores)/len(scores))


##########################################################
# RUN

if __name__ == "__main__":
    result = run_agent("If I have pre-existing condition and surgery, what happens?")
    print("\nFINAL:", result["final"])

    print("\n--- TRACE ---")
    replay_trace()

    print("\n--- EVALUATION ---")
    evaluate()