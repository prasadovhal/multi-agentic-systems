# multi-agentic-systems
Understanding multi-agentic systems, and evolution from RAG


# Phase 1: Basic RAG (Knowledge Retrieval)
- The Problem: Answering questions based on a specific, private dataset that the LLM wasn't trained on.

- The Data: Five distinct documents (100+ words each) regarding a specific fictional company’s HR policies.


## Step-by-Step Implementation:

- Data Ingestion: Load the five text documents.
- Chunking: Split text into manageable pieces using RecursiveCharacterTextSplitter.
- Embedding: Convert text chunks into vectors using OpenAIEmbeddings.
- Vector Store: Store and index vectors in FAISS.
- Retrieval & Generation: Create a chain that fetches relevant context and passes it to the LLM to answer a user query.

# Phase 2: Single-Agentic System (Reasoning & Tool Use)
- The Complexity Increase: The user asks a question that requires action or multi-step logic, such as: "Based on the policy, calculate my remaining PTO and send a summary email to my manager."
- Why RAG Fails: Standard RAG can only "read." It cannot perform math reliably, access external APIs, or decide on a sequence of actions.

## Step-by-Step Implementation:

- Tool Definition: Wrap the RAG system from Phase 1 as a "Retriever Tool."
- Additional Tools: Create a "Calculator Tool" and an "Email API Tool."
- Agent Initialization: Define a "Reasoning Engine" (the LLM) that has access to these tools.
- The Loop: The agent receives the prompt, decides it needs the RAG tool first, then the Calculator, then the Email tool.
- Execution: The agent executes the plan and returns a final confirmation.

# Phase 3: Multi-Agentic System (Specialized Roles & Collaboration)
- The Complexity Increase: The task becomes a high-level project: "Audit our entire 5-document policy for legal compliance, rewrite the sections that conflict with New York state law, and generate a formatted PDF report."
- Why Single Agents Fail: A single agent gets "confused" or suffers from context window drift when handling long, multi-faceted tasks (Legal Auditing vs. Creative Writing vs. File Formatting).

## Step-by-Step Implementation:

- Define Roles: Create three distinct agents:
- The Researcher: Uses RAG to extract existing policies.
- The Legal Analyst: Compares extracted text against a "Legal Database" tool.
- The Editor: Synthesizes findings into a professional report format.
- State Management: Use a shared "State" or "Blackboard" where agents can pass data to one another.
- Workflow Mapping (The Graph): Define the edges—once the Researcher is done, the Legal Analyst starts; if the Analyst finds an error, it loops back to the Researcher.
- Supervision: Implement a "Manager Agent" to review the final output before delivery.