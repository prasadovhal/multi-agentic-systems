import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

############################################################################
## Load data

from langchain_community.document_loaders import TextLoader, DirectoryLoader

DATA_PATH = "D:/Study/Git_repo/multi-agentic-systems/codes/docs/"

loader = DirectoryLoader(
    DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

print(len(docs))

###########################################################################
## chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500)

chunks = splitter.split_documents(docs)

##########################################################################
## embeddings + Vector DB store

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c.page_content for c in chunks]
embeddings = embed_model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

##########################################################################
## Retrieval

def retrieve(query, k=3):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [texts[i] for i in indices[0]]

###########################################################################

## LLM

import ollama

def llm(prompt: str):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
