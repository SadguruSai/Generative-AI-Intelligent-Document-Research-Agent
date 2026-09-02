# Intelligent Document Research Agent

Small interview-ready RAG project built from the resume bullet:

**Python, LangGraph, LangChain, FAISS, Sentence-Transformers, uv, Tavily, Streamlit**

The app lets a user upload documents, builds a semantic vector index, and answers questions with cited evidence. The agent is stateful through LangGraph: it plans, retrieves, drafts an answer, verifies whether the answer has evidence, and retries retrieval once if evidence is weak.

## Run

```powershell
uv sync
uv run streamlit run app.py
```

Optional web search:

```powershell
$env:TAVILY_API_KEY="your_key"
uv run streamlit run app.py
```

## What To Demo Tomorrow

1. Open the app.
2. Upload a PDF/TXT/CSV/XLSX file from `data/` or your own file.
3. Click **Build research index**.
4. Ask: `What are the main topics in this document?`
5. Show the answer, citations, and agent trace.

## Architecture

```text
Upload documents
  -> LangChain loaders
  -> Recursive text chunking
  -> Sentence-Transformers embeddings
  -> FAISS vector index
  -> LangGraph research workflow
       plan -> retrieve -> answer -> verify -> retry or finish
  -> Streamlit answer with citations
```

## Files

- `app.py`: Streamlit UI.
- `src/data_loader.py`: loads PDF, TXT, CSV, and XLSX files as LangChain `Document` objects.
- `src/embedding.py`: chunks documents and creates local embeddings.
- `src/vector_stor.py`: small FAISS similarity-search wrapper.
- `src/search.py`: LangGraph research agent.
- `main.py`: CLI smoke demo using files in `data/temp`.

## Interview Explanation Script

"I built an agentic RAG assistant for document research. The user uploads documents in Streamlit. I load them with LangChain document loaders, split them into overlapping chunks, embed those chunks using a local Sentence-Transformers model, and store the vectors in FAISS for fast similarity search.

The agent workflow is controlled by LangGraph. The graph has four steps: plan the query, retrieve relevant chunks, produce a source-grounded answer, and verify that evidence exists. If the answer is unsupported, the graph routes back to retrieval with a broader query before finalizing. Tavily is optional for current web information when an API key is available.

The important part is that the answer is not free-form guessing. It is built from retrieved evidence and shows citations, so the user can trace every answer back to source documents."

## Good Interview Points

- **Why FAISS?** Fast local vector similarity search without needing a hosted vector database.
- **Why Sentence-Transformers?** Local embeddings make the demo cheaper and runnable without API keys.
- **Why LangGraph?** It makes the agent flow explicit and debuggable instead of hiding control flow inside one chain.
- **Why verification?** It reduces hallucination by checking whether the answer has retrieved evidence and retrying retrieval when needed.
- **Why chunking?** Large documents do not fit cleanly into one prompt, so chunking improves retrieval precision.

## Honest Limitations

- The answer step is extractive and simple so the demo works without an LLM key.
- For production, I would connect an LLM for fluent synthesis, persist the FAISS index, add user auth, and improve evaluation with golden-question tests.
