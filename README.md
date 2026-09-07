# Intelligent Document Research Agent

Small interview-ready RAG project built from the resume bullet:

**Python, LangGraph, LangChain, FAISS, Sentence-Transformers, uv, Tavily, Streamlit**

The app lets a user upload documents, builds a semantic vector index, and answers questions with cited evidence. The agent is stateful through LangGraph: it plans, retrieves, generates an LLM answer, verifies whether the answer has evidence, and retries retrieval once if evidence is weak. Evaluation metrics (retrieval precision, faithfulness, answer relevance, latency) are shown for every response.

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
2. Upload a PDF/TXT/CSV/XLSX file from `data/samples/` or your own file.
3. Click **Build research index**.
4. Ask: `What are the main topics in this document?`
5. Show the LLM answer, citations, agent trace, and evaluation metrics.

## Architecture

```text
Upload documents
  -> LangChain loaders
  -> Recursive text chunking (220 tokens, token-based splitting)
  -> Sentence-Transformers embeddings (all-MiniLM-L6-v2)
  -> FAISS vector index (IndexFlatIP)
  -> LangGraph research workflow
       plan -> retrieve -> answer (Groq Llama 3.1) -> verify -> retry or evaluate -> END
  -> Evaluation metrics (precision, faithfulness, relevance, latency)
  -> Streamlit answer with citations + metrics
```

## Files

- `app.py`: Streamlit UI with chat history, metrics display, and `.env` loading.
- `src/data_loader.py`: loads PDF, TXT, CSV, and XLSX files as LangChain `Document` objects.
- `src/embedding.py`: chunks documents and creates local embeddings.
- `src/vector_stor.py`: FAISS similarity-search wrapper.
- `src/search.py`: LangGraph research agent with LLM answer generation and evaluation.
- `main.py`: CLI smoke demo using files in `data/samples/`.

## Evaluation Metrics

Every answer includes four evaluation metrics:

| Metric | What it measures | How it's computed |
|---|---|---|
| **Retrieval Precision** | % of retrieved chunks that are relevant to the question | `relevant_chunks / total_chunks_retrieved` |
| **Faithfulness** | % of answer terms grounded in the retrieved evidence | `answer_terms_in_evidence / total_answer_terms` |
| **Answer Relevance** | % of question keywords present in the answer | `question_keywords_in_answer / total_question_keywords` |
| **Latency** | Total wall-clock time for the full pipeline | `time_end - time_start` |

Additional stats shown: total chunks retrieved, relevant chunks used, retrieval attempts.

## Good Interview Points

- **Why FAISS?** Fast local vector similarity search without needing a hosted vector database. Simpler than ChromaDB for a demo — zero setup, runs in RAM.
- **Why Sentence-Transformers?** Local embeddings make the demo cheaper and runnable without API keys.
- **Why LangGraph?** It makes the agent flow explicit and debuggable instead of hiding control flow inside one chain. The loop (verify → retry) is easy to add and inspect.
- **Why verification?** It reduces hallucination by checking whether the answer has retrieved evidence and retrying retrieval when needed.
- **Why chunking?** Large documents do not fit cleanly into one prompt, so chunking improves retrieval precision.

## Honest Limitations

- Faithfulness is measured by term overlap, not full entailment. A production system would use an LLM judge or NLI model.
- The FAISS index is in-memory and rebuilt on every app restart.
- No concurrent user support or document access control.
