# Interview Notes

## 60-Second Pitch

I built an intelligent document research agent using a RAG pipeline and an agent workflow. Users upload documents in Streamlit, the app chunks and embeds the content using Sentence-Transformers, stores vectors in FAISS, and uses LangGraph to control the research process. The graph plans the query, retrieves evidence, drafts an answer, verifies that the answer is supported, and retries retrieval when needed. Answers include citations so the user can validate the source.

## Flow To Explain

```text
User question
  -> plan question
  -> embed query
  -> FAISS top-k search
  -> build cited answer from retrieved chunks
  -> verify evidence exists
  -> retry once if unsupported
```

## Key Code To Show

- `src/search.py`: the LangGraph nodes and routing.
- `src/embedding.py`: chunking and embedding.
- `src/vector_stor.py`: FAISS search.
- `app.py`: upload, index, ask, display answer.

## Questions You May Get

**Why not just ask an LLM directly?**

Because direct LLM answers can hallucinate. RAG grounds the answer in retrieved documents and gives citations.

**What does LangGraph add?**

It gives explicit state and routing. I can inspect each step: planning, retrieval, answer generation, verification, and retry.

**How do you detect unsupported responses?**

In this demo, the verifier checks whether retrieved evidence exists before finalizing. In a production version, I would use an LLM judge or entailment model to compare claims against evidence.

**How would you scale it?**

Persist FAISS or move to a vector database, cache embeddings, process uploads asynchronously, and add evaluation tests for answer quality.

**What would you improve next?**

Add an LLM synthesis step, better citation formatting, persistent indexes, and automated evaluation on sample questions.
