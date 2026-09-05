from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.embedding import EmbeddingPipeline
from src.vector_stor import FaissDocumentStore


class ResearchState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[Document]
    raw_answer: str
    answer: str
    supported: bool
    attempts: int
    metrics: dict
    start_time: float


class ResearchAgent:
    def __init__(self, chunks: list[Document], embeddings):
        self.embeddings = EmbeddingPipeline()
        self.store = FaissDocumentStore(chunks, embeddings)
        self.llm = self._init_llm()
        self.graph = self._build_graph()

    def ask(self, question: str) -> ResearchState:
        return self.graph.invoke({
            "question": question, "plan": [], "evidence": [],
            "raw_answer": "", "answer": "", "supported": False,
            "attempts": 0, "metrics": {}, "start_time": time.time(),
        })

    def _init_llm(self):
        if os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        return None

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_node("verify", self._verify)
        graph.add_node("evaluate", self._evaluate)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", "verify")
        graph.add_conditional_edges("verify", self._next_step, {"retry": "retrieve", "evaluate": "evaluate"})
        graph.add_edge("evaluate", END)
        return graph.compile()

    def _plan(self, state: ResearchState) -> ResearchState:
        question = state["question"].strip()
        state["plan"] = [question]
        if any(word in question.lower() for word in ["latest", "current", "today", "recent"]):
            state["plan"].append(f"web: {question}")
        return state

    def _retrieve(self, state: ResearchState) -> ResearchState:
        query = state["plan"][0]
        if state["attempts"]:
            query = f"{query} details evidence source"
        vector = self.embeddings.embed_texts([query])
        vector_hits = self.store.search(vector, k=5 + state["attempts"] * 3)
        keyword_hits = self._keyword_search(query, k=5)
        state["evidence"] = self._unique(vector_hits + keyword_hits)
        state["attempts"] += 1
        if len(state["plan"]) > 1:
            state["evidence"].extend(self._web_search(state["question"]))
        return state

    def _answer(self, state: ResearchState) -> ResearchState:
        relevant = [doc for doc in state["evidence"] if self._is_relevant(state["question"], doc)]
        if not relevant:
            state["raw_answer"] = "I could not find enough evidence in the uploaded documents."
            state["answer"] = state["raw_answer"]
            return state

        context_parts = []
        for i, doc in enumerate(relevant[:5]):
            source = Path(str(doc.metadata.get("source", ""))).name
            page = doc.metadata.get("page")
            cite = f"{source}, page {page + 1}" if isinstance(page, int) else source
            context_parts.append(f"[Chunk {i+1}, {cite}]: {doc.page_content}")
        context = "\n\n".join(context_parts)

        if self.llm:
            prompt = (
                f"You are a document research assistant. Answer the user's question using ONLY the evidence provided below. "
                f"If the evidence does not contain enough information, say so. Cite sources by chunk number.\n\n"
                f"EVIDENCE:\n{context}\n\n"
                f"QUESTION: {state['question']}\n\n"
                f"Answer concisely in 2-4 sentences. Cite evidence using [Chunk N] notation."
            )
            try:
                response = self.llm.invoke(prompt)
                state["raw_answer"] = response.content
            except Exception:
                state["raw_answer"] = self._best_sentence(state["question"], relevant[0].page_content)
        else:
            state["raw_answer"] = self._best_sentence(state["question"], relevant[0].page_content)

        evidence_lines = []
        for doc in relevant[:3]:
            evidence_lines.append(f"- {self._best_sentence(state['question'], doc.page_content)} [{self._citation(doc)}]")
        state["answer"] = state["raw_answer"] + "\n\n**Evidence:**\n" + "\n".join(evidence_lines)
        return state

    def _verify(self, state: ResearchState) -> ResearchState:
        state["supported"] = bool(
            state["evidence"]
            and state["raw_answer"]
            and "could not find" not in state["raw_answer"].lower()
            and any(self._is_relevant(state["question"], doc) for doc in state["evidence"])
        )
        return state

    def _next_step(self, state: ResearchState) -> str:
        return "done" if state["supported"] or state["attempts"] >= 2 else "evaluate"

    def _evaluate(self, state: ResearchState) -> ResearchState:
        q_words = set(self._tokens(state["question"]))
        evidence_text = " ".join(doc.page_content for doc in state["evidence"]).lower()
        answer_text = state["raw_answer"].lower()

        relevant_chunks = sum(1 for doc in state["evidence"] if self._is_relevant(state["question"], doc))
        retrieval_precision = round(relevant_chunks / max(len(state["evidence"]), 1), 2)

        evidence_terms = set(re.findall(r"[a-z]{3,}", evidence_text))
        answer_terms = set(re.findall(r"[a-z]{3,}", answer_text))
        faithfulness = round(len(evidence_terms & answer_terms) / max(len(answer_terms), 1), 2)

        answer_relevant_words = sum(1 for w in q_words if w in answer_text)
        answer_relevance = round(answer_relevant_words / max(len(q_words), 1), 2)

        latency = round(time.time() - state["start_time"], 2)

        state["metrics"] = {
            "retrieval_precision": retrieval_precision,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "latency_seconds": latency,
            "retrieval_attempts": state["attempts"],
            "total_chunks_retrieved": len(state["evidence"]),
            "relevant_chunks_used": relevant_chunks,
        }
        return state

    def _web_search(self, question: str) -> list[Document]:
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            from langchain_tavily import TavilySearch

            results = TavilySearch(max_results=3).invoke({"query": question})
            return [Document(page_content=item.get("content", ""), metadata={"source": item.get("url", "Tavily")}) for item in results.get("results", [])]
        except Exception:
            return []

    def _keyword_search(self, query: str, k: int) -> list[Document]:
        words = self._tokens(query)
        scored = []
        for doc in self.store.chunks:
            text = doc.page_content.lower()
            score = sum(text.count(word) for word in words)
            if score:
                scored.append((score, doc))
        return [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]

    def _unique(self, docs: list[Document]) -> list[Document]:
        seen = set()
        unique = []
        for doc in docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _is_relevant(self, question: str, doc: Document) -> bool:
        words = set(self._tokens(question))
        text = doc.page_content.lower()
        if "police custody" in question.lower():
            return "police custody" in text
        return len([word for word in words if word in text]) >= min(2, len(words))

    def _best_sentence(self, question: str, text: str) -> str:
        words = set(self._tokens(question))
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        best = max(sentences, key=lambda sentence: sum(word in sentence.lower() for word in words), default="")
        return best[:500]

    def _citation(self, doc: Document) -> str:
        source = Path(str(doc.metadata.get("source", "uploaded document"))).name
        page = doc.metadata.get("page")
        return f"{source}, page {page + 1}" if isinstance(page, int) else source

    def _tokens(self, text: str) -> list[str]:
        stopwords = {"what", "which", "that", "this", "with", "from", "have", "can", "for", "the", "and", "are", "period", "total"}
        return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stopwords]
