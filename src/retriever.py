"""
Retriever: Retrieval-Augmented Generation (RAG) component.

Indexes custom knowledge documents (artist bios, genre guides, listening
context guides) and retrieves the most relevant passages for a given query.
Uses a from-scratch TF-IDF implementation with no external dependencies.

The retrieved context is injected into the explainer's prompt so the AI can
reference specific artist background, genre characteristics, and situational
advice — producing richer, more informed recommendations.
"""

import os
import math
import re
from collections import Counter
from src.models import UserProfile
from src.logger import AgentLogger


# ── Document representation ─────────────────────────────────────────────

class Document:
    """A single knowledge document with metadata."""

    def __init__(self, doc_type: str, name: str, content: str, source_file: str):
        self.doc_type = doc_type  # "artist", "genre", or "context"
        self.name = name          # e.g. "The Neons", "pop", "workout"
        self.content = content    # full text
        self.source_file = source_file
        self.tokens = _tokenize(content)
        self.tf = _term_frequency(self.tokens)

    def __repr__(self):
        return f"Document({self.doc_type}: {self.name})"


# ── Tokenization and TF-IDF ────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "but", "or", "nor",
    "not", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "about",
    "above", "after", "again", "also", "as", "at", "because", "before",
    "below", "between", "by", "down", "during", "for", "from", "if",
    "in", "into", "it", "its", "of", "off", "on", "once", "out",
    "over", "then", "there", "these", "this", "those", "through", "to",
    "under", "until", "up", "when", "where", "which", "while", "who",
    "whom", "why", "with", "that", "their", "them", "they", "what",
}


def _tokenize(text: str) -> list:
    """Lowercase, split on non-alpha, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _term_frequency(tokens: list) -> dict:
    """Compute normalized term frequency for a token list."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


def _idf(term: str, documents: list) -> float:
    """Compute inverse document frequency for a term across all documents."""
    doc_count = sum(1 for doc in documents if term in doc.tf)
    if doc_count == 0:
        return 0.0
    return math.log(len(documents) / doc_count) + 1.0


# ── Knowledge base loader ───────────────────────────────────────────────

def _parse_document_file(filepath: str) -> list:
    """Parse a knowledge file into Document objects.

    Supports two formats:
      - ARTIST/BIO blocks (artists.txt)
      - GENRE/GUIDE or CONTEXT/GUIDE blocks (genres.txt, contexts.txt)
    """
    docs = []
    source = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Determine document type from filename
    if "artist" in source.lower():
        doc_type = "artist"
        header_key = "ARTIST:"
        body_key = "BIO:"
    elif "genre" in source.lower():
        doc_type = "genre"
        header_key = "GENRE:"
        body_key = "GUIDE:"
    elif "context" in source.lower():
        doc_type = "context"
        header_key = "CONTEXT:"
        body_key = "GUIDE:"
    else:
        doc_type = "unknown"
        header_key = None
        body_key = None

    if header_key and body_key:
        # Split into blocks by the header key
        blocks = content.split(header_key)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Extract name (first line) and body (after body_key)
            lines = block.strip().split("\n")
            name = lines[0].strip()
            body = ""
            for line in lines:
                if line.strip().startswith(body_key):
                    body = line.strip()[len(body_key):].strip()
                    break
            if body:
                full_text = f"{name}: {body}"
                docs.append(Document(doc_type, name, full_text, source))
    else:
        # Fallback: treat entire file as one document
        docs.append(Document(doc_type, source, content, source))

    return docs


class KnowledgeBase:
    """
    In-memory knowledge base with TF-IDF retrieval.

    Loads documents from the data/knowledge/ directory and provides
    relevance-ranked search over them.
    """

    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(
                os.path.dirname(__file__), "..", "data", "knowledge"
            )
        self.knowledge_dir = os.path.abspath(knowledge_dir)
        self.documents = []
        self._load()

    def _load(self):
        """Load all .txt files from the knowledge directory."""
        if not os.path.isdir(self.knowledge_dir):
            return
        for filename in sorted(os.listdir(self.knowledge_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_dir, filename)
                self.documents.extend(_parse_document_file(filepath))

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Search the knowledge base using TF-IDF cosine similarity.

        Returns a list of (document, score) tuples, sorted by relevance.
        """
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = _term_frequency(query_tokens)

        scores = []
        for doc in self.documents:
            score = 0.0
            for term, qtf in query_tf.items():
                if term in doc.tf:
                    idf = _idf(term, self.documents)
                    score += qtf * idf * doc.tf[term] * idf
            scores.append((doc, round(score, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, s) for doc, s in scores[:top_k] if s > 0]

    def retrieve_for_profile(
        self,
        user_input: str,
        profile: UserProfile,
        recommended_artists: list,
        logger: AgentLogger,
        top_k: int = 5,
    ) -> list:
        """
        Smart retrieval: combines the user's query, parsed profile, and
        recommended artist names to find the most relevant knowledge.

        This is the main RAG entry point used by the agent.
        """
        # Build a rich query combining all available signals
        query_parts = [user_input]
        query_parts.append(profile.favorite_genre)
        query_parts.append(profile.favorite_mood)

        # Add artist names from recommendations
        for artist in recommended_artists:
            query_parts.append(artist)

        combined_query = " ".join(query_parts)

        results = self.search(combined_query, top_k=top_k)

        logger.log_step("RETRIEVER", f"Retrieved {len(results)} documents", {
            "query_terms": _tokenize(combined_query)[:10],
            "results": [(str(doc), score) for doc, score in results],
        })

        return results

    def format_context(self, results: list) -> str:
        """Format retrieved documents into a context string for the explainer."""
        if not results:
            return ""

        sections = []
        for doc, score in results:
            sections.append(f"[{doc.doc_type.upper()}: {doc.name}] {doc.content}")

        return "\n\n".join(sections)

    def stats(self) -> dict:
        """Return statistics about the knowledge base."""
        type_counts = Counter(doc.doc_type for doc in self.documents)
        return {
            "total_documents": len(self.documents),
            "by_type": dict(type_counts),
            "unique_terms": len(set(
                term for doc in self.documents for term in doc.tf
            )),
        }
