from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer


class VectorMemory:
    """Wraps the conversation-turn and facts Chroma collections that used to
    live as module-level globals in agent.py. Instantiate once (see main.py)
    and reuse across the whole session.
    """

    def __init__(self, path: str = "./hermes_memory"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma = chromadb.PersistentClient(path=path)
        self.conversations = self.chroma.get_or_create_collection("conversations")
        self.facts = self.chroma.get_or_create_collection("facts")

    def remember(self, text, role):
        if text is None:
            text = ""
        emb = self.embedder.encode(text).tolist()
        self.conversations.add(
            ids=[f"{datetime.now().isoformat()}-{role}"],
            embeddings=[emb], documents=[text],
            metadatas=[{"role": role, "ts": datetime.now().isoformat()}]
        )

    def store_fact(self, text: str):
        if text is None:
            return
        if self.facts.count() > 0:
            emb = self.embedder.encode(text).tolist()
            results = self.facts.query(
                query_embeddings=[emb],
                n_results=3,
                include=["documents", "distances"]
            )
            if results.get("distances") and results["distances"][0]:
                for doc, dist in zip(results["documents"][0], results["distances"][0]):
                    if dist < 0.15:
                        print(f"  [fact duplicate skipped (dist={dist:.3f}): {text}]")
                        return
                    text_norm = text.lower().replace("user", "owner").replace("i ", "owner ")
                    doc_norm = doc.lower().replace("user", "owner").replace("i ", "owner ")
                    if text_norm == doc_norm:
                        print(f"  [fact duplicate skipped (text match): {text}]")
                        return

        emb = self.embedder.encode(text).tolist()
        self.facts.add(
            ids=[f"{datetime.now().isoformat()}-fact"],
            embeddings=[emb],
            documents=[text],
            metadatas=[{
                "type": "fact",
                "created": datetime.now().isoformat(),
                "source": "conversation"
            }]
        )
        print(f"  [fact stored: {text}]")

    def cap_conversations(self, max_turns=20):
        count = self.conversations.count()
        if count > max_turns:
            all_data = self.conversations.get()
            ids = all_data["ids"]
            to_delete = ids[:-max_turns]
            self.conversations.delete(ids=to_delete)
            print(f"  [pruned {len(to_delete)} old conversation turns]")

    def recall(self, query, k=3):
        if self.facts.count() == 0 and self.conversations.count() == 0:
            return []

        emb = self.embedder.encode(query).tolist()
        seen = set()
        results = []

        def add_unique(docs):
            for doc in docs:
                key = doc.lower().strip(" .!?,;:")
                if key not in seen and len(key) > 3:
                    seen.add(key)
                    results.append(doc)

        if self.facts.count() > 0:
            fact_results = self.facts.query(
                query_embeddings=[emb],
                n_results=min(k, self.facts.count())
            )
            if fact_results.get("documents") and fact_results["documents"][0]:
                add_unique(fact_results["documents"][0])

        if self.conversations.count() > 0:
            conv_results = self.conversations.query(
                query_embeddings=[emb],
                n_results=min(k, self.conversations.count())
            )
            if conv_results.get("documents") and conv_results["documents"][0]:
                add_unique(conv_results["documents"][0])

        return results[:k]
