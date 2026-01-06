from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections.abc import Callable
import numpy as np
import faiss
import toml


class IndexManager:
    """FAISSインデックスの作成・保存・検索を管理するクラス"""

    @staticmethod
    def create_reference_text(reference: dict) -> str:
        parts = [
            f"タイトル: {reference.get('title', '')}",
            f"理解: {reference.get('understanding', '')}",
        ]
        if reference.get('year'):
            parts.insert(1, f"出版年: {reference['year']}年")
        return "\n".join(parts)

    def _build(self, index_file):
        texts = [IndexManager.create_reference_text(ref) for ref in self.references]
        embeddings = self.embedder.embed_batch(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        faiss.write_index(self.index, str(index_file))
        print(f"Built index: {self.index.ntotal} vectors, {self.index.d} dimensions")

    def __init__(self, embedder, references_file, index_file, force_rebuild=False):
        self.embedder = embedder
        references_file_path = Path(references_file)
        index_file_path = Path(index_file)
        self.references = toml.loads(
            references_file_path.read_text(encoding="utf-8"),
        ).get("references", [])
        self.index = None
        if (
            (not force_rebuild) and index_file_path.is_file()
            and references_file_path.stat().st_mtime <= index_file_path.stat().st_mtime
        ):
            self.index = faiss.read_index(str(index_file))
        else:
            self._build(index_file)

    def search_impl(self, query_embedding, top_k):
        return self.index.search(query_embedding.astype(np.float32).reshape(1, -1), top_k)

    def search(self, query, top_k):
        query_embedding = self.embedder.embed(query)
        distances, indices = self.search_impl(query_embedding, min(top_k, len(self.references)))
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.references):
                reference = self.references[idx].copy()
                reference["_similarity_score"] = float(1 / (1 + distance))
                results.append(reference)
        return results
