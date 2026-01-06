import importlib.resources
from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections.abc import Callable
import numpy as np
import faiss
import toml


class IndexManager:
    """FAISSインデックスの作成・保存・検索を管理するクラス"""

    def get_text(self, reference_id, understanding_id):
        return self.references[reference_id]["understandings"][understanding_id]

    def _build(self, index_file):
        texts = [self.get_text(*u) for u in self.understandings]
        embeddings = self.embedder.embed_batch(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        faiss.write_index(self.index, str(index_file))
        print(f"Built index: {self.index.ntotal} vectors, {self.index.d} dimensions")

    def __init__(self, embedder, references_file, index_file, force_rebuild=False):
        self.embedder = embedder
        references_file_path = Path(references_file)
        if not references_file_path.is_file():
            print(f"Not found: {references_file_path}")
            print("Using references.sample.toml instead")
            references_file_path = importlib.resources.files('kazusa') / "references.sample.toml"
        index_file_path = Path(index_file)
        self.references = toml.loads(
            references_file_path.read_text(encoding="utf-8"),
        )["references"]
        self.understandings = [
            (i_ref, i_u)
            for i_ref, ref in enumerate(self.references)
            for i_u, _ in enumerate(ref["understandings"])
        ]
        self.index = None
        if (
            (not force_rebuild) and index_file_path.is_file()
            and references_file_path.stat().st_mtime <= index_file_path.stat().st_mtime
        ):
            print(f"Read index file: {index_file_path}")
            self.index = faiss.read_index(str(index_file))
            print(f"Read index: {self.index.ntotal} vectors, {self.index.d} dimensions")
        else:
            self._build(index_file)

    def search_impl(self, query_embedding, top_k):
        return self.index.search(query_embedding.astype(np.float32).reshape(1, -1), top_k)

    def search(self, query, top_k):
        query_embedding = self.embedder.embed(query)
        distances, indices = self.search_impl(query_embedding, min(top_k, len(self.understandings)))
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            i_ref, i_u = self.understandings[idx]
            ref = self.references[i_ref]
            result = {
                "title": ref["title"],
                "understanding": ref["understandings"][i_u],
                "_similarity_score": float(1 / (1 + distance)),
            }
            for key in ["year", "urls"]:
                if key in ref:
                    result[key] = ref[key]
            results.append(result)
        return results
