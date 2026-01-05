from kazusa.embedder import Embedder
import numpy as np


class TestEmbedder:
    def test_embed_single(self):
        embedder = Embedder()
        text = "テスト"
        embedding = embedder.embed(text)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_embed_batch(self):
        embedder = Embedder()
        texts = ["テスト1", "テスト2", "テスト3"]
        embeddings = embedder.embed_batch(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0
