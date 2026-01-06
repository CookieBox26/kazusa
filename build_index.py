import importlib.resources
from kazusa.embedder import Embedder
from kazusa.index_manager import IndexManager


if __name__ == "__main__":
    """インデクスのビルドのみ行います"""
    root_dir = importlib.resources.files('kazusa')
    embedder = Embedder()
    references_file = root_dir / "references.toml"
    index_file = root_dir / "faiss.index"
    IndexManager(
        embedder,
        references_file,
        index_file,
        force_rebuild=False,
    )
