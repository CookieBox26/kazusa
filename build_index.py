from pathlib import Path
from kazusa.embedder import Embedder
from kazusa.index_manager import IndexManager


if __name__ == "__main__":
    """インデクスのビルドのみ行います"""
    root_dir = Path(__file__).parent
    embedder = Embedder()
    references_file = root_dir / "references.toml"
    index_file = root_dir / "faiss.index"
    IndexManager(
        embedder,
        references_file.as_posix(),
        index_file.as_posix(),
        force_rebuild=True,
    )
