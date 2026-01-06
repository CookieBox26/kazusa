import sys
import importlib.resources
from typing import List, Dict, Any
from fastmcp import FastMCP
from kazusa.embedder import Embedder
from kazusa.index_manager import IndexManager


mcp = FastMCP("Kazusa")
index_manager: IndexManager | None = None


def initialize(references_file, index_file):
    global index_manager
    embedder = Embedder()
    index_manager = IndexManager(
        embedder,
        references_file,
        index_file,
    )


@mcp.tool()
def search_references(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    かずさ (Kazusa) の記憶から文献を検索します。
    かずさの記憶にある文献についての質問に答えるときに使用してください。
    """
    return index_manager.search(query, top_k)


@mcp.tool()
def list_all_references() -> List[Dict[str, Any]]:
    """
    かずさ (Kazusa) が記憶している全ての文献をリストします。
    かずさがどんな文献を知っているか確認するときに使用してください。
    """
    return index_manager.references


@mcp.tool()
def get_reference_by_arxiv(arxiv_id: str) -> Dict[str, Any] | None:
    """
    かずさ (Kazusa) の記憶から特定の arXiv ID の文献を取得します。
    """
    for reference in index_manager.references:
        if reference.get("arxiv_id") == arxiv_id:
            return reference
    return None


if __name__ == "__main__":
    root_dir = importlib.resources.files('kazusa')
    references_file = root_dir / "references.toml"
    index_file = root_dir / "faiss.index"
    initialize(references_file, index_file)
    mcp.run()
