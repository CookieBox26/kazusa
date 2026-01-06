import os
import importlib.resources
from typing import List, Dict, Any
from fastmcp import FastMCP
from kazusa.embedder import Embedder
from kazusa.index_manager import IndexManager


mcp = FastMCP("Kazusa")
index_manager: IndexManager | None = None
librarian_name = os.environ.get("KAZUSA_LIBRARIAN_NAME", "かずさ")


def initialize(references_file, index_file):
    global index_manager
    embedder = Embedder()
    index_manager = IndexManager(
        embedder,
        references_file,
        index_file,
    )


@mcp.tool(
    description=f"{librarian_name}の記憶から文献を検索します。"
    f"{librarian_name}の記憶にある文献についての質問に答えるときに使用してください。"
)
def search_references(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return index_manager.search(query, top_k)


@mcp.tool(
    description=f"{librarian_name}が記憶している全ての文献をリストします。"
    f"{librarian_name}がどんな文献を知っているか確認するときに使用してください。"
)
def list_all_references() -> List[Dict[str, Any]]:
    return index_manager.references


@mcp.tool(
    description=f"{librarian_name}の記憶から特定の arXiv ID の文献を取得します。"
)
def get_reference_by_arxiv(arxiv_id: str) -> Dict[str, Any] | None:
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
