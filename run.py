import importlib.resources
from kazusa.server import mcp, initialize


if __name__ == "__main__":
    root_dir = importlib.resources.files('kazusa')
    references_file = root_dir / "references.toml"
    index_file = root_dir / "faiss.index"
    initialize(references_file, index_file)
    mcp.run()
