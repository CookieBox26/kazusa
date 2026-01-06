import pytest
import numpy as np
import time
from pathlib import Path
from kazusa.embedder import Embedder
from kazusa.index_manager import IndexManager


class TestIndexManager:
    references_file = 'tests/data/test_refs_tmp.toml'
    index_file = 'tests/data/test_index_tmp.index'

    def setup_method(self):
        Path(type(self).references_file).write_text((
            '[[references]]\n'
            'title = "Test"\n'
            'understanding = "テスト"\n'
        ), encoding="utf-8")
        Path(type(self).index_file).unlink(missing_ok=True)

    def teardown_method(self):
        Path(type(self).references_file).unlink(missing_ok=True)
        Path(type(self).index_file).unlink(missing_ok=True)

    @pytest.mark.slow
    def test_loads_existing_index_when_newer(self):
        """既存のインデックスファイルがreferencesより新しい場合は読み込む"""
        embedder = Embedder()
        manager1 = IndexManager(embedder, type(self).references_file, type(self).index_file)
        index_mtime_before = Path(type(self).index_file).stat().st_mtime
        time.sleep(0.1)
        manager2 = IndexManager(embedder, type(self).references_file, type(self).index_file)
        index_mtime_after = Path(type(self).index_file).stat().st_mtime
        assert index_mtime_before == index_mtime_after

    @pytest.mark.slow
    def test_rebuilds_when_references_newer(self):
        """referencesファイルがインデックスより新しい場合は再ビルド"""
        embedder = Embedder()
        manager1 = IndexManager(embedder, type(self).references_file, type(self).index_file)
        index_mtime_before = Path(type(self).index_file).stat().st_mtime
        time.sleep(0.1)
        Path(type(self).references_file).write_text((
            '[[references]]\n'
            'title = "Test"\n'
            'understanding = "テスト"\n\n'
            '[[references]]\n'
            'title = "Test2"\n'
            'understanding = "テスト2"\n'
        ), encoding="utf-8")
        manager2 = IndexManager(embedder, type(self).references_file, type(self).index_file)
        index_mtime_after = Path(type(self).index_file).stat().st_mtime
        assert index_mtime_after > index_mtime_before
        assert manager2.index.ntotal == 2

    @pytest.mark.slow
    def test_force_build_flag(self):
        """force_build=Trueの場合は必ず再ビルド"""
        embedder = Embedder()
        manager1 = IndexManager(embedder, type(self).references_file, type(self).index_file)
        index_mtime_before = Path(type(self).index_file).stat().st_mtime
        time.sleep(0.1)
        manager2 = IndexManager(embedder, type(self).references_file, type(self).index_file, force_rebuild=True)
        index_mtime_after = Path(type(self).index_file).stat().st_mtime
        assert index_mtime_after > index_mtime_before

    def test_search_method(self):
        """search()メソッドがクエリ文字列から検索結果を返す"""
        references_file = 'tests/data/references.toml'
        index_file = 'tests/data/index.index'
        embedder = Embedder()
        index_manager = IndexManager(embedder, references_file, type(self).index_file)
        results = index_manager.search("Transformer", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "_similarity_score" in results[0]
        assert "title" in results[0]


class TestCreateReferenceText:
    def test_full_reference_data(self):
        reference = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "understanding": "Transformerアーキテクチャを提案した論文",
            "arxiv_id": "1706.03762",
            "urls": ["https://arxiv.org/abs/1706.03762"]
        }
        text = IndexManager.create_reference_text(reference)
        assert "Attention Is All You Need" in text
        assert "2017" in text
        assert "Transformerアーキテクチャを提案した論文" in text
        assert "タイトル:" in text
        assert "出版年:" in text
        assert "理解:" in text

    def test_reference_without_year(self):
        reference = {"title": "Test Reference", "understanding": "テスト用の文献"}
        text = IndexManager.create_reference_text(reference)
        assert "Test Reference" in text
        assert "テスト用の文献" in text
        assert "出版年:" not in text

    def test_reference_with_empty_understanding(self):
        reference = {"title": "Empty Understanding Reference", "year": 2023, "understanding": ""}
        text = IndexManager.create_reference_text(reference)
        assert "理解:" in text
        assert "Empty Understanding Reference" in text

    def test_minimal_reference(self):
        reference = {"title": "Minimal Reference"}
        text = IndexManager.create_reference_text(reference)
        assert "Minimal Reference" in text
        assert "理解:" in text
