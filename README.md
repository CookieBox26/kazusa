# Kazusa

あなたの文献理解を登録して LLM と共有するための MCP サーバです。  
あなたの文献理解をベクトル化して [faiss](https://github.com/facebookresearch/faiss) で検索します。  
MCP サーバの実装には [fastmcp](https://github.com/jlowin/fastmcp) を使用しています。

## MCP サーバの立て方

### 1. 依存関係のインストール
`pyproject.toml` にある依存関係があればよいです。
```bash
pip install -e .  # torch が環境に既にある場合
pip install -e ".[torch-cpu]"  # torch が環境にない場合
# GPU 版の torch を利用の場合は所望の CUDA バージョン対応のものを別途入れてください
pytest  # 必要に応じて単体テストしてください
pytest -m "not slow"  # 時間短縮のため一部のテストを省略する場合 (現在あまり時間短縮効果はない)
```


### 2. 文献理解の登録

`references.toml` に文献理解を登録します。

```toml
[[references]]
title = "文献のタイトル"
year = 2023
understanding = "あなたの理解や要約をここに書く"
arxiv_id = "2301.12345"  # オプション
urls = ["https://arxiv.org/abs/2301.12345"]  # オプション
```

以下で `faiss.index` にインデクスをビルドしておくことができますが、サーバ起動時に `faiss.index` がなければ自動でビルドします。
```bash
python build_index.py
```

### 3. MCPサーバの起動

以下で MCP サーバが起動します。初回起動時または前回起動時から `references.toml` が更新された場合、サーバは自動的にインデクスをリビルドします。
```bash
python server.py
```

なお、**この MCP サーバをローカルの Claude から利用する場合は MCP サーバを起動しておく必要はありません** (Claude が自動的にサーバを起動します)。

## MCP サーバの使い方

この MCP サーバは現在以下の3つのツールを提供しています。
- `search_references`: クエリで文献を検索
- `list_all_references`: 全ての文献をリスト
- `get_reference_by_arxiv`: arXiv IDで文献を取得

### Claude CLI の場合

この MCP サーバを利用したい場所で以下を実行し MCP サーバを Claude に追加します。  
パスは適切なものにしてください。  
```bash
claude mcp add --transport stdio kazusa -- python ~/workspace/kazusa/server.py
```
claude を起動後 `/mcp` を実行すると利用できることが確認できます。  
例えば「どんな文献が登録されていますか？」といったプロンプトで、利用するか確認されるはずです。  
MCP サーバを削除したいときは以下を実行してください。
```bash
claude mcp remove kazusa
```

#### 対話例
```
> コールドスタートに関係する文献は？

● コールドスタートに関連する文献を検索します。

● kazusa - search_references (MCP)(query: "コールドスタート", top_k: 5)
  ⎿  {
       "result": [
         {
     … +44 lines (ctrl+o to expand)

● kazusa - search_references (MCP)(query: "cold start", top_k: 5)
  ⎿  {
       "result": [
         {
     … +44 lines (ctrl+o to expand)

● 検索結果を確認しましたが、現在登録されている文献の中にはコールドスタート問題に直接関連する論文は含まれていません（類似度スコアが全て0.4以下と低い値です）。

  ただし、強いて言えば以下の文献が若干関連性があるかもしれません：

  GPT-3: Language Models are Few-Shot Learners (2020)
  - arXiv ID: 2005.14165
  - 類似度スコア: 0.40 (日本語クエリ) / 0.36 (英語クエリ)
  - 内容: Few-shot学習で様々なタスクをこなせることを示した論文
  - URL: https://arxiv.org/abs/2005.14165
```