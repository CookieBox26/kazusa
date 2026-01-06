# Kazusa

あなたの文献理解を登録して LLM と共有するための MCP サーバです。  
あなたの文献理解をベクトル化して [faiss](https://github.com/facebookresearch/faiss) で検索します。  
MCP サーバの実装には [fastmcp](https://github.com/jlowin/fastmcp) を使用しています。  

> [!IMPORTANT]
> この MCP サーバは、あなたが登録した理解を架空の人物「かずさ」がもつものとします。  
> 「かずさの記憶にこの言葉はありますか？」といった対話でこの MCP サーバが利用されます。  
> 環境変数 `KAZUSA_LIBRARIAN_NAME` でお好きな名前に変更することもできます。

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

`kazusa/references.toml` に文献理解を登録します。

```toml
[[references]]
title = "文献のタイトル"
year = 2023
understanding = "あなたの理解や要約をここに書く"
arxiv_id = "2301.12345"  # オプション
urls = ["https://arxiv.org/abs/2301.12345"]  # オプション
```

以下で `kazusa/faiss.index` にインデクスをビルドしておくことができますが、サーバ起動時に `kazusa/faiss.index` がなければ自動でビルドします。
```bash
python build_index.py
```

### 3. MCPサーバの起動

以下で MCP サーバが起動します。初回起動時または前回起動時から `references.toml` が更新された場合、サーバは自動的にインデクスをリビルドします。
```bash
python run.py
```

なお、**この MCP サーバをローカルの Claude から利用する場合は MCP サーバを起動しておく必要はありません** (Claude が自動的にサーバを起動します)。

## MCP サーバの使い方

この MCP サーバは現在以下の3つのツールを提供しています。
- `search_references`： クエリで文献を検索
- `list_all_references`： 全ての文献をリスト
- `get_reference_by_arxiv`： arXiv IDで文献を取得

### Claude CLI の場合

この MCP サーバを利用したいディレクトリで以下を実行し MCP サーバを Claude に追加します。  
このプロジェクトへのパスは適切なものにしてください。  
```bash
claude mcp add --transport stdio kazusa -- python ~/workspace/kazusa/run.py
```
名前をカスタマイズする場合は以下のようにしてください。
```bash
claude mcp add --transport stdio kazusa --env KAZUSA_LIBRARIAN_NAME=アリス \
  -- python ~/workspace/kazusa/run.py
```
claude を起動後 `/mcp` を実行すると `kazusa` が利用できることが確認できます。  
例えば「かずさは事前学習について何か知っていますか？」といったプロンプトで利用されるはずです。  

MCP サーバを削除したいときは以下を実行してください。  
```bash
claude mcp remove kazusa
```

#### 対話例
```
> かずさは事前学習について何か知っていますか？

● かずさの記憶から事前学習に関する文献を検索してみますね。

● kazusa - search_references (MCP)(query: "事前学習", top_k: 5)
  ⎿  {
       "result": [
         {
     … +41 lines (ctrl+o to expand)

● はい、かずさは事前学習に関連する文献をいくつか記憶しています。以下の文献が見つかりました：

  1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
    - arXiv: 1810.04805
    - 双方向Transformerを使った事前学習モデル。マスク言語モデルとNext Sentence Predictionで学習。
    - https://arxiv.org/abs/1810.04805
  2. GPT-3: Language Models are Few-Shot Learners (2020)
    - arXiv: 2005.14165
(略)
```
