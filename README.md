# ゴルフスイング分析｜腰の移動量（cm）

既知の長さの棒を基準に、P1（アドレス）からの腰の中心の左右・上下の移動量をP1〜P10で表示します。正面・後方の動画に対応しています。

## GitHubにアップロードする

1. GitHubで試用用のリポジトリを作成します。
2. ZIPを展開し、中のファイルと `.streamlit` フォルダをリポジトリ直下へアップロードします。空のリポジトリでは「uploading an existing file」から、ファイルがある場合は「Add file」→「Upload files」から進みます。
3. コミット後、`app.py`、`requirements.txt`、`packages.txt` が同じ階層にあり、`.streamlit/config.toml` もアップロードされていることを確認します。

[GitHubのアップロード手順](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## Streamlit Community Cloudで起動する

1. Streamlit Community Cloudで「Create app」を選び、新しいGitHubリポジトリを指定します。
2. 「Branch」にアップロードしたブランチ名（通常は `main`）、「Main file path」に `app.py` を指定します。
3. 「Advanced settings」で Python 3.12 を選び、「Deploy」を押します。AIモデルは最初の骨格解析時に取得します。

[Streamlit Community Cloudのデプロイ手順](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 腰の移動量を測る

動画をAI骨格解析し、P1をアドレスに合わせます。プレビュー下の「基準棒で腰の移動量を測る（cm）」で棒の長さを入力し、映像上で棒の両端をクリックします。カメラを固定し、棒と腰が同じ奥行きになるように撮影してください。表示値は画面上の2D推定値です。

Windowsでは、展開後に `start_windows.bat` で起動できます。詳しい操作は [日本語ガイド](README_日本語.md) を参照してください。
