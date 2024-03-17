# 概要


卒業論文で製作した自己教師あり学習(DINOv2)とkMeans,RFを組み合わせた画像分類器です．高解像度な画像を分割し自己教師を適用することで，計算資源上の問題をクリアし，また，判断要因となる画像領域の考察を行うことができます．



# 環境構築と実行方法
1. データの配置
    - `git clone このリポジトリ`
    - 画像データを配置()
2. conda環境の利用

    `conda activate dinov2`

3. Docker run


    `make docker-run`

---以下コンテナの中---


4. mlflowのUIサーバー立ち上げ


    `nohup mlflow ui --port 5000 -h 0.0.0.0 & `


5. 訓練実行

    `python run_all.py`

6. 実験結果確認

    [localhost:5000](http://localhost:5000)

## はじめに

このセクションでは、プロジェクトの簡単な概要を提供します。プロジェクトの目的や存在意義、誰が利用するのかについて説明します。

## 特徴


## はじめ方

プロジェクトのセットアップ方法や使用方法に関する手順を提供します。必要な前提条件、インストール手順、使用例を含めます。

### 必要条件

プロジェクトを使用する前にインストールが必要なソフトウェアやライブラリをリストします。

### インストール

プロジェクトをインストールする手順をステップバイステップで説明します。必要なコマンドやスクリプトを含めます。

### 使用方法

プロジェクトの使用方法の例を提供します。コードスニペットやコマンドラインの使用例、スクリーンショットなどを含めます。


