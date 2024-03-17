# 概要


卒業論文で製作した自己教師あり学習(DINOv2)とkMeans,RFを組み合わせた画像分類器です．高解像度な画像を分割し自己教師を適用することで，計算資源上の問題をクリアし，また，判断要因となる画像領域の考察を行うことができます．主な流れは以下の通りです。詳細は論文をご確認ください。

画像を任意の大きさに分割　→　分割画像から特徴量を抽出　→　それを基に画像を分類

# データの準備

画像データのディレクトリ構造は次のようにします

- `<ROOT>/train/0/0_1.jpeg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/1/1_81.jpeg`
- `<ROOT>/val/val_00000001.jpeg`
- `<ROOT>/val/[...]`
- `<ROOT>/val/val_00000060.jpeg`
- `<ROOT>/test/test_00000001.jpeg`
- `<ROOT>/test/[...]`
- `<ROOT>/test/test_00000090.jpeg`
- `<ROOT>/labels.txt`

# 環境構築と実行方法

1. データの配置
    - `git clone このリポジトリ`
    - 画像データを配置 (例として../dataset以下にCTS画像を配置)
    - DINOv2の公式Githubで紹介されている事前訓練済みのdinov2モデルをダウンロードし、pretrained_weights以下に配置

2. conda環境の利用

    `conda activate dinov2`

3. 画像を分割して保存

    `python split_and_save.py`

4. 分割画像(パッチ)から特徴量を抽出


    `bash run_script/CTS.txt `


5. 画像分類の訓練実行

    `python train_and_val.py`

6. 画像分類のテスト実行

   `python test.py`

# 設定

-

-

-

-




## 特徴


## はじめ方

プロジェクトのセットアップ方法や使用方法に関する手順を提供します。必要な前提条件、インストール手順、使用例を含めます。

### 必要条件

プロジェクトを使用する前にインストールが必要なソフトウェアやライブラリをリストします。

### インストール

プロジェクトをインストールする手順をステップバイステップで説明します。必要なコマンドやスクリプトを含めます。

### 使用方法

プロジェクトの使用方法の例を提供します。コードスニペットやコマンドラインの使用例、スクリーンショットなどを含めます。


