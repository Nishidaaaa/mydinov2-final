# 概要

卒業論文で製作した自己教師あり学習(DINOv2)とkMeans,RFを組み合わせた画像分類器です．DINOv2の公式Githubで配布された,ImageNet-1kでのk-NN分類を行うプログラムを改造して作っています.高解像度な画像を分割し自己教師を適用することで計算資源上の問題をクリアし，判断要因となる画像領域の考察を行うことができます．主な流れは以下の通りです.詳細は論文をご確認ください.

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

以下のようなテストデータのファイル名と正解ラベルの対応を表すcsvファイルを用意する.

```
Filename,Label
test_00000001.jpg,0
test_00000002.jpg,0
...
test_00000089.jpg,1
test_00000090.jpg,1
```

# 環境構築と実行方法

1. データの配置
    - `git clone このリポジトリ`
    - DINOv2の公式Githubで紹介されている事前訓練済みのdinov2モデルをダウンロードし,pretrained_weights以下に配置

ああああ

2. conda環境の利用
    - condaをインストールする
    - `conda env create -f conda.yaml`
    - `conda activate dinov2`

3. 画像を分割して保存

    - split_and_save.pyで画像データのパスや画像分割サイズを指定
    
    - `python split_and_save.py`

4. 分割画像(パッチ)から特徴量を抽出

    - dinov2/data/datasets/image_net.pyの_Splitクラス内のlength関数に分割後の画像枚数を入力
      
    - CTS.txtでプログラムを実行するフォルダや分割画像データのパスを指定

    - `bash run_script/CTS.txt `


6. 画像分類の訓練実行
    
    `python train_and_val.py`

8. 画像分類のテスト実行

    - test.pyでテストデータのラベルを表したcsvファイルのパスを指定

    - `python test.py`

# 設定や注意

- image_net.pyにおいて,画像データはjpgファイルをデフォルト(63行目)にしています

- dinov2/run/submit.pyでGPUを指定しています

- kMeansのクラスタ数は16にしてあります

- ホストsolarで動作確認を行いました(2024/03/18)

- 一部のpythonパッケージはconda環境の外でインストールしたためバージョンが異なる場合があります

- 本ページのdinov2のプログラムは公式Githubからcloneしたものを選抜して書き換えたものです

# おわりに

コメントが不十分な部分や分かりづらい部分などあるかと思われます. スミマセン.

# 引用

[dinov2公式Github](https://github.com/facebookresearch/dinov2)


