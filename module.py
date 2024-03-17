import cv2
import json
import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import csv
from PIL import Image, ImageDraw
import seaborn as sns
from matplotlib.colors import Normalize

def sort_cluster(data_frame,n_clusters):
    
    data = data_frame
    # ラベル0の割合を計算する
    label_counts = data.groupby('cluster')['label'].value_counts(normalize=True).unstack().fillna(0)
    label0_ratios = label_counts[0] / label_counts.sum(axis=1)

    # ラベル0の割合が多い順にクラスタをソート
    sorted_clusters = label0_ratios.sort_values(ascending=False).index

    # 新しいクラスタラベルを振り直す
    new_cluster_labels = {cluster: i for i, cluster in enumerate(sorted_clusters, 0)}
    data['new_cluster'] = data['cluster'].map(new_cluster_labels)

    # 新しいクラスタラベルの対応関係をDataFrameに格納
    cluster_mapping = pd.DataFrame({'old_cluster': sorted_clusters, 'new_cluster': range(n_clusters)})
    
    return cluster_mapping

def make_cluster_bar(csv_name,n_clusters,CTS):

    df = pd.read_csv(csv_name)
    # グループごとに積み上げ棒グラフを作成
    fig, ax = plt.subplots()

    # デフォルトのMatplotlibカラーサイクルを取得
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
    # ラベルごとにグループ化
    groups = df.groupby('label')

    # カウントを初期化
    bottom = [0] * len(df['cluster'].unique())
    # クラスタごとの要素数を保存する辞書
    cluster_counts_dict = {cluster: [] for cluster in np.arange(n_clusters)}  # クラスタ数の範囲を指定

    # 各グループごとに積み上げ棒グラフを作成
    for i, (label, group) in enumerate(groups):
        cluster_counts = group['cluster'].value_counts().sort_index()
        all_clusters = range(df['cluster'].min(), df['cluster'].max() + 1)
        # 欠損値を0で埋める
        cluster_counts = cluster_counts.reindex(all_clusters, fill_value=0)
        # カラーサイクルから色を選択
        color = colors[i % len(colors)]
        # 積み上げ棒グラフをプロット
        ax.bar(cluster_counts.index, cluster_counts, label=f'Label {label}', color=color, bottom=bottom)
        #ax.bar(cluster_counts.index, cluster_counts, label=False, color=color, bottom=bottom)
        
        bottom = [sum(x) for x in zip(bottom, cluster_counts)]

        # 各クラスタごとの要素数を保存
        for cluster, count in zip(cluster_counts.index, cluster_counts):
            cluster_counts_dict[cluster].append(count)


    # グラフの設定
    ax.set_xlabel('cluster number')
    ax.set_ylabel('patch count')
    ax.set_title('patch clustering')
    ax.legend(title='Label')

    # クラスタごとの要素数を表示（小数第1位まで）
    first_elements = []
    if(CTS):
        for cluster, counts in cluster_counts_dict.items():
            total_count = sum(counts)
            percentage = int(100-(counts[0] / total_count * 100))
            first_elements.append(percentage)
        
        # first_elementsの数値を積み上げ棒グラフの上に表示
        for x, value in zip(ax.patches, first_elements):
            ax.text(x.get_x() + x.get_width() / 2, x.get_y() + x.get_height(), f'{value}%', 
                    ha='center', va='bottom', color='black', fontweight='bold', fontsize=8)
    
    # 背景色を白に設定
    fig.patch.set_facecolor('white')

    # グラフを表示
    return fig 


def find_label_in_csv(csv_filename, filenames_to_search):
    # CSVファイルをデータフレームとして読み込む
    df = pd.read_csv(csv_filename)

    # 検索結果を格納するリスト
    labels_found = []

    # ファイル名ごとに検索
    for filename in filenames_to_search:
        # DataFrameで検索して該当する行があればその"Label"を取得
        label = df.loc[df['Filename'] == filename, 'Label'].values
        if len(label) > 0:
            labels_found.append(label[0])

    return labels_found

def restore_images_from_json(json_file, output_folder, n_clusters):
    os.makedirs(output_folder, exist_ok=True) 
    # JSONファイルを読み込む
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    # カラーマップを定義
    cmap = sns.diverging_palette(center="dark", h_neg=240, h_pos=0, n=n_clusters, l=60, s=75, as_cmap=True)
    norm = Normalize(vmin=0, vmax=n_clusters-1)  # クラスター数に合わせて調整
    # 各元画像ごとに処理
    for image_file, image_info_list in json_data.items():
        # 元画像のパス
        original_image_path = os.path.join(output_folder, image_file)
        # 元画像のサイズを取得
        original_image_size = (0, 0)
        for image_info in image_info_list:
            position = image_info["position"]
            original_image_size = (
                max(original_image_size[0], position["x"] + position["width"]),
                max(original_image_size[1], position["y"] + position["height"])
            )

        # 新しい画像を作成
        original_image = Image.new("RGB", original_image_size, "white")
        # 分割画像を元画像に合成
        for image_info in image_info_list:
            position = image_info["position"]
            split_path = image_info["split_path"]
            cluster = image_info["cluster"]

            # 分割画像を開く
            split_image = Image.open(split_path)
            # クラスターに基づいてヒートマップ風の色を割り当て
            rgba_color = cmap(norm(cluster))
            cluster_color = tuple(int(c * 255) for c in rgba_color[:3])
            # 画像の各ピクセルの色を変更する処理
            draw = ImageDraw.Draw(split_image)
            width, height = split_image.size
            draw.rectangle([0, 0, width, height], fill=cluster_color)
            del draw
            # 分割画像を元画像に合成
            original_image.paste(split_image, (position["x"], position["y"]))
        # 元画像を保存
        restored_image_path = os.path.join(output_folder, f"{image_file}")
        original_image.save(restored_image_path)

def make_histgrams(data,n_clusters,test):
    
    # 各キーごとに最初に現れた "label" と "cluster" を取得
    aggregation = {}
    if test: #testのとき(labelなし)
        for key, value in data.items():
            cluster_counts = {cluster: sum(1 for item in value if item.get("cluster") == cluster) for cluster in set(item.get("cluster") for item in value)}
            aggregation[key] = {"cluster": cluster_counts}

        # aggregationからcluster_histgramsを生成
        cluster_histgrams = {key: {"histgram": [value["cluster"].get(cluster, 0) for cluster in range(n_clusters)]} for key, value in aggregation.items()}
        keys_only = list(cluster_histgrams.keys())
        histgrams = [entry["histgram"] for entry in cluster_histgrams.values()]
        return keys_only, histgrams

    else:
        for key, value in data.items():
            label = next((item["label"] for item in value if item.get("label") is not None), None)
            cluster_counts = {cluster: sum(1 for item in value if item.get("cluster") == cluster) for cluster in set(item.get("cluster") for item in value)}

            aggregation[key] = {"label": label, "cluster": cluster_counts}

        # aggregationからcluster_histgramsを生成
        cluster_histgrams = {key: {"label": value["label"], "histgram": [value["cluster"].get(cluster, 0) for cluster in range(n_clusters)]} for key, value in aggregation.items()}
        keys_only = list(cluster_histgrams.keys())
        original_labels = [entry["label"] for entry in cluster_histgrams.values()]
        histgrams = [entry["histgram"] for entry in cluster_histgrams.values()]
        return original_labels, histgrams, keys_only
