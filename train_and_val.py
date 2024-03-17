import numpy as np
import torch
import glob
from urllib.parse import urlparse
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
import cv2
import mlflow
import time
import joblib
import networkx as nx
import leidenalg
import sys
import igraph as ig
import json
import csv
from sklearn.metrics import confusion_matrix

from module import  make_cluster_bar, make_histgrams,sort_cluster,restore_images_from_json

experiment = mlflow.set_experiment('Path_clustering')

# pt_folder = "pathmnist_x252_pt"
# reconstructed_folder = "CTS_reconstructed"
# json_folder = "../pathmnist/pathmnist_x252_json"
# output = "pathmnist_x252_result"
# experiment = mlflow.set_experiment('path_clustering')

# n_clusters = 16
# pt_folder = "CTS_100_pt"
# json_folder = "../pathmnist/CTS_100_json"
# output = f"finaldata/CTS_100/no_pca/{n_clusters}"
# CTS = True

n_clusters = 16
pt_folder = "CTS_100_pt"
json_folder = "../pathmnist/CTS_100_json"
output = f"finaldata/CTS_100/no_pca/{n_clusters}"
CTS = True

with mlflow.start_run(experiment_id=experiment.experiment_id):
    print("start training ...")
    if not os.path.exists(output):
        os.makedirs(output)
    
    #特徴量をtensorからnumpyに 
    train_features = torch.load(f"{pt_folder}/train_features.pt").detach().cpu().numpy()
    train_labels = torch.load(f"{pt_folder}/train_labels.pt").detach().cpu().numpy()
    train_fullpaths = torch.load(f"{pt_folder}/train_fullpaths.pt")
    val_features = torch.load(f"{pt_folder}/val_features.pt").detach().cpu().numpy()
    val_labels = torch.load(f"{pt_folder}/val_labels.pt").detach().cpu().numpy()
    val_fullpaths = torch.load(f"{pt_folder}/val_fullpaths.pt")
    test_features = torch.load(f"{pt_folder}/test_features.pt").detach().cpu().numpy()
    # test_labels = torch.load(f"{pt_folder}/test_labels.pt").detach().cpu().numpy()
    test_fullpaths = torch.load(f"{pt_folder}/test_fullpaths.pt")
    
    concatenated_features = np.concatenate((train_features, val_features), axis=0)
    concatenated_features = preprocessing.normalize(concatenated_features)

    pca_full = PCA()
    pca_full.fit(concatenated_features)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    #print("Explained Variance Ratio:", explained_variance_ratio)

    # 累積寄与率の計算
    cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)

    # 累積寄与率が0.8を超える最小の成分数を見つける
    num_components = np.where(cumulative_variance_ratio >= 0.8)[0][0] + 1
    
    print(f"累積寄与率が80%になる主成分の数: {num_components}")
    
    # pca = PCA(n_components=num_components)
    # concatenated_features = pca.fit_transform(concatenated_features)
    # print('pca done')
    # joblib.dump(pca.components_, f"{output}/projection_matrix.joblib")

    #kmeansでクラスタリング
    print('kmeans training ...')
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=1)
    model.fit(concatenated_features)
    concatenated_cluster = model.labels_
    print('kmeans done')
    joblib.dump(model, f"{output}/kmeans_model.joblib")

    # ラベル,パス,クラスタをcsvに保存
    train_csv = f"{output}/train_data.csv"
    train_data = {'label': train_labels, 'fullpath': train_fullpaths, 'cluster': concatenated_cluster[:len(train_features)]}
    df1 = pd.DataFrame(train_data)
    # trainのfilename的に_1,_10,_2,のような順番になるのでソートして_1,_2,_10のようにしたい
    df1["filename_sort_key"] = df1["fullpath"].apply(lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    df1_sorted = df1.sort_values(by="filename_sort_key").drop(columns="filename_sort_key")

    val_csv = f"{output}/val_data.csv"
    val_data = {'label': val_labels, 'fullpath': val_fullpaths, 'cluster': concatenated_cluster[len(train_features):]}
    df2 = pd.DataFrame(val_data)
    
    if(CTS == True):#根拠領域をheatmapで出力するので、ラベルの割合でクラスタをソートする
        # trainとvalをくっつけてクラスタをソートする 
        concatenated_df = pd.concat([df1_sorted, df2], ignore_index=True)
        # 古いクラスタと新しいクラスタの対応関係を入手
        cluster_mapping = sort_cluster(concatenated_df,n_clusters)
        cluster_mapping.to_csv(f'{output}/cluster_mapping.csv', index=False)
        new_cluster = dict(zip(cluster_mapping['old_cluster'], cluster_mapping['new_cluster']))
        # cluster_mappingに基づいてクラスタをソート
        df1_sorted['new_cluster'] = df1_sorted['cluster'].map(new_cluster)
        result_data = df1_sorted[['label','fullpath','new_cluster']]
        result_data = result_data.rename(columns={'new_cluster': 'cluster'})
        result_data.to_csv(train_csv, index=False)
        # cluster_mappingに基づいてクラスタをソート
        df2['new_cluster'] = df2['cluster'].map(new_cluster)
        result_data = df2[['label','fullpath','new_cluster']]
        result_data = result_data.rename(columns={'new_cluster': 'cluster'})
        result_data.to_csv(val_csv, index=False)
    else:
        df1_sorted.to_csv(train_csv, index=False)
        df2.to_csv(val_csv, index=False)
    #クラスタリング結果保存
    fig = make_cluster_bar(train_csv,n_clusters,CTS)
    fig.savefig(f"{output}/train_clustering.png")
    fig = make_cluster_bar(val_csv,n_clusters,CTS)
    fig.savefig(f"{output}/val_clustering.png")
    
    

    train_json = f"{json_folder}/train.json"
    with open(train_json, 'r') as json_file:
        train_dict = json.load(json_file)

    df = pd.read_csv(train_csv)
    train_cluster = df['cluster']
    train_cluster = [int(x) for x in train_cluster]
    train_label = df['label']
    train_label = [int(x) for x in train_label]
    cluster_index = 0
    for key, value in train_dict.items():
        for item in value:
            item["cluster"] = train_cluster[cluster_index]
            item["label"] = train_label[cluster_index]
            cluster_index += 1
    
    val_json = f"{json_folder}/val.json"
    with open(val_json, 'r') as json_file:
        val_dict = json.load(json_file)

    df = pd.read_csv(val_csv)
    val_cluster = df['cluster']
    val_cluster = [int(x) for x in val_cluster]
    val_label = df['label']
    val_label = [int(x) for x in val_label]
    cluster_index = 0
    for key, value in val_dict.items():
        for item in value:
            item["cluster"] = val_cluster[cluster_index]
            item["label"] = val_label[cluster_index]
            cluster_index += 1
    
    train_json_with_cluster = f"{json_folder}/train_with_cluster.json"
    with open(train_json_with_cluster, 'w') as json_file:
        json.dump(train_dict, json_file, indent=2)
    
    val_json_with_cluster = f"{json_folder}/val_with_cluster.json"
    with open(val_json_with_cluster, 'w') as json_file:
        json.dump(val_dict, json_file, indent=2)
    
    original_train_labels, train_histgrams, train_filenames  = make_histgrams(train_dict,n_clusters,test=False)
    original_val_labels, val_histgrams, val_filenames = make_histgrams(val_dict,n_clusters,test=False)

    
    # ランダムフォレストモデルの定義
    rf_model = RandomForestClassifier()
    print("ramdomforest training ...")
    # ハイパーパラメータの範囲を指定
    grid = {
        'n_estimators': [100, 200, 300],  # 生成する決定木の数
        'max_features': ['auto', 'sqrt', 'log2'],  # 最大の特徴量数
        'max_depth': [10, 20, 30, None],  # 木の深さ
        'min_samples_split': [2, 5, 10],  # 内部ノードを分割するための最小サンプル数
        'min_samples_leaf': [1, 2, 4],  # 葉の最小サンプル数
        'bootstrap': [True, False],  # ブートストラップサンプリングの使用
        'criterion': ['gini', 'entropy']  # 不純度の測定基準
    }

    with open(f'{output}/parameter_grid.json', 'w') as f:
        json.dump(grid, f)
    # グリッドサーチの設定
    grid_search = GridSearchCV(estimator=rf_model, param_grid=grid, cv=5, n_jobs=-1)

    data_hist = train_histgrams + val_histgrams
    data_label = original_train_labels + original_val_labels
    # グリッドサーチ実行
    grid_search.fit(data_hist, data_label)
    print("ramdomforest done")
    # 最適なモデルおよびハイパーパラメータの組み合わせを取得
    best_model = grid_search.best_estimator_
    bestParam = grid_search.best_params_
    bestScore = grid_search.best_score_
    
    # モデルの保存
    joblib.dump(best_model, f"{output}/random_forest_model.joblib")
    print('training saved ---> random_forest_model.joblib')
    # バリデーションデータでの評価
    train_pred = best_model.predict(train_histgrams)
    train_accuracy = accuracy_score(original_train_labels, train_pred)
    # print("train accuracy:", train_accuracy)
    val_pred = best_model.predict(val_histgrams)
    val_accuracy = accuracy_score(original_val_labels, val_pred)
    # print("val Accuracy:", val_accuracy)
    print(f'{bestParam}のときに正答率が最高で{bestScore}')
    
    data_pred = best_model.predict(data_hist)
    
    fig, ax = plt.subplots()
    conf_matrix = confusion_matrix(data_label, data_pred)
    # seabornを使用してヒートマップを描画
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(original_val_labels), yticklabels=np.unique(val_pred))
    plt.title(f"dinov2 result {round(bestScore, 3)}")
    plt.xlabel('pred')
    plt.ylabel('labels')

    # 保存する場合

    plt.savefig(f"{output}/dinov2_matrix.png")

    
    fig, ax = plt.subplots()
    # 混同行列の計算
    conf_matrix = confusion_matrix(original_val_labels, val_pred)

    # seabornを使用してヒートマップを描画
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(original_val_labels), yticklabels=np.unique(val_pred))
    plt.title(f"dinov2 val result {round(val_accuracy, 3)}")
    plt.xlabel('val_pred')
    plt.ylabel('val_labels')

    # 保存する場合
    plt.savefig(f"{output}/dinov2_matrix_val.png")

    fig, ax = plt.subplots()
    # 混同行列の計算
    conf_matrix = confusion_matrix(original_train_labels, train_pred)

    # seabornを使用してヒートマップを描画
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(original_train_labels), yticklabels=np.unique(train_pred))
    plt.title(f"dinov2 train result {round(train_accuracy, 3)}")
    plt.xlabel('train_pred')
    plt.ylabel('train_labels')

    # 保存する場合
    plt.savefig(f"{output}/dinov2_matrix_train.png")
    
    # CSVファイルを書き込みモードで開く
    pred_csv = f"{output}/train_pred.csv"
    with open(pred_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーの書き込み
        csvwriter.writerow(['Number', 'Letter'])
        # リストの内容を行として書き込み
        for i in range(len(train_filenames)):
            csvwriter.writerow([train_filenames[i], train_pred[i]])


    if(CTS):
        print("heatmap making ...")
        # 画像の保存先フォルダ
        heatmap_train_folder = f"{output}/heatmaps/train"  # 保存先フォルダのパスを指定してください
        # 画像を復元
        restore_images_from_json(train_json_with_cluster, heatmap_train_folder, n_clusters)
        # 画像の保存先フォルダ
        heatmap_val_folder = f"{output}/heatmaps/val"  # 保存先フォルダのパスを指定してください
        # 画像を復元
        restore_images_from_json(val_json_with_cluster, heatmap_val_folder, n_clusters)
        print("heatmap done")
    else:
        print("done")