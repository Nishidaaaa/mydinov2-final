import numpy as np
import torch
import glob
from urllib.parse import urlparse
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
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

from module import  make_cluster_bar, make_histgrams,find_label_in_csv,restore_images_from_json 

# n_clusters = 16
# json_folder = "../pathmnist/pathmnist_x252_json"
# output = "pathmnist_x252_result"
# feature_folder = "features"
# CTS = False 
# test_label_csv = '../pathmnist/pathmnist_test_label.csv' 

n_clusters = 16
json_folder = "/a/n-nishida/dataset/CTS_json"
output = "CTS_result"
feature_folder = "features"
CTS = True
test_label_csv = "/a/n-nishida/dataset/CTS_test_label.csv"  


test_features = torch.load(f"{feature_folder}/test_features.pt").detach().cpu().numpy()
test_fullpaths = torch.load(f"{feature_folder}/test_fullpaths.pt")


test_features = preprocessing.normalize(test_features)

saved_projection_matrix = joblib.load(f"{output}/projection_matrix.joblib")
# テストデータに対して射影行列を適用して次元削減
test_features = np.dot(test_features, saved_projection_matrix.T)

model = joblib.load(f"{output}/kmeans_model.joblib")
test_cluster = model.predict(test_features)

test_csv = f"{output}/test_data.csv"
test_data = {'fullpath': test_fullpaths, 'cluster': test_cluster}
df = pd.DataFrame(test_data)

if(CTS):
    cluster_mapping = pd.read_csv(f"{output}/cluster_mapping.csv")
    new_cluster = dict(zip(cluster_mapping['old_cluster'], cluster_mapping['new_cluster']))
    # cluster_mappingに基づいてクラスタをソート
    df['new_cluster'] = df['cluster'].map(new_cluster)
    result_data = df[['fullpath','new_cluster']]
    result_data = result_data.rename(columns={'new_cluster': 'cluster'})
    result_data.to_csv(test_csv, index=False)
else:
    df.to_csv(test_csv, index=False)

test_json = f"{json_folder}/Test.json"
with open(test_json, 'r') as json_file:
    test_dict = json.load(json_file)

df = pd.read_csv(test_csv)
test_cluster = df['cluster']
test_cluster = [int(x) for x in test_cluster]
cluster_index = 0
for key, value in test_dict.items():
    for item in value:
        item["cluster"] = test_cluster[cluster_index]
        cluster_index += 1

test_json_with_cluster = f"{json_folder}/Test_with_cluster.json"
with open(test_json_with_cluster, 'w') as json_file:
    json.dump(test_dict, json_file, indent=2)

filenames, test_histgrams = make_histgrams(test_dict,n_clusters,test=True)


original_test_labels = find_label_in_csv(test_label_csv, filenames)

rf_model = joblib.load(f"{output}/random_forest_model.joblib")

test_pred = rf_model.predict(test_histgrams)

test_accuracy = accuracy_score(original_test_labels, test_pred)
print("test Accuracy:", test_accuracy)

fig, ax = plt.subplots()
# 混同行列の計算
conf_matrix = confusion_matrix(original_test_labels, test_pred)

# seabornを使用してヒートマップを描画
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(original_test_labels), yticklabels=np.unique(test_pred))
plt.title(f"dinov2 test result {round(test_accuracy, 3)}")
plt.xlabel('test_pred')
plt.ylabel('test_labels')

# 保存する場合

plt.savefig(f"{output}/dinov2_matrix_test.png")

if(CTS):
    print("heatmap making ...")
    # 画像の保存先フォルダ
    heatmap_test_folder = f"{output}/heatmaps/test"  # 保存先フォルダのパスを指定してください
    # 画像を復元
    restore_images_from_json(test_json_with_cluster, heatmap_test_folder, n_clusters)

else:
    print("done")
