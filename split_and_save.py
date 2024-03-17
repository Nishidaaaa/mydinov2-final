import cv2
import json
import os
import numpy as np
import glob
import shutil

from dinov2.data.datasets import ImageNet

def split_and_save(image_path, output, split_size, index, mode, ext):
    label = os.path.basename(os.path.dirname(image_path)) 
    output_folder = f"{output}/{mode}/{label}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image = cv2.imread(image_path)
    # 画像の高さと幅を取得
    height, width = image.shape[:2]

    # 分割する領域のサイズ
    split_height = split_size
    split_width = split_size

    # 分割された画像の位置情報とデータを保存するためのリスト
    data = []

    # 画像を分割して保存
    for y in range(0, height, split_height):
        for x in range(0, width, split_width):
            # 分割された領域を取得
            split_region = image[y:y+split_height, x:x+split_width]

            # 分割された領域の位置情報を保存
            position_info = {'x': x, 'y': y, 'width': split_width, 'height': split_height}

            # 分割画像の保存先パス
            actual_index = len(data) + index
            if mode == "train":
                split_path = f"{output_folder}/{label}_{actual_index}.{ext}"
            elif mode == "val":
                split_path = f"{output_folder}/val_{str(actual_index).zfill(8)}.{ext}"
            elif mode == "Test":
                split_path = f"{output_folder}/Test_{str(actual_index).zfill(8)}.{ext}"
            # 分割画像を保存
            cv2.imwrite(split_path, split_region)

            # データに位置情報と分割画像のパスを追加
            data.append({'position': position_info, 'split_path': split_path})
    #output_json_path = f"{json_path}/{mode}/{file_name}_positions.json"
    
    #with open(output_json_path, 'w') as f:
        #json.dump(data, f)

    return data

if __name__ == "__main__":
    # 元の画像のフォルダ(絶対パス)
    base_folder = "/a/n-nishida/dataset/CTS"
    # 画像分割サイズ
    split_size = 224 
    # 画像拡張子
    ext = "jpg" 
    # 分割画像を保存するフォルダ
    output = f"{base_folder}_split"
    # 位置情報を保存するフォルダ
    json_folder = f"{base_folder}_json"
    
    os.makedirs(json_folder)
    
    train_dict = {}
    index = 0
    os.makedirs(output + "/train")
    train_files = glob.glob(f"{base_folder}/train/*/*.{ext}")
    print("train split ...")
    for filepath in train_files:
        mode = "train"
        file_name = os.path.basename(filepath)
        label = os.path.basename(os.path.dirname(filepath))

        # 分割して位置情報を保存
        data = split_and_save(filepath, output, split_size, index, mode, ext)
        index += len(data)
        train_dict[file_name] = data
    with open(f"{json_folder}/train.json", 'w') as file:
        json.dump(train_dict, file, indent=2)
    
    val_dict = {}
    index = 0
    os.makedirs(output + "/val")
    val_files = glob.glob(f"{base_folder}/val/*/*.{ext}")
    print("val split ...")
    for filepath in val_files:
        mode = "val"
        file_name = os.path.basename(filepath)
        label = os.path.basename(os.path.dirname(filepath))

        # 分割して位置情報を保存
        data = split_and_save(filepath, output, split_size, index, mode, ext)
        index += len(data)
        val_dict[file_name] = data
    with open(f"{json_folder}/val.json", 'w') as file:
        json.dump(val_dict, file, indent=2)   

    test_dict = {}
    index = 0
    os.makedirs(output + "/Test")
    test_files = glob.glob(f"{base_folder}/test/*/*.{ext}")
    print("test split ...")
    for filepath in test_files:
        mode = "Test"
        file_name = os.path.basename(filepath)
        label = os.path.basename(os.path.dirname(filepath))

        # 分割して位置情報を保存
        data = split_and_save(filepath, output, split_size, index, mode, ext)
        index += len(data)
        test_dict[file_name] = data
    with open(f"{json_folder}/Test.json", 'w') as file:
        json.dump(test_dict, file, indent=2)

    source_file = f"{base_folder}/labels.txt" 
    shutil.copy(source_file, output)
    print("extra preparation ...")
    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root=output , extra=f"{output}_extra")
        dataset.dump_extra()

    