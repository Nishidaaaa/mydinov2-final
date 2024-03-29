export PYTHONPATH=/a/n-nishida/mydinov2:$PYTHONPATH 
python dinov2/run/eval/knn.py \
    --config-file ./config_file/vitg14_reg4_pretrain.yaml \
    --pretrained-weights ./pretrained_weights/dinov2_vitg14_pretrain.pth \
    --output-dir ./pathmnist_x252_knn \
    --train-dataset ImageNet:split=TRAIN:root=/a/n-nishida/pathmnist/pathmnist_x252_split:extra=/a/n-nishida/pathmnist/pathmnist_x252_split_extra \
    --val-dataset ImageNet:split=VAL:root=/a/n-nishida/pathmnist/pathmnist_x252_split:extra=/a/n-nishida/pathmnist/pathmnist_x252_split_extra \
    --test-dataset ImageNet:split=Test:root=/a/n-nishida/pathmnist/pathmnist_x252_split:extra=/a/n-nishida/pathmnist/pathmnist_x252_split_extra 
