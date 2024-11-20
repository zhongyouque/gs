#!/bin/bash

# 定义输入参数组
declare -a dataset_paths=(
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/bicycle"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/bonsai"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/counter"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/garden"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/kitchen"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/room"
    "/home/zy/Project/test/gaussian-splatting/data/360_v2/stump"
    # 添加更多数据集路径...
)

declare -a model_paths=(
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/bicycle"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/bonsai"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/counter"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/garden"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/kitchen"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/room"
    "/home/zy/Project/test/gaussian-splatting/output/mipnerf360_v0/stump"
    # 添加更多模型路径...
)
# 检查参数组数是否匹配
if [ ${#dataset_paths[@]} -ne ${#model_paths[@]} ]; then
    echo "Error: The number of dataset paths and model paths must match."
    exit 1
fi

# 循环调用原始脚本
for i in "${!dataset_paths[@]}"; do
    dataset_path=${dataset_paths[$i]}
    model_path=${model_paths[$i]}

    # 调用原始脚本
    echo "Running training and rendering for dataset $dataset_path and model $model_path"
    ./test_4.sh "$dataset_path" "$model_path"
done

echo "All processes completed."