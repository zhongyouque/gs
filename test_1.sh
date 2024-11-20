#!/bin/bash

# 定义路径
dataset_path=$1  # 输入路径：COLMAP 或 NeRF Synthetic 数据集路径
model_path=$2    # 输入路径：训练好的模型路径
cuda_device=$3   # 输入参数：CUDA_VISIBLE_DEVICES 的值
dp=$4            # 输入参数：--dp 值
x=$5             # 输入参数：--x 值
y=$6             # 输入参数：--y 值
small=$7         # 输入参数：--small 值
ip=$8            # 输入参数：--ip 值

# 创建日志文件路径，将日志文件放在 model_path 目录下
log_file="$model_path/output_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$model_path"

# 开始记录时间
start_time=$(date +%s)

echo "Starting training..." | tee -a "$log_file"
# 执行训练命令，并保存输出
CUDA_VISIBLE_DEVICES="$cuda_device" python train.py \
    -s "$dataset_path" -m "$model_path" \
    --clone 'v2' --split 'v4' \
    --dp "$dp" --x "$x" --y "$y" --small "$small" \
    --ip "$ip" --eval 2>&1 | tee -a "$log_file"

# 训练结束时间
end_time=$(date +%s)
train_duration=$((end_time - start_time))

echo "Training completed in $train_duration seconds" | tee -a "$log_file"

# 开始渲染
echo "Starting rendering..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES="$cuda_device" python render.py \
    -m "$model_path" --iteration 30000 --skip_train 2>&1 | tee -a "$log_file"

echo "Starting rendering..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES="$cuda_device" python render.py \
    -m "$model_path" --iteration 25000 --skip_train 2>&1 | tee -a "$log_file"

echo "Starting rendering..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES="$cuda_device" python render.py \
    -m "$model_path" --iteration 20000 --skip_train 2>&1 | tee -a "$log_file"

# 渲染结束时间
end_time=$(date +%s)
render_duration=$((end_time - start_time - train_duration))

echo "Rendering completed in $render_duration seconds" | tee -a "$log_file"

# 开始计算指标
echo "Starting metrics calculation..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES="$cuda_device" python metrics.py \
    -m "$model_path" 2>&1 | tee -a "$log_file"

# 计算结束时间
end_time=$(date +%s)
metrics_duration=$((end_time - start_time - train_duration - render_duration))

echo "Metrics calculation completed in $metrics_duration seconds" | tee -a "$log_file"

# 总时间
total_duration=$((end_time - start_time))
echo "Total process completed in $total_duration seconds" | tee -a "$log_file"
