cuda_device_list=7
dataset_path="/home/zy/Project/old/gaussian-splatting/data/mipnerf360/bicycle"   # 固定数据集路径
model_base_path="/home/zy/Project/test/gaussian-splatting/outputs/bicycle_test_V22"  # 模型路径的基础目录

dp_list=(0.006 0.008)          # dp 参数列表
x_list=(1.0 1.2)              # x 参数列表
y_list=(0.2 0.4 0.6)             # y 参数列表
small_list=(1.8 2.2 2.6)          # small 参数列表
ip="127.0.0.1"
# 主脚本路径
main_script="test_0.sh"

# 循环遍历参数组合
for dp in "${dp_list[@]}"; do
    for x in "${x_list[@]}"; do
        for y in "${y_list[@]}"; do
            for small in "${small_list[@]}"; do
                # 为每组参数创建唯一的模型路径
                model_path="$model_base_path/7/dp_${dp}_x_${x}_y_${y}_small_${small}"
                
                # 调用主脚本
                bash "$main_script" "$dataset_path" "$model_path" "$cuda_device_list" "$dp" "$x" "$y" "$small" "$ip"
            done
        done
    done
done