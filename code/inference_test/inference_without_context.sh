#!/bin/bash

# 获取模型路径和是否使用适配器

peft_model_path="/data/zjy/databrew/models/stqa_lora_synthetic_50_finetune_20240930_135121/checkpoint-486"  # 这里修改为你的模型路径
use_adapter=True  # 或者 False
test_file_path="../../data/in/streaming_test.csv"

# 获取当前日期和时间
current_date_time=$(date "+%Y%m%d_%H%M%S")
# 获取模型名称，假设peft_model_path的最后一个路径是模型名称
peft_model_name=$(basename -- "$peft_model_path")

# 如果use_adapter = False,peft_model_name = base
if [ "$use_adapter" = True ]; then
    peft_model_name=$(basename -- "$peft_model_path")
else
    peft_model_name="base"
fi


# 定义任务名称
task_name="${peft_model_name}_inference"

# 构建log文件名和结果文件名
log_file_path="../../experiments/infer_no_context_${current_date_time}_${peft_model_name}.log"
result_file_path="../../data/out/${peft_model_name}_res.csv"

echo "Script executed successfully. Log and result files are named as follows:"
echo "Log file: $log_file_path"
echo "Result file: $result_file_path"
# 运行Python脚本
python inference_without_context.py \
        --result_file_path "$result_file_path" \
        --test_file_path $test_file_path \
        --task "$task_name" \
        --use_adapter "$use_adapter" \
        --peft_model_path "$peft_model_path" \
        >"$log_file_path"

# 检查脚本是否成功执行
if [ $? -eq 0 ]; then
    echo "Script executed successfully. Log and result files are named as follows:"
    echo "Log file: $log_file_path"
    echo "Result file: $result_file_path"
else
    echo "Script failed to execute. Check the log file for details."
    echo "Log file: $log_file_path"
fi