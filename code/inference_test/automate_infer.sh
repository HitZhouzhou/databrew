#!/bin/bash

# 模型路径列表，使用空格分隔
peft_model_paths=(
    # "../../models/synthetic_streaming_exqa_finetune_20240929_220300"
    "../../models/streaming_test_finetune_20240929_221129"
)

use_adapter=True  # 或者 False
test_file_path="../../data/in/streaming_test.csv"

# 遍历每个模型路径并进行推理测试
for peft_model_path in "${peft_model_paths[@]}"; do

    # 获取当前日期和时间
    current_date_time=$(date "+%Y%m%d_%H%M%S")
    # 获取模型名称，假设 peft_model_path 的最后一个路径是模型名称
    peft_model_name=$(basename -- "$peft_model_path")

    # 如果 use_adapter = False, peft_model_name = base
    if [ "$use_adapter" = True ]; then
        peft_model_name=$(basename -- "$peft_model_path")
    else
        peft_model_name="base"
    fi

    # 定义任务名称
    task_name="${peft_model_name}_inference"

    # 构建 log 文件名和结果文件名
    log_file_path="../../experiments/infer_no_context_${current_date_time}_${peft_model_name}.log"
    result_file_path="../../data/out/${current_date_time}_${peft_model_name}_res.csv"

    # 输出正在执行的模型信息
    echo "Running inference for model: $peft_model_name"
    echo "Log file: $log_file_path"
    echo "Result file: $result_file_path"

    # 运行 Python 脚本
    python inference_without_context.py \
        --result_file_path "$result_file_path" \
        --test_file_path "$test_file_path" \
        --task "$task_name" \
        --use_adapter "$use_adapter" \
        --peft_model_path "$peft_model_path" \
        >"$log_file_path"

    # 检查脚本是否成功执行
    if [ $? -eq 0 ]; then
        echo "Script executed successfully for model: $peft_model_name"
        echo "Log file: $log_file_path"
        echo "Result file: $result_file_path"
    else
        echo "Script failed to execute for model: $peft_model_name. Check the log file for details."
        echo "Log file: $log_file_path"
    fi

    # 分隔线用于日志的美观和调试
    echo "------------------------------------------------------------"

done
