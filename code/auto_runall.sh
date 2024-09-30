#!/bin/bash

source ~/.bashrc

# export WORLD_SIZE=8  # 使用8个GPU进行分布式训练
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 定义 MULTIPLES 列表
MULTIPLES=(50 100 200)  # 修改这个列表，加入你希望测试的倍数
TEST_FILE_PATH="../../data/in/streaming_test.csv"
BASE_MODEL="../../models/Llama-2-7b-hf/"
# 获取当前日期和时间，记录开始时间
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
SCRIPT_START=$(date +%s)

for MULTIPLE in "${MULTIPLES[@]}"; do
    # 输出当前倍数信息
    echo "========== 正在处理 MULTIPLE = $MULTIPLE =========="

    # Step 1: 数据合成
    echo "Step 1: 数据合成"
    cd ./data_generation

    UNANNOTATED_TEXT_PATH="../../data/in/streaming_test.csv"
    INPUT_FILENAME=$(basename -- "$UNANNOTATED_TEXT_PATH")
    SYNTHETIC_DATA_PATH="../../data/out/synthetic_${INPUT_FILENAME%.*}_${MULTIPLE}.json"

    # 检查合成数据文件是否已经存在
    if [ -f "$SYNTHETIC_DATA_PATH" ]; then
        echo "数据文件 $SYNTHETIC_DATA_PATH 已存在，跳过数据合成步骤。"
    else
        # 使用 source 激活 conda 环境
        source /data/zjy/anaconda3/bin/activate bonito

        # 运行数据合成脚本
        STEP1_START=$(date +%s)
        python bonito_sample.py \
            --unannotated_text_path "$UNANNOTATED_TEXT_PATH" \
            --multiple "$MULTIPLE" \
            --synthetic_data_path "$SYNTHETIC_DATA_PATH" >> ../../experiments/bonito/bonito_${MULTIPLE}.out

        if [ $? -ne 0 ]; then
            echo "数据合成失败，请查看 ../../experiments/bonito/bonito_${MULTIPLE}_$(date "+%Y%m%d_%H%M%S").out 获取详细信息。"
            exit 1
        fi
        STEP1_END=$(date +%s)
    fi

    cd ..

    # Step 2: 模型微调
    echo "Step 2: 模型微调"
    cd ./model_finetuning

    # 使用 source 激活 conda 环境
    source /data/zjy/anaconda3/bin/activate finetune

    # 运行模型微调脚本
    FINETUNE_OUTPUT_DIR="../../models/stqa_lora_synthetic_${MULTIPLE}_finetune_$(date "+%Y%m%d_%H%M%S")"
    mkdir -p "$FINETUNE_OUTPUT_DIR"

    STEP2_START=$(date +%s)
    python finetune.py \
        --base_model "$BASE_MODEL" \
        --data_path "$SYNTHETIC_DATA_PATH" \
        --output_dir "$FINETUNE_OUTPUT_DIR"  \
        --batch_size 1024 \
        --micro_batch_size 128 \
        --lora_r 128 \
        --lora_alpha 128 \
        --lora_dropout 0 \
        >> ../../experiments/finetune/finetune_${MULTIPLE}.log

    if [ $? -ne 0 ]; then
        echo "模型微调失败，请查看 ../../experiments/finetune/finetune_${MULTIPLE}_$(date "+%Y%m%d_%H%M%S").log 获取详细信息。"
        exit 1
    fi
    STEP2_END=$(date +%s)

    cd ..

    # Step 3: 推理测试
    echo "Step 3: 推理测试"
    cd ./inference_test

    # 使用 source 激活 conda 环境
    source /data/zjy/anaconda3/bin/activate RAG

    # 获取当前日期和时间
    current_date_time=$(date "+%Y%m%d_%H%M%S")
    peft_model_path="$FINETUNE_OUTPUT_DIR"
    peft_model_name=$(basename -- "$peft_model_path")
    use_adapter=True

    if [ "$use_adapter" = True ]; then
        peft_model_name=$(basename -- "$peft_model_path")
    else
        peft_model_name="base"
    fi

    task_name="${peft_model_name}_inference"
    log_file_path="../../experiments/inference/infer_no_context_${current_date_time}_${peft_model_name}.log"
    result_file_path="../../data/out/${peft_model_name}_res.csv"

    echo "Running inference for model: $peft_model_name"
    echo "Log file: $log_file_path"
    echo "Result file: $result_file_path"

    STEP3_START=$(date +%s)
    python inference_without_context.py \
        --result_file_path "$result_file_path" \
        --test_file_path "$TEST_FILE_PATH" \
        --task "$task_name" \
        --use_adapter "$use_adapter" \
        --peft_model_path "$peft_model_path" \
        > "$log_file_path"

    if [ $? -ne 0 ]; then
        echo "推理测试失败，请查看 $log_file_path 获取详细信息。"
        exit 1
    fi
    STEP3_END=$(date +%s)

    # 计算每个步骤的执行时间（秒）
    STEP1_TIME=$((STEP1_END - STEP1_START))
    STEP2_TIME=$((STEP2_END - STEP2_START))
    STEP3_TIME=$((STEP3_END - STEP3_START))
    TOTAL_TIME=$((STEP1_TIME + STEP2_TIME + STEP3_TIME))

    # 定义函数将秒转换为 xx min xx s 格式
    format_time() {
        local seconds=$1
        local minutes=$((seconds / 60))
        local remaining_seconds=$((seconds % 60))
        echo "${minutes} min ${remaining_seconds} s"
    }

    # 格式化每个步骤的时间
    STEP1_TIME_FORMATTED=$(format_time $STEP1_TIME)
    STEP2_TIME_FORMATTED=$(format_time $STEP2_TIME)
    STEP3_TIME_FORMATTED=$(format_time $STEP3_TIME)
    TOTAL_TIME_FORMATTED=$(format_time $TOTAL_TIME)
    result_log="../../experiments/${task_name}_results.log"
    # 将所有结果和时间写入总的日志文件
    FINAL_LOG_FILE="../../experiments/final_summary_${current_date_time}_MULTIPLE_${MULTIPLE}.log"

    {
      echo "任务名：$task_name"
      echo "任务开始执行时间：$START_TIME"
      echo "数据合成时间：$STEP1_TIME_FORMATTED"
      echo "模型微调时间：$STEP2_TIME_FORMATTED"
      echo "推理测试时间：$STEP3_TIME_FORMATTED"
      echo "总执行时间：$TOTAL_TIME_FORMATTED"
      echo
      echo "推理测试结果："
      cat "$result_log"
    } > "$FINAL_LOG_FILE"

    echo "任务名：$task_name"
    echo "任务开始执行时间：$START_TIME"
    echo "数据合成时间：$STEP1_TIME_FORMATTED"
    echo "模型微调时间：$STEP2_TIME_FORMATTED"
    echo "推理测试时间：$STEP3_TIME_FORMATTED"
    echo "总执行时间：$TOTAL_TIME_FORMATTED"
    echo
    echo "推理测试结果："
    cat "$result_log"

    echo "所有步骤完成，查看 log/ 目录中的输出文件以获取详细信息。"
    echo "最终结果文件保存为：$FINAL_LOG_FILE"

    cd ..

done

# 计算整个脚本的总执行时间
SCRIPT_END=$(date +%s)
SCRIPT_TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))
SCRIPT_TOTAL_TIME_FORMATTED=$(format_time $SCRIPT_TOTAL_TIME)

echo "整个脚本的总执行时间：$SCRIPT_TOTAL_TIME_FORMATTED"
