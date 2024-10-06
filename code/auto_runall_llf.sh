#!/bin/bash

source ~/.bashrc

# export WORLD_SIZE=8  # 使用8个GPU进行分布式训练
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 定义 MULTIPLES 列表
MULTIPLES=(1 5 10 20 50 100 200 400)  # 修改这个列表，加入你希望测试的倍数

BASE_DIR="/data/zjy/databrew/code"
TEST_FILE_PATH="/data/zjy/databrew/data/in/streaming_test.csv"
BASE_MODEL="/data/zjy/databrew/models/Llama-2-7b-hf/"
LLAMA_FACTORY_DIR="/data/zjy/databrew/LLaMA-Factory"

# 设置是否需要筛选数据
NEED_FILTER=true
FILTER_STR="true"
if [ "$NEED_FILTER" = false ]; then
    FILTER_STR="false"
fi
# 获取当前日期和时间，记录开始时间
SCRIPT_START=$(date +%s)

for MULTIPLE in "${MULTIPLES[@]}"; do
    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    # 输出当前倍数信息
    echo "========== 正在处理 MULTIPLE = $MULTIPLE NEED_FLITER = $FILTER_STR=========="

    # Step 1: 数据合成
    echo "Step 1: 数据合成"
    cd "$BASE_DIR/data_generation"

    UNANNOTATED_TEXT_PATH="$(realpath ../../data/in/streaming_test.csv)"
    INPUT_FILENAME=$(basename -- "$UNANNOTATED_TEXT_PATH")
    SYNTHETIC_DATA_PATH="$(realpath ../../data/out/synthetic_${INPUT_FILENAME%.*}_${MULTIPLE}_con.json)"

    # 检查合成数据文件是否已经存在
    if [ -f "$SYNTHETIC_DATA_PATH" ]; then
        echo "数据文件 $SYNTHETIC_DATA_PATH 已存在，跳过数据合成步骤。"
    else
        # 使用 source 激活 conda 环境
        source /data/zjy/anaconda3/bin/activate bonito

        # 运行数据合成脚本
        STEP1_START=$(date +%s)
        echo "数据合成输出文件：$(realpath ../../experiments/bonito/bonito_${MULTIPLE}.out)"
        python bonito_sample.py \
            --unannotated_text_path "$UNANNOTATED_TEXT_PATH" \
            --multiple "$MULTIPLE" \
            --synthetic_data_path "$SYNTHETIC_DATA_PATH" > "$(realpath ../../experiments/bonito/bonito_${MULTIPLE}.out)"

        if [ $? -ne 0 ]; then
            echo "数据合成失败，请查看 $(realpath ../../experiments/bonito/bonito_${MULTIPLE}_$(date "+%Y%m%d_%H%M%S").out) 获取详细信息。"
            exit 1
        fi
        STEP1_END=$(date +%s)
    fi

    # Step 1.1: 数据筛选（可选）
    if [ "$NEED_FILTER" = true ]; then
        echo "Step 1.1: 筛选合成数据"
        FILTERED_DATA_PATH="$(realpath ../../data/out/filtered_${INPUT_FILENAME%.*}_${MULTIPLE}_con.json)"

        if [ -f "$FILTERED_DATA_PATH" ]; then
            echo "筛选后的数据文件 $FILTERED_DATA_PATH 已存在，跳过筛选步骤。"
        else
            # 使用 source 激活 conda 环境
            source /data/zjy/anaconda3/bin/activate bonito

            # 运行数据筛选脚本
            echo "数据筛选输出文件：$(realpath ../../experiments/bonito/bonito_select_${MULTIPLE}.out)"
            STEP1_1_START=$(date +%s)
            python bonito_select.py \
                --input_json_path "$SYNTHETIC_DATA_PATH" \
                --filtered_json_path "$FILTERED_DATA_PATH" \
                --model_path "../../models/Bonito_Mix_v0.2" > "$(realpath ../../experiments/bonito/bonito_select_${MULTIPLE}.out)"

            if [ $? -ne 0 ]; then
                echo "数据筛选失败，请查看 $(realpath ../../experiments/bonito/bonito_select_${MULTIPLE}_$(date "+%Y%m%d_%H%M%S").out) 获取详细信息。"
                exit 1
            fi
            STEP1_1_END=$(date +%s)
        fi

        SYNTHETIC_DATA_PATH="$FILTERED_DATA_PATH"
    fi

    # Step 1.2: 更新 dataset_info.json
    echo "Step 1.2: 更新 dataset_info.json"
    DATASET_NAME="synthetic_${INPUT_FILENAME%.*}_${MULTIPLE}_con"
    DATASET_INFO_FILE="$(realpath $LLAMA_FACTORY_DIR/data/dataset_info.json)"

    # 将新数据添加到 dataset_info.json
    if grep -q "\"$DATASET_NAME\"" "$DATASET_INFO_FILE"; then
        echo "数据集 $DATASET_NAME 已存在于 dataset_info.json 中，跳过更新。"
    else
        jq ". + {\"$DATASET_NAME\": {\"file_name\": \"$(basename $SYNTHETIC_DATA_PATH)\"}}" "$DATASET_INFO_FILE" > tmp.$$.json && mv tmp.$$.json "$DATASET_INFO_FILE"
        echo "数据集 $DATASET_NAME 已添加到 dataset_info.json 中。"
    fi

    # 将合成数据文件移动到 LLaMA-Factory 数据目录
    echo "Step 1.3: 移动数据文件到 LLaMA-Factory"
    cp "$SYNTHETIC_DATA_PATH" "$(realpath $LLAMA_FACTORY_DIR/data/)"

    # Step 2: 模型微调
    echo "Step 2: 模型微调"
    cd "$LLAMA_FACTORY_DIR"

    source /data/zjy/anaconda3/bin/activate llama_factory
    # 定义输出目录
    FINETUNE_OUTPUT_DIR=$(realpath "saves/llama2-7b/lora/sft/${DATASET_NAME}_$(date "+%Y%m%d_%H%M%S")")
    mkdir -p "$FINETUNE_OUTPUT_DIR"

    # 设置微调配置文件
    TRAIN_CONFIG_FILE="examples/train_lora/llama2_lora_sft_${DATASET_NAME}.yaml"
    cp examples/train_lora/llama2_lora_sft.yaml "$TRAIN_CONFIG_FILE"
    sed -i "s/dataset: alpaca_stqa_10/dataset: $DATASET_NAME/" "$TRAIN_CONFIG_FILE"
    sed -i "s|output_dir: saves/llama2-7b/lora/sft/stqa_10|output_dir: $FINETUNE_OUTPUT_DIR|" "$TRAIN_CONFIG_FILE"

    # 定义函数用于执行微调，失败时加倍 `gradient_accumulation_steps` 重试
    run_finetune() {
        local config_file=$1
        local output_file=$2
        local success=false
        while true; do
            echo "运行模型微调脚本：$(realpath config_file)"
            llamafactory-cli train "$config_file" >> "$output_file"

            if [ $? -eq 0 ]; then
                success=true
                break
            else
                echo "模型微调失败，正在修改配置文件并重新尝试。"
                # 获取当前 gradient_accumulation_steps 并加倍
                current_steps=$(grep "gradient_accumulation_steps" "$config_file" | awk '{print $2}')
                new_steps=$((current_steps * 2))
                sed -i "s/gradient_accumulation_steps: $current_steps/gradient_accumulation_steps: $new_steps/" "$config_file"
                echo "gradient_accumulation_steps 已加倍为 $new_steps，重新开始微调。"
            fi
        done

        $success
    }

    # 运行模型微调
    STEP2_START=$(date +%s)
    TRAIN_OUTPUT_FILE="$(realpath $BASE_DIR/../experiments/llama_factory/llama2_lora_sft_${DATASET_NAME}_$(date "+%Y%m%d_%H%M%S").log)"
    echo "模型微调输出文件：$TRAIN_OUTPUT_FILE"
    run_finetune "$TRAIN_CONFIG_FILE" "$TRAIN_OUTPUT_FILE"
    STEP2_END=$(date +%s)

    # Step 3: 推理测试
    echo "Step 3: 推理测试"
    cd "$BASE_DIR/inference_test"

    # 使用 source 激活 conda 环境
    source /data/zjy/anaconda3/bin/activate RAG

    # 获取当前日期和时间
    current_date_time=$(date "+%Y%m%d_%H%M%S")
    peft_model_path=$(realpath $LLAMA_FACTORY_DIR/$FINETUNE_OUTPUT_DIR)
    peft_model_name=$(basename -- "$peft_model_path")
    use_adapter=True

    if [ "$use_adapter" = True ]; then
        peft_model_name=$(basename -- "$peft_model_path")
    else
        peft_model_name="base"
    fi

    task_name="${peft_model_name}_inference"
    log_file_path="$(realpath ../../experiments/inference/infer_no_context_${current_date_time}_${peft_model_name}.log)"
    result_file_path="$(realpath ../../data/out/${peft_model_name}_res.csv)"

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

    # 定义函数将秒转换为 hh:mm:ss 格式
    format_time() {
        local total_seconds=$1
        local hours=$((total_seconds / 3600))
        local minutes=$(( (total_seconds % 3600) / 60 ))
        local seconds=$((total_seconds % 60))
        printf "%02d:%02d:%02d" $hours $minutes $seconds
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
