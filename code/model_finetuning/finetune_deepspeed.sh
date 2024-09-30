export WORLD_SIZE=8  # 使用8个GPU进行分布式训练
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed finetune_deepspeed_.py \
    --base_model '../../models/Llama-2-7b-hf/' \
    --output_dir '../../models/stqa_lora_synthetic_50_finetune' \
    --batch_size 512 \
    --micro_batch_size 32 \
    --lora_r 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path '../../data/out/synthetic_streaming_test_50.json' \
    --resume_from_checkpoint '/data/zjy/databrew/models/stqa_lora_synthetic_50_finetune_20240930_135121/checkpoint-486' \
    --num_epochs 1 \
    --deepspeed 'deepspeed_config.json' 