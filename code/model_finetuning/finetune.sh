export WORLD_SIZE=4

python finetune.py \
    --base_model '../../models/Llama-2-7b-hf/' \
    --data_path '../../data/out/synthetic_streaming_exqa_10.json' \
    --output_dir '../../models/stqa_lora_train' \
    --batch_size 512 \
    --micro_batch_size 16\
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    # --resume_from_checkpoint '../../models/stqa_lora_streaming_train/checkpoint-210/'