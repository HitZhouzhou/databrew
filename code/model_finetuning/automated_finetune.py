import os
import subprocess
from datetime import datetime

def finetune_multiple_datasets(base_model, data_paths, output_base_dir, batch_size=512, micro_batch_size=16, lora_r=128, lora_alpha=32, lora_dropout=0.1, resume_from_checkpoint=None):
    """
    Automatically fine-tune a model on multiple datasets sequentially.

    Args:
    - base_model (str): Path to the base model directory.
    - data_paths (list of str): List of paths to the datasets for fine-tuning.
    - output_base_dir (str): Base directory for saving the output models.
    - batch_size (int): Batch size for training.
    - micro_batch_size (int): Micro batch size for training.
    - lora_r (int): LoRA rank parameter.
    - lora_alpha (int): LoRA alpha parameter.
    - lora_dropout (float): LoRA dropout rate.
    - resume_from_checkpoint (str or None): Path to the checkpoint to resume training from, or None.
    """
    for data_path in data_paths:
        # Generate output directory based on the data_path and current timestamp
        dataset_name = os.path.basename(data_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, f"{dataset_name}_finetune_{timestamp}")

        # Construct the command for fine-tuning
        command = [
            "python", "finetune.py",
            "--base_model", base_model,
            "--data_path", data_path,
            "--output_dir", output_dir,
            "--batch_size", str(batch_size),
            "--micro_batch_size", str(micro_batch_size),
            "--lora_r", str(lora_r),
            "--lora_alpha", str(lora_alpha),
            "--lora_dropout", str(lora_dropout)
        ]

        # Add resume from checkpoint if specified
        if resume_from_checkpoint:
            command.extend(["--resume_from_checkpoint", resume_from_checkpoint])

        # Print the command to be executed
        print(f"Executing command: {' '.join(command)}")

        # Run the command
        subprocess.run(command)

# Example usage
if __name__ == "__main__":
    base_model = '../../models/Llama-2-7b-hf/'
    data_paths = [
        '../../data/out/synthetic_streaming_exqa_10.json'
    ]
    output_base_dir = '../../models/'

    finetune_multiple_datasets(
        base_model=base_model,
        data_paths=data_paths,
        output_base_dir=output_base_dir,
        batch_size=1024,
        micro_batch_size=128,
        lora_r=128,
        lora_alpha=128,
        lora_dropout=0,
        resume_from_checkpoint=None
    )
