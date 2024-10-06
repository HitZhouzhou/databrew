import os
import json
import argparse
from bonito import Bonito
from vllm import SamplingParams
import pandas as pd
from format_question import extract_question
from datasets import Dataset

# 解析命令行参数
parser = argparse.ArgumentParser(description='Generate synthetic data using Bonito and save to JSON.')
parser.add_argument('--unannotated_text_path', type=str, required=True, help='Path to the input CSV file containing unannotated text.')
parser.add_argument('--multiple', type=int, required=True, help='The number of samples to generate per context (synthesis multiple).')
parser.add_argument('--synthetic_data_path', type=str, required=True, help='The path of the synthetic data.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
args = parser.parse_args()

# 获取命令行参数
unannotated_text_path = args.unannotated_text_path
multiple = args.multiple
synthetic_data_path = args.synthetic_data_path

# Initialize the Bonito model
bonito = Bonito("../../models/Bonito-test-v0.1", max_model_len=30000, tensor_parallel_size=8)

# 从CSV文件中读取文本数据
unannotated_text = pd.read_csv(unannotated_text_path)['text']

# 'input'列包含文本内容，'output'列是一个空列表
data = {'input': unannotated_text.tolist(), 'output': [[] for _ in range(len(unannotated_text))]}

# 使用字典创建一个Dataset对象
dataset = Dataset.from_dict(data)

# Function to generate synthetic data using Bonito
def generate_synthetic_data(bonito, dataset, multiple, seed):
    sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=multiple, seed=seed)
    synthetic_dataset = bonito.generate_tasks(
        dataset,
        context_col="input",
        task_type="exqa",
        sampling_params=sampling_params
    )
    return synthetic_dataset

# Generate synthetic data
if multiple > 200:
    # 分两次生成每次一半的数据
    half_multiple = multiple // 2

    synthetic_dataset_1 = generate_synthetic_data(bonito, dataset, half_multiple, args.seed)
    synthetic_dataset_2 = generate_synthetic_data(bonito, dataset, half_multiple, args.seed + 1)

    # 提取 'new_input' 和 'output' 列并交替合并
    json_data = []
    for item_1, item_2 in zip(synthetic_dataset_1, synthetic_dataset_2):
        # 从第一个数据集中提取
        new_input_1 = item_1['new_input']
        contest_1 = item_1['context']
        output_1 = item_1['output']
        instruction_1 = extract_question(new_input_1)
        json_data.append({
            "instruction": instruction_1,
            "context": contest_1,
            "input": "",
            "output": output_1
        })

        # 从第二个数据集中提取
        new_input_2 = item_2['new_input']
        contest_2 = item_2['context']
        output_2 = item_2['output']
        instruction_2 = extract_question(new_input_2)
        json_data.append({
            "instruction": instruction_2,
            "context": contest_2,
            "input": "",
            "output": output_2
        })

    # 如果数量是奇数，需要多处理一个
    if len(synthetic_dataset_1) > len(synthetic_dataset_2):
        extra_item = synthetic_dataset_1[-1]
        new_input_extra = extra_item['new_input']
        context_extra = extra_item['context']
        output_extra = extra_item['output']
        instruction_extra = extract_question(new_input_extra)
        json_data.append({
            "instruction": instruction_extra,
            "context": context_extra,
            "input": "",
            "output": output_extra
        })

else:
    # 如果 multiple 不大于 200，直接生成
    synthetic_dataset = generate_synthetic_data(bonito, dataset, multiple, args.seed)
    json_data = []

    for item in synthetic_dataset:
        new_input = item['new_input']
        context = item['context']
        output = item['output']
        instruction = extract_question(new_input)
        json_data.append({
            "instruction": instruction,
            "context": context,
            "input": "",
            "output": output
        })

# 将数据写入 JSON 文件
with open(synthetic_data_path, mode='w', encoding='utf-8') as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)

# 打印保存路径
print(f"Generated synthetic JSON file saved to: {synthetic_data_path}")
