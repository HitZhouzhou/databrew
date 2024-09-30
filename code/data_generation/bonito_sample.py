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

# 设置采样参数，指定 n 的值来生成多个样本
sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=multiple)

# 生成合成微调数据集
synthetic_dataset = bonito.generate_tasks(
    dataset,
    context_col="input",
    task_type="exqa",
    sampling_params=sampling_params
)

# 提取 'new_input' 和 'output' 列
json_data = []

for item in synthetic_dataset:
    new_input = item['new_input']
    output = item['output']

    instruction = extract_question(new_input)
    json_data.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })

# # 将数据写入 JSON 文件
with open(synthetic_data_path, mode='w', encoding='utf-8') as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)
# 打印保存路径
print(f"Generated synthetic JSON file saved to: {synthetic_data_path}")
