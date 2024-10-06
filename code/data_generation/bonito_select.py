import os
import json
import argparse
import pandas as pd
import warnings
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from typing import List
import time
import re
import random
# 解析命令行参数
parser = argparse.ArgumentParser(description='Filter JSON data using Bonito model and save filtered data.')
parser.add_argument('--input_json_path', type=str, required=True, help='Path to the input JSON file containing the data to be filtered.')
parser.add_argument('--filtered_json_path', type=str, required=True, help='Path where the filtered JSON data will be saved.')
parser.add_argument('--model_path', type=str, default="../../models/Bonito_Mix_v0.2", help='Path to the Bonito model.')
args = parser.parse_args()

# 获取命令行参数
input_json_path = args.input_json_path
filtered_json_path = args.filtered_json_path
model_path = args.model_path
counter = {
    "ambiguous_num": 0,
    "other_num": 0
}
# 初始化 VLLM 模型引擎
def initialize_engine(model_path) -> LLMEngine:
    """Initialize the LLMEngine."""
    engine_args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=128,
        max_cpu_loras=2,
        max_num_seqs=512,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8
    )
    return LLMEngine.from_engine_args(engine_args)
def should_keep_output(answer):
    # 预定义表示肯定、否定和模棱两可的输出模式
    positive_patterns = [
        r"yes, the answer is correct",  # "yes, the answer is correct"
        r"is correct",  # "answer is correct"
        r"the provided answer is correct",  # "the provided answer is correct"
        r"\byes\b",  # "yes"
        r"judgement:\nyes",
        
    ]

    negative_patterns = [
        r"no answer",  # "no answer"
        r"incorrect",  # "incorrect"
        r"the answer is incorrect",  # "the answer is incorrect"
        r"not correct",  # "not correct"
        r"no, that's not correct",
        r"judgement:\nno",
        r"the answer is wrong",
    ]

    ambiguous_patterns = [
        r"correct or incorrect",  # "correct or incorrect"
        r"yes, no, or maybe",  # "yes, no, or maybe"
        r"what is the name of",  # "what is the name of the tight end"
        r".*?\?",  # 匹配以问号结尾的句子，表示模棱两可
        r"is this answer correct?",
        r"\bwhat is",
    ]

    for pattern in negative_patterns:
        if re.search(pattern, answer):
            return False  # 直接抛弃
    
    for pattern in positive_patterns:
        if re.search(pattern, answer):
            return True  # 保留下来

    # 处理模棱两可的情况，80% 概率保留
    for pattern in ambiguous_patterns:
        if re.search(pattern, answer):
            counter["ambiguous_num"] += 1
            return random.uniform(0,1)<0.8  # 随机决定是否保留
    counter["other_num"] += 1
    return random.uniform(0,1)<0.5 # 默认保留


# 主函数
def main():
    # 读取输入 JSON 数据
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化引擎和采样参数
    engine = initialize_engine(model_path)
    sampling_params = SamplingParams(temperature=0,max_tokens=1000)
    # 生成 prompts 列表
    prompts = []
    for index, item in enumerate(data):
        prompt = f"<|tasktype|>\nanswer evaluation\n<|context|>\n{item['context']}\n<|task|>\nQuestion:{item['instruction']}\nAnswer:{item['output']}\n<|task|>\n"
        prompts.append((prompt, index))

    # 执行推理
    print('Start to process requests...')
    filtered_data = []
    time_start = time.time()
    num_requests = len(prompts)
    finished_requests = 0

    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, idx = prompts.pop(0)
            engine.add_request(str(idx), prompt, sampling_params)

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            # 在请求完成后进行处理
            if request_output.finished:
                answer = request_output.outputs[0].text.strip().lower()
                
                if should_keep_output(answer):
                    filtered_data.append(data[int(request_output.request_id)])
                print(f"Answer:\n{answer}\nshould_keep_output: {should_keep_output(answer)}\n")
                finished_requests += 1
                print(f"Finished request: {finished_requests}/{num_requests}")


    time_end = time.time()
    print('Process requests time cost:', time_end - time_start, 's\n')
    print("总数据量:", len(data))
    print("保留的百分比:", len(filtered_data) / len(data))
    print("模棱两可百分比:", counter["ambiguous_num"]/len(data))
    print("无法判断百分比:", counter["other_num"]/len(data))
    # 将过滤后的数据保存为新的 JSON 文件
    with open(filtered_json_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, indent=4, ensure_ascii=False)
    
    print(f"Filtered JSON file saved to: {filtered_json_path}")

if __name__ == '__main__':
    main()
