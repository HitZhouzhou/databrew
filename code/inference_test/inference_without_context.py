import os
import ast
import time
import csv
import json
import pandas as pd
import warnings
import tqdm
# 超参数
import argparse
# 计算分数函数
from score_compute import pre_recall_score
# 加速推理
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Optional, Tuple
import gc
# 设置使用的GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
from utils.prompter import Prompter

warnings.filterwarnings("ignore")
result_file_path = ''
prompts_file_path = ''
test_file_path = ''
top = 0
task = ''
use_adapter = False
peft_model_path = ''

def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    engine_args = EngineArgs(model="../../models/Llama-2-7b-hf",
                             tokenizer= '../../models/Llama-2-7b-hf',
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=128,
                             max_cpu_loras=2,
                             max_num_seqs=128,
                             tensor_parallel_size=8,
                             gpu_memory_utilization=0.8
                             )
    return LLMEngine.from_engine_args(engine_args)
def format_prompt():
    pass

def main(args):
    prompter = Prompter(args.prompt_template_name)
    test_data = pd.read_csv(test_file_path)
    sampling_params = SamplingParams(temperature=0)#,
                    # logprobs=1,
                    # prompt_logprobs=1,
                    # max_tokens=512,
                    # stop_token_ids=[32003])

    lora_request = LoRARequest("streaming_qa", 1, peft_model_path)
    prompts = []
    # 生成prompts
    test_data['full_prompt'] = test_data.apply(
        lambda row: prompter.generate_prompt(row['question'], '', ''),
        axis=1
    )
    for index, row in test_data.iterrows():
        prompts.append((row['full_prompt'], row['qa_id']))
    print('Start to process requests')
    engine = initialize_engine()
    request_outputs = []
    results = pd.DataFrame(columns=['id', 'llm_ans', 'ground_truths_list','precision','recall','f1'])
    time_start = time.time() 
    num_requests = len(prompts)
    finished_requests = 0
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt,id= prompts.pop(0)
            if(use_adapter):
                engine.add_request(id,
                                    prompt,
                                    sampling_params,
                                    lora_request=lora_request)
            else:
                engine.add_request(str(id),
                                    prompt,
                                    sampling_params)
        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                results = results._append({'id': request_output.request_id, 'llm_ans': request_output.outputs[0].text, 'ground_truths_list': test_data[test_data["qa_id"]==request_output.request_id]['answers'].values[0],'precison':0,'recall':0,'f1':0}, ignore_index=True)
                finished_requests += 1
                print("Finished request:", finished_requests, "/", num_requests)
                # 打印问题,llm回答,真实答案
                print("Question:\n", test_data[test_data["qa_id"]==request_output.request_id]['full_prompt'].values[0])
                print("LLM Answer:", request_output.outputs[0].text)
                print("Ground Truths:", test_data[test_data["qa_id"]==request_output.request_id]['answers'].values[0])
                print("-"*50)
    time_end = time.time()
    print('Process requests time cost:',time_end-time_start,'s\n') 
    results.to_csv(result_file_path,index=False)
    # 计算分数
    for index,row in results.iterrows():
        if (type(row['llm_ans']) is not str):
            row['llm_ans'] = ''
            
        row['ground_truths_list'] = row['ground_truths_list'].split('\\')
        precision ,recall ,f1 = pre_recall_score([row['ground_truths_list']], [row['llm_ans']])
        results.at[index, 'precision'] = precision
        results.at[index, 'recall'] = recall
        results.at[index, 'f1'] = f1
    results.to_csv(result_file_path,index=False) 
    # 打印平均分数
    # 打印当前任务名称
    print('Task:',task)
    print('Average Precision:',results['precision'].mean())
    print('Average Recall:',results['recall'].mean())
    print('Average F1:',results['f1'].mean())
    with open(f"../../experiments/{task}_results.log", 'w') as f:
        f.write(f"Task: {task}\n")
        f.write(f"Average Precision: {results['precision'].mean()}\n")
        f.write(f"Average Recall: {results['recall'].mean()}\n")
        f.write(f"Average F1: {results['f1'].mean()}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the model with a specified config file.")
    parser.add_argument('--result_file_path', type=str, help='Path to the result file')
    parser.add_argument('--test_file_path', type=str, help='Path to the test file')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--use_adapter', type=str, default=True, help='Whether to use adapter')
    parser.add_argument('--peft_model_path', type=str, help='Path to the peft model')
    parser.add_argument('--prompt_template_name',default="alpaca_short",type=str, help='Prompt template name')
     
    args = parser.parse_args()
    result_file_path = args.result_file_path
    test_file_path = args.test_file_path
    task = args.task
    use_adapter = (args.use_adapter=="True")
    peft_model_path = args.peft_model_path
    # 打印所有参数
    print('result_file_path:',result_file_path)
    print('test_file_path:',test_file_path)
    print('task:',task)
    print('use_adapter:',use_adapter)
    print('peft_model_path:',peft_model_path)
    
    main(args)

