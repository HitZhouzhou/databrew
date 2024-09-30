from collections import Counter
import string
import re
import pandas as pd
# 引入添加颜色的库
from colorama import Fore, Back, Style, init

def pre_recall_score(ground_truths_list, predictions):
    """
    计算预测结果的精确率(Precision)和召回率(Recall)得分。
    
    参数:
    ground_truths_list (list of list of str): 真实答案的列表，其中每个子列表包含一个样本的所有正确答案。
    predictions (list of str): 模型预测的答案列表。
    
    返回:
    tuple: 包含以下元素的元组
        float: 所有预测结果的平均精确率百分比。
        float: 所有预测结果的平均召回率百分比。
    """
    max_pres = []  # 存储每个预测的最大精确率
    max_recalls = []  # 存储每个预测的最大召回率
    if len(ground_truths_list) == 0 or len(predictions) == 0:
        return 0,0,0 
    for i in range(len(predictions)):
        prediction = predictions[i]  # 获取单个预测答案
        ground_truths = ground_truths_list[i]  # 获取对应的所有真实答案
        
        pres = []  # 存储当前预测的所有精确率值
        recalls = []  # 存储当前预测的所有召回率值
        
        for ground_truth in ground_truths:
            # 标准化预测和真实答案
            prediction_tokens = normalize_answer(str(prediction)).split()
            ground_truth_tokens = normalize_answer(str(ground_truth)).split()
            
            # 计算两个答案的交集
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())  # 公共元素的数量
            
            # 如果没有公共元素，则精确率和召回率都是0
            if num_same == 0:
                pres.append(0.0)
                recalls.append(0.0)
                continue
            
            # 计算精确率和召回率
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            
            pres.append(precision)
            recalls.append(recall)
        
        # 选取最大的精确率和召回率
        max_pres.append(max(pres))
        max_recalls.append(max(recalls))
    if len(max_pres) == 0 or len(max_recalls) == 0:
        return 0,0,0 
    # 计算平均精确率和召回率
    avg_precision = 100.0 * sum(max_pres) / len(max_pres)
    avg_recall = 100.0 * sum(max_recalls) / len(max_recalls)
    if(avg_precision + avg_recall == 0):
        f1 = 0
    else:
        f1 = f1_score(avg_precision, avg_recall)
    
    
    return avg_precision, avg_recall, f1

def normalize_answer(s):
    """
    标准化答案字符串，包括转换为小写、去除标点符号、去除文章以及多余的空白。
    
    参数:
    s (str): 原始答案字符串。
    
    返回:
    str: 标准化后的答案字符串。
    """
    def remove_articles(text):
        """
        移除文本中的文章(a, an, the)。
        """
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        """
        将多余的空白替换为一个空格。
        """
        return " ".join(text.split())

    def remove_punc(text):
        """
        移除文本中的标点符号。
        """
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """
        将文本转换为小写。
        """
        return text.lower()

    # 组合所有的标准化步骤
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(precision, recall):
    """
    计算给定精确率和召回率的F1得分。
    
    参数:
    precision (float): 精确率百分比。
    recall (float): 召回率百分比。
    
    返回:
    float: F1得分。
    """
    return 2 * (precision * recall) / (precision + recall)

def main():
    # 计算给定预测答案和真实答案的精确率和召回率
    RAG_results_path = './Data/Results/results830.csv'
    WithoutRAG_results_path = './Data/Results/results_withoutRAG_901.csv'

    RAG_results = pd.read_csv(RAG_results_path)
    RAG_predictions = RAG_results['llm_ans']
    RAG_ground_truths_list = RAG_results['ground_truths_list']

    WithoutRAG_results = pd.read_csv(WithoutRAG_results_path)
    WithoutRAG_predictions = WithoutRAG_results['llm_ans']
    WithoutRAG_ground_truths_list = WithoutRAG_results['ground_truths_list']

     
    # 计算得分
    RAG_precision, RAG_recall = pre_recall_score(RAG_ground_truths_list, RAG_predictions)
    
    WithoutRAG_precision, WithoutRAG_recall = pre_recall_score(WithoutRAG_ground_truths_list, WithoutRAG_predictions)
    
    # 打印结果
    print(Fore.GREEN+f"With RAG:Calculated Precision: {RAG_precision:.2f}%")
    print(Fore.GREEN+f"With RAG:Calculated Recall: {RAG_recall:.2f}%")
    print()
    print(Fore.GREEN+f"Without RAG:Calculated Precision: {WithoutRAG_precision:.2f}%")
    print(Fore.GREEN+f"Without RAG:Calculated Recall: {WithoutRAG_recall:.2f}%")

if __name__ == "__main__":
    main()