import pandas as pd
import csv
import json
import random
DATA_PATH = '../../data/'
# CSV文件路径
input_csv_file = '../../data/out/synthetic_streaming_exqa.csv'
# JSON文件路径
output_json_file = input_csv_file.replace('.csv', '.json')

# 读取CSV文件
data = []
df = pd.read_csv(input_csv_file)
for index, row in df.iterrows():
    data.append({
        "instruction": row["question"],  # 留空
        "input": "",  # 从CSV的text列获取
        "output": row["answers"].split('\\')[0]  # 从CSV的question列获取
    })

# 将数据写入JSON文件
with open(output_json_file, mode='w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"CSV file has been converted to JSON and saved as {output_json_file}")