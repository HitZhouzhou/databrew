import pandas as pd
# df = pd.read_csv('../data/out/synthetic_streaming_exqa.csv')

def extract_question(question):
    if question.find("Given that {{context}} Therefore,") != -1:
        question = question.split("Given that {{context}} Therefore,")[1].strip()
    elif question.find("{{context}} Are we justified in saying that") != -1:
        question = question.replace("{{context}}","").strip()
    elif question.find("{{context}} Based on that information,") != -1:
        question = question.split("{{context}} Based on that information,")[1].strip()
    elif question.find("{{context}} Based on the previous passage,") != -1:
        question = question.split("{{context}} Based on the previous passage,")[1].strip()
    elif question.find("{{context}} Using only the above description and what you know about the world,") != -1:
        question = question.split("what you know about the world,")[1].strip()
    elif question.find("{{context}}\nQuestion:") != -1:
        question = question.split("{{context}}\nQuestion:")[1].strip()
    elif question.find("Suppose it's true that {{context}} Then,") != -1:
        question = question.split("Suppose it's true that {{context}} Then,")[1].strip()
    elif question.find("Take the following as truth: {{context}}\nThen ") != -1:
        question = question.split("Take the following as truth: {{context}}\nThen ")[1].strip()
    elif question.find("Suppose {{context}} Can we infer that") != -1:
        question = question.split("Suppose {{context}}")[1].strip()
    elif question.find("Given {{context}} Should we assume that") != -1:
        question = question.split("Given {{context}}")[1].strip()
    elif question.find("Assume it is true that {{context}} \n\nTherefore,") != -1:
        question = question.split("Assume it is true that {{context}} \n\nTherefore,")[1].strip()
    elif question.find("{{context}} \n\nKeeping in mind the above text, consider:") != -1:
        question = question.split("{{context}} \n\nKeeping in mind the above text, consider:")[1].strip()
    else:
        # 如果没有匹配的模式，返回原始问题
        question = question.strip()

    return question

# # 假设 df 是一个包含你的数据的 pandas DataFrame
# df['question'] = df['input'].apply(extract_question)
# df.to_csv("../data/out/synthetic_streaming_exqa.csv", index=False)
# print(df['question'])