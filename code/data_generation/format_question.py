import json
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
    elif question.find("The answer to the question:") != -1:
        question = question.split("The answer to the question:")[1].strip()
    elif question.find("Q:") != -1:
        question = question.split("Q:")[1].strip()
    elif question.find("What is the answer for the question:") != -1:
        question = question.split("What is the answer for the question:")[1].strip()
    elif question.find("The following article contains an answer for the question:") != -1:
        question = question.split("The following article contains an answer for the question:")[1].strip()
    elif question.find("Given the following context:") != -1:
        question = question.split("answer the following question:")[1].strip()
    elif question.find("With reference to the above context,") != -1:
        question = question.split("With reference to the above context,")[1].strip()
    elif question.find("Found the following article online, use it to answer the question:") != -1:
        question = question.split("Found the following article online, use it to answer the question:")[1].strip()
    elif question.find("I have a test where I am given the following article, what is an answer for the question:") != -1:
        question = question.split("I have a test where I am given the following article, what is an answer for the question:")[1].strip()
    else:
        # 如果没有匹配的模式，返回原始问题
        question = question.strip()

    return question

def main():
    # 读取 JSON 文件
    with open('../../data/out/synthetic_streaming_test_1.json', 'r') as f:
        data = json.load(f)

    # 对于每个字典对象，使用 extract_question 函数更新 instruction 字段
    for item in data:
        item['instruction'] = extract_question(item['instruction'])

    # 将更新后的数据写回到一个新的 JSON 文件
    with open('../../data/out/synthetic_streaming_test_1_1.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()