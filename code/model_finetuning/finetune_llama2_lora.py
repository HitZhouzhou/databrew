from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 加载已经微调过的 LoRA 模型和 tokenizer
model_name_or_path = "../../models/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 加载问答数据集
dataset = load_dataset("your_question_answering_dataset_script", "your_config")

# 将数据集转换为模型可接受的格式
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(questions, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    for i, label in enumerate(examples["answers"]):
        start_index = questions[i].find(label)  # 找到答案的起始位置
        end_index = start_index + len(label)  # 找到答案的结束位置
        inputs["labels"][i][start_index:end_index+1] = [-100]  # 将答案位置的标签设置为 -100
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义 Trainer 的训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 进行微调
trainer.train()

# 保存微调后的模型
trainer.save_model("path/to/save/finetuned_model")