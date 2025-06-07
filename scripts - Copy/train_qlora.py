from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# Set Model and Config
model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

# Load and Tokenize data
dataset = load_dataset("json", data_files="data/faq_instruction_format.jsonl")["train"]
dataset = dataset.train_test_split(test_size=0.1)

# Formatting prompt for instruction tuning
def format_prompt(example):
    return f"{example['instruction'].strip()}\n\n{example['input'].strip()}\n\n"

def tokenize(examples):
    inputs = []
    labels = []

    for instruction, question, answer in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        prompt = f"{instruction.strip()}\n\n{question.strip()}\n\n"
        input_ids = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
        )
        label_ids = tokenizer(
        answer,
        truncation=True,
        padding="max_length",
        max_length=128
        )

        input_ids["labels"] = label_ids["input_ids"]
        inputs.append(input_ids)

    batch = {k: [dic[k] for dic in inputs] for k in inputs[0]}
    return batch


# Apply tokenizer
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names, batched=True)

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

)

# Adding LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collactor for auto padding batches
data_collactor = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False                           #casual LM, not masked LM
)

# Training Configuration
training_args = TrainingArguments(
    output_dir="models/mistral-7b-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    report_to="none"
)

# Initializing the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collactor
)

# start fine tuning the model
trainer.train()

# Save the final model
trainer.save_model("models/mistral-7b-finetuned")
tokenizer.save_pretrained("models/mistral-7b-finetuned")