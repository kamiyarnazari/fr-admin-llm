import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch
import evaluate

# ---------------------------
# Load Metrics
# ---------------------------
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
f1_metric = evaluate.load("f1")

# ---------------------------
# Load Test Data
# ---------------------------
def load_test_data(path, limit=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            data.append({
                "prompt": item["instruction"],
                "reference": item["output"]
            })
    return data

# ---------------------------
# Load Pipeline (batched)
# ---------------------------
def load_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---------------------------
# Evaluate Model (batched)
# ---------------------------
def evaluate_model_batched(model_name, pipe, data, batch_size=8):
    predictions = []
    references = []
    prompts = [sample["prompt"] for sample in data]
    refs = [sample["reference"] for sample in data]

    print(f"\n🔁 Generating predictions in batches of {batch_size}...")

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating {model_name}"):
        batch_prompts = prompts[i:i + batch_size]
        batch_refs = refs[i:i + batch_size]

        batch_outputs = pipe(batch_prompts, max_new_tokens=256, do_sample=False, batch_size=batch_size)

        for prompt, output in zip(batch_prompts, batch_outputs):
            # Remove the prompt from generated text if it's included
            generated = output[0]["generated_text"].replace(prompt, "").strip()
            predictions.append(generated)

        references.extend(batch_refs)

    # Compute metrics
    rouge_score = rouge.compute(predictions=predictions, references=references)["rougeL"]
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    f1_score = f1_metric.compute(predictions=predictions, references=references)["f1"]

    return {
        "Rouge-L": round(rouge_score * 100, 1),
        "BLEU": round(bleu_score * 100, 1),
        "F1": round(f1_score * 100, 1)
    }

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    test_file = "data/faq_instruction_format.jsonl"
    test_data = load_test_data(test_file, limit=None)  # or limit=20 for testing

    model_paths = {
        "mistral-7b-instruct": "mistralai/Mistral-7B-v0.1",
        "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
        "fr-admin-llm": "models/mistral-7b-finetuned"  # ⬅ Update path as needed
    }

    results = {}

    for model_name, model_path in model_paths.items():
        print(f"\n🔍 Loading {model_name}...")
        pipe = load_pipeline(model_path)
        metrics = evaluate_model_batched(model_name, pipe, test_data, batch_size=8)
        results[model_name] = metrics

    # Display results
    print("\n📊 Evaluation Results")
    print(f"{'Model':<22} {'Rouge-L':<8} {'BLEU':<6} {'F1':<6}")
    print("-" * 40)
    for model, scores in results.items():
        print(f"{model:<22} {scores['Rouge-L']:<8} {scores['BLEU']:<6} {scores['F1']:<6}")
