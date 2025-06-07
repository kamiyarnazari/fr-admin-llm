#!/usr/bin/env python3
# train_mistral_finetune.py

import os
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


BASE_MODEL    = "mistralai/Mistral-7B-v0.1"  # or your local checkpoint
DATA_PATH     = "data/faq_instruction_format.jsonl"
OUTPUT_DIR    = "models/mistral-7b-finetuned"
BATCH_SIZE    = 1         # per device; often 1 or 2 for 7B
MICRO_BATCH   = 1         # gradient_accumulation if needed
EPOCHS        = 3
LEARNING_RATE = 3e-4
MAX_LENGTH    = 512       # total tokens (prompt + answer)
LORA_RANK     = 8         # LoRA rank
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
# ───────────────────────────────────────────────────────────────────────────────


def tokenize_and_mask(examples, tokenizer, max_length=512):
    """
    For each example, concatenate:
      full_text = instruction + "\n\n" + input + "\n\n" + output
    Tokenize full_text to length max_length.
    Then re-tokenize just (instruction + "\n\n" + input + "\n\n") to find prompt_len.
    Build labels so that tokens [0..prompt_len-1] = -100, tokens [prompt_len..] = actual token IDs.
    """
    input_texts  = []
    label_texts  = []
    instructs = examples["instruction"]
    questions = examples["input"]
    answers   = examples["output"]

    # Build lists of full_text and prompt_text
    for instr, ques, ans in zip(instructs, questions, answers):
        # Full text = instruction + question + answer
        full = f"{instr.strip()}\n\n{ques.strip()}\n\n{ans.strip()}\n"
        input_texts.append(full)
        # Tokenize Lable
        label_texts.append(full)

    # Tokenizing the entire (instruction+question+answer)
    tokenized_inputs = tokenizer(
        input_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Tokenizing again to get labels and then mask prompt tokens manually
    with tokenizer.as_target_tokenizer():
        tokenized_labels = tokenizer(
            label_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    input_ids      = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    labels_full    = tokenized_labels["input_ids"]

    batch_size = input_ids.shape[0]
    labels_masked = labels_full.clone()

    # For each sample, find prompt length and set labels[0:prompt_len] = -100 to be ingnored by tokenizer
    for i in range(batch_size):
        instr = instructs[i].strip()
        ques  = questions[i].strip()
        # Re-tokenize just “instruction + question” so we know how many tokens to mask
        prompt_text = f"{instr}\n\n{ques}\n\n"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # Mask out the prompt tokens
        labels_masked[i, :prompt_len] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels_masked,
    }


def main():
    # Loading tokenizer & base model (4-bit) + prepare for LoRA
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token 

    # Load base model in 4-bit with bitsandbytes (ensure accelerate + bitsandbytes installed)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=None,  # default 4-bit
    )

    # Prepare model for int4 + LoRA
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA (low-rank adapters)
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # typical for these LMs
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # should show only LoRA params

    # Loading dataset and apply tokenization & label-masking
    raw_dset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )

    # Remove any unused columns (just keep instruction, input, output)
    raw_dset = raw_dset.remove_columns(
        [c for c in raw_dset.column_names if c not in ["instruction", "input", "output"]]
    )

    # Apply our custom tokenize_and_mask function
    tokenized_dset = raw_dset.map(
        lambda examples: tokenize_and_mask(examples, tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=raw_dset.column_names
    )


    # TrainingArguments & Trainer
    total_train_batch = BATCH_SIZE * max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(tokenized_dset) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=MICRO_BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dset,
        args=training_args,
        tokenizer=tokenizer,
    )


    # Trainining
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    print(f"Fine-tuning complete. Model saved to `{OUTPUT_DIR}`.")

if __name__ == "__main__":
    main()
