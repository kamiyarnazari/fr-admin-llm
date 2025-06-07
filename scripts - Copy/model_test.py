from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig

# Path to the directory where you saved the fine-tuned model
model_path = "models/mistral-7b-finetuned"

# Load PEFT config
peft_config = PeftConfig.from_pretrained(model_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapters into the base model
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_fast=False)

# Instruction used during training
instruction = "Tu es un assistant administratif français. Réponds de façon claire et concise."

# Inference loop
while True:
    print("\n Pose ta question (ou tape 'exit' pour quitter):")
    user_question = input("> ").strip()
    if user_question.lower() == "exit":
        break
    
    prompt = f"{instruction}\n\n{user_question}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and clean output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nRéponse:\n")
    print(response.replace(prompt, "").strip())
