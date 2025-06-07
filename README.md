# 🧠 fr-admin-llm

> Fine-tuned LLM for answering French administrative questions

This repository contains the training pipeline and evaluation code for a domain-specific large language model (LLM) fine-tuned on frequently asked questions (FAQs) related to French administration. The model is designed to serve as a fallback for the [`fr-admin-chatbot`](https://github.com/kamiyarnazari/fr-admin-chatbot) project when retrieval-based answers are not satisfactory.

---

## 💡 Objective

To train a lightweight, instruction-tuned LLM that can answer common French administrative questions by:

- Fine-tuning a base LLM using a custom dataset of 618 unique question–answer pairs
- Evaluating the model against baselines (e.g. `mistral-7b-instruct`, `zephyr-7b-beta`)
- Integrating the best-performing model as a fallback module in the chatbot system

---

## 🚀 How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (QLoRA)**

   ```bash
   python scripts/train.py \
     --model_name mistral-7b-instruct \
     --dataset_path data/faqs.json \
     --output_dir models/mistral-7b-finetuned \
     --use_lora
   ```

3. **Evaluate the model**

   ```bash
   python scripts/evaluate.py \
     --model_dir models/mistral-7b-finetuned \
     --dataset_path data/faqs.json \
     --baseline_models mistral-7b-instruct zephyr-7b-beta \
     --output_csv comparison_results.csv
   ```

---

## 📊 Sample Results

| Model                | Rouge-L | BLEU  | F1    |
|---------------------|---------|-------|-------|
| mistral-7b-instruct | 62.1    | 41.3  | 58.7  |
| zephyr-7b-beta       | 64.5    | 43.2  | 60.4  |
| **fr-admin-llm**     | **75.3**| **56.7**| **73.9** |

✅ The fine-tuned `fr-admin-llm` model outperforms base models on domain-specific tasks.

---

## 🧩 Integration

This model is used in the [`fr-admin-chatbot`](https://github.com/kamiyarnazari/fr-admin-chatbot) project as a fallback response generator when semantic retrieval confidence is low.

---

## 📎 Notes

- Training was done on a single GPU using QLoRA for efficiency.
- The dataset is fully custom, composed of real-world French administrative FAQs.

