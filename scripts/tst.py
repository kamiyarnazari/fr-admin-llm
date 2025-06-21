import json

with open("data/faq_instruction_format.jsonl", "r", encoding="utf-8") as f:
    for i in range(5):
        print(json.loads(f.readline()))