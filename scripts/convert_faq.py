import json
import os


INPUT_PATH  = "data/faq_cleaned.json"
OUTPUT_PATH = "data/faq_instruction_format.jsonl"


if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Could not find input file at: {INPUT_PATH}")

# Open and parse the original JSON array 
with open(INPUT_PATH, "r", encoding="utf-8") as fin:
    try:
        faqs = json.load(fin)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from {INPUT_PATH}:\n"
            f"  {e.msg} at line {e.lineno} column {e.colno}"
        )

if not isinstance(faqs, list):
    raise RuntimeError(f"Expected a JSON array in {INPUT_PATH}, but got {type(faqs)}")

# Write out one line per FAQ in the new JSONL format
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for entry in faqs:
        question = entry.get("question", "").strip()
        answer   = entry.get("answer", "").strip()

        # Skip any entries that are missing question or answer
        if question == "" or answer == "":
            continue

        new_record = {
            "instruction": "Réponds à la question administrative suivante :",
            "input":       question,
            "output":      answer
        }
        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

print(f"Converted {len(faqs)} entries into: {OUTPUT_PATH}")
