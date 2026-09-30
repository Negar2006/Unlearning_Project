import pandas as pd
import json

CSV_PATH = "Six-CD/Datasets/SIX-CD/Nudity.csv"
OUT_PATH = "data/evaluation_set.json"

df = pd.read_csv(CSV_PATH)

NUM_SAMPLES = 50
eval_subset = df.iloc[:NUM_SAMPLES]

eval_data = []
for _, row in eval_subset.iterrows():
    eval_data.append({
        "case_number": int(row["case_number"]),
        "evaluation_seed": int(row["evaluation_seed"]),
        "prompt": str(row["prompt"])
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)

print(f"Evaluation set with {len(eval_data)} prompts successfully saved to '{OUT_PATH}'.")