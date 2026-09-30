import pandas as pd

csv_path = "Six-CD/Datasets/SIX-CD/Nudity.csv"
df = pd.read_csv(csv_path)

print(f"Total benchmark prompts: {len(df)}")
print(df.head())