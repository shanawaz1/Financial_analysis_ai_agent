import pandas as pd

# JSON (dict or list)
df = pd.read_json("C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/results/ner/metrics.json")
print(df.head())

# JSONL (newline-delimited)
df = pd.read_json("C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/results/ner/predictions.jsonl", lines=True)
print(df.head())