
import pandas as pd, os
data_csv = "C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/data/asr.csv"
clean_path = "data/asr_cleaned.csv"
with open(data_csv, "r", encoding="windows-1252", errors="replace") as infile, \
     open(clean_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        outfile.write(line)

df = pd.read_csv(clean_path)