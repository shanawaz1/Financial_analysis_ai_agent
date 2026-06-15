import pandas as pd, os
data_csv = "C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/data/asr.csv"
with open(data_csv, 'rb') as f:
    raw_bytes = f.read(1500)
    print(raw_bytes[:100])  # Preview first 100 bytes

csv_path = os.path.join(
    "C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/data",
    "asr.csv"
)

# Read with safe encoding
asr = pd.read_csv(csv_path, encoding='latin1')
with open(data_csv, 'rb') as f:
    print(f.read(1500))  # Preview first 1500 bytes

# Strip quotes and whitespace from each path
clean_paths = asr['audio_path'].apply(lambda p: str(p).strip().strip('"').strip("'"))

# Check for missing files
missing = [p for p in clean_paths if not os.path.exists(p)]
print("Audio files missing:", len(missing))
if missing:
    print(missing[:10])

import json, os
ner_path = os.path.join("C:/Users/shana/source/repos/face-attendance-system/AI agent/Financial_Analyst_AI/evaluation/data", "ner.jsonl")
#if os.path.exists(ner_path):
    #with open(ner_path) as f:
        #for i, line in enumerate(f):
            #obj = json.loads(line)
           # #print(i, obj.get("text")[:50], " -> ", obj.get("entities")[:3])
            #if i>5: break