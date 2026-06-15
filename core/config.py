import os

ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "knkarthick/MEETING_SUMMARY")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "yiyanghkust/finbert-tone")
FLS_MODEL = os.getenv("FLS_MODEL", "yiyanghkust/finbert-fls")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
DEVICE = int(os.getenv("DEVICE", "-1"))

FINBERT_LABELS = ["positive", "negative", "neutral"]
FLS_LABELS = ["Not Forward-Looking", "Forward-Looking"]

FLS_KEYWORDS = [
    "expect", "expects", "expected", "forecast", "forecasts", "forecasted",
    "anticipate", "anticipates", "anticipated", "project", "projects", "projected",
    "guidance", "outlook", "plan", "plans", "planned",
    "will", "would", "going to", "aim", "aims", "aimed",
    "target", "targets", "targeted", "estimate", "estimates", "estimated",
    "future", "next quarter", "next year", "long-term", "short-term", "ahead",
    "prospect", "prospects", "opportunity", "opportunities",
    "intend", "intends", "intended", "strategy", "strategic",
    "goal", "objective", "potential", "may", "might", "could",
]

NER_LABEL_COLORS = {"COMPANY": "#7CFC00", "LOCATION": "#00BFFF"}
