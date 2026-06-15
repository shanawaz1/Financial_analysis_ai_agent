from functools import lru_cache

import spacy
import transformers
from transformers import pipeline


@lru_cache(maxsize=1)
def get_nlp():
    transformers.logging.set_verbosity_error()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def get_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        chunk_length_s=30,
        stride_length_s=(5, 5),
        device=-1,
    )


@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="knkarthick/MEETING_SUMMARY")


@lru_cache(maxsize=1)
def get_fin_model():
    return pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone",
    )


@lru_cache(maxsize=1)
def get_fls_model():
    return pipeline(
        "text-classification",
        model="yiyanghkust/finbert-fls",
        tokenizer="yiyanghkust/finbert-fls",
    )
