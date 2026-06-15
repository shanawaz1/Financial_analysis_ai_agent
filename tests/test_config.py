from core.config import (
    FINBERT_LABELS,
    FLS_LABELS,
    FLS_KEYWORDS,
    NER_LABEL_COLORS,
    ASR_MODEL,
    SUMMARIZER_MODEL,
    SENTIMENT_MODEL,
    FLS_MODEL,
    SPACY_MODEL,
    DEVICE,
)


def test_finbert_labels():
    assert FINBERT_LABELS == ["positive", "negative", "neutral"]
    assert len(FINBERT_LABELS) == 3


def test_fls_labels():
    assert FLS_LABELS == ["Not Forward-Looking", "Forward-Looking"]
    assert len(FLS_LABELS) == 2


def test_fls_keywords_not_empty():
    assert len(FLS_KEYWORDS) > 10
    assert "expect" in FLS_KEYWORDS
    assert "forward-looking" not in {k.lower() for k in FLS_KEYWORDS}


def test_ner_color_map():
    assert "COMPANY" in NER_LABEL_COLORS
    assert "LOCATION" in NER_LABEL_COLORS
    assert NER_LABEL_COLORS["COMPANY"] == "#7CFC00"


def test_default_models():
    assert ASR_MODEL == "openai/whisper-large-v3"
    assert SUMMARIZER_MODEL == "knkarthick/MEETING_SUMMARY"
    assert SENTIMENT_MODEL == "yiyanghkust/finbert-tone"
    assert FLS_MODEL == "yiyanghkust/finbert-fls"
    assert SPACY_MODEL == "en_core_web_sm"


def test_device_default():
    assert DEVICE == -1


def test_env_override(monkeypatch):
    monkeypatch.setenv("SPACY_MODEL", "en_core_web_trf")
    monkeypatch.setenv("DEVICE", "0")
    import importlib
    import core.config
    importlib.reload(core.config)
    assert core.config.SPACY_MODEL == "en_core_web_trf"
    assert core.config.DEVICE == 0
