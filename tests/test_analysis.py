import pytest
import numpy as np

from core.analysis import (
    split_in_sentences,
    make_spans,
    keyword_boost,
    fin_ner,
    speech_to_text,
    summarize_text,
    text_to_sentiment,
)


class TestSplitInSentences:
    def test_basic_split(self):
        sents = split_in_sentences("First sentence. Second sentence.")
        assert len(sents) == 2
        assert sents[0] == "First sentence."
        assert sents[1] == "Second sentence."

    def test_in_cache(self):
        sents = split_in_sentences("Cache test.")
        assert sents == ["Cache test."]
        sents2 = split_in_sentences("Cache test.")
        assert sents2 == sents

    def test_empty_text(self):
        assert split_in_sentences("") == []


class TestMakeSpans:
    def test_basic(self):
        results = [{"label": "positive"}, {"label": "negative"}]
        spans = make_spans("Good. Bad.", results)
        assert len(spans) == 2
        assert spans[0] == ("Good.", "positive")
        assert spans[1] == ("Bad.", "negative")


class TestKeywordBoost:
    def test_boost_fls_when_low(self):
        text = "We expect strong growth next quarter."
        probs = [0.8, 0.2]
        boosted = keyword_boost(text, probs, mode="fls")
        assert boosted[1] > 0.2
        assert boosted[0] < 0.8

    def test_no_boost_when_high(self):
        text = "We expect strong growth."
        probs = [0.3, 0.7]
        boosted = keyword_boost(text, probs, mode="fls")
        assert boosted == probs

    def test_no_keywords(self):
        text = "The company reported earnings."
        probs = [0.8, 0.2]
        boosted = keyword_boost(text, probs, mode="fls")
        assert boosted == probs

    def test_boost_sentiment(self):
        text = "We expect strong growth."
        probs = [0.2, 0.7, 0.1]
        boosted = keyword_boost(text, probs, mode="sentiment")
        assert boosted[0] > 0.2
        assert boosted[1] < 0.7

    def test_single_word_text(self):
        probs = [0.5, 0.5]
        assert keyword_boost("", probs) == probs

    def test_case_insensitive(self):
        text = "EXPECT improved revenue"
        probs = [0.9, 0.1]
        boosted = keyword_boost(text, probs, mode="fls")
        assert boosted[1] > 0.1


class TestFinNer:
    def test_company_detection(self):
        text = "Apple Inc. is based in Cupertino."
        result = fin_ner(text)
        labels = [label for _, label in result if label is not None]
        assert "COMPANY" in labels

    def test_location_detection(self):
        text = "Microsoft has offices in Redmond."
        result = fin_ner(text)
        labels = [label for _, label in result if label is not None]
        assert "LOCATION" in labels or "COMPANY" in labels

    def test_empty_text(self):
        assert fin_ner("") == []

    def test_no_entities(self):
        result = fin_ner("The quick brown fox jumps.")
        labels = [label for _, label in result if label is not None]
        assert len(labels) == 0


class TestSpeechToText:
    def test_none_input(self):
        result = speech_to_text(None)
        assert "Please upload" in result


class TestSummarizeText:
    def test_empty_input(self):
        result = summarize_text("")
        assert "No text" in result

    def test_whitespace_only(self):
        result = summarize_text("   ")
        assert "No text" in result


class TestTextToSentiment:
    def test_empty_input(self):
        result = text_to_sentiment("")
        assert "No text" in result
