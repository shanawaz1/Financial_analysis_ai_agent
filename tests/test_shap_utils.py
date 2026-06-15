import pytest
import numpy as np

from core.shap_utils import _build_prob_table, _aggregate_shap_tokens


class TestBuildProbTable:
    def test_basic(self):
        html = _build_prob_table(["A", "B"], [0.75, 0.25])
        assert "A" in html
        assert "B" in html
        assert "0.75" in html
        assert "0.25" in html
        assert "div" in html

    def test_empty(self):
        html = _build_prob_table([], [])
        assert "div" in html


class FakeExplanation:
    def __init__(self, data, values):
        self.data = data
        self.values = values


class TestAggregateShapTokens:
    def test_basic_aggregation(self):
        sv1 = FakeExplanation(["good", "bad", "stock"], np.array([0.5, -0.3, 0.2]))
        sv2 = FakeExplanation(["stock", "market"], np.array([-0.1, 0.4]))

        tokens, scores = _aggregate_shap_tokens([sv1, sv2])

        assert len(tokens) <= 5
        assert len(tokens) == len(scores)
        assert "stock" in tokens
        assert all(s >= 0 for s in scores)

    def test_single_explanation(self):
        sv = FakeExplanation(["good"], np.array([0.5]))
        tokens, scores = _aggregate_shap_tokens([sv])
        assert tokens == ("good",)
        assert scores == (0.5,)

    def test_empty_input(self):
        with pytest.raises(ValueError):
            _aggregate_shap_tokens([])

    def test_zero_values(self):
        sv = FakeExplanation(["a", "b"], np.array([0.0, 0.0]))
        tokens, scores = _aggregate_shap_tokens([sv])
        assert scores == (0.0, 0.0)
