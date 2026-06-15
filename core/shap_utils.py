import io
import base64
from functools import lru_cache

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.analysis import keyword_boost
from core.config import FINBERT_LABELS, FLS_LABELS
from core.models import get_fin_model, get_fls_model


def finbert_predict(texts):
    clean_texts = []
    for t in texts:
        if isinstance(t, (list, np.ndarray)):
            clean_texts.append(" ".join(map(str, t)))
        else:
            clean_texts.append(str(t))
    outputs = get_fin_model()(clean_texts)
    prob_vectors = []
    for out in outputs:
        probs = [0.0, 0.0, 0.0]
        label = out["label"].lower()
        if label == "positive":
            probs[0] = out["score"]
        elif label == "negative":
            probs[1] = out["score"]
        else:
            probs[2] = out["score"]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        prob_vectors.append(probs)
    return np.array(prob_vectors)


def fls_predict(texts):
    clean_texts = []
    for t in texts:
        if isinstance(t, (list, np.ndarray)):
            clean_texts.append(" ".join(map(str, t)))
        else:
            clean_texts.append(str(t))
    outputs = get_fls_model()(clean_texts)
    prob_vectors = []
    for out in outputs:
        if out["label"].lower() == "forward-looking":
            prob_vectors.append([out["score"], 1 - out["score"]])
        else:
            prob_vectors.append([1 - out["score"], out["score"]])
    return np.array(prob_vectors)


def fls_predict_scalar_boosted(texts):
    raw_probs_batch = fls_predict(texts)
    boosted_scores = []
    for text, raw_probs in zip(texts, raw_probs_batch):
        boosted = keyword_boost(text, raw_probs, mode="fls")
        boosted_scores.append(boosted[1])
    return np.array(boosted_scores)


text_masker = shap.maskers.Text(" ")


@lru_cache(maxsize=1)
def get_finbert_explainer():
    return shap.Explainer(finbert_predict, text_masker)


@lru_cache(maxsize=1)
def get_fls_explainer():
    return shap.Explainer(fls_predict, text_masker)


@lru_cache(maxsize=1)
def get_fls_boosted_explainer():
    return shap.Explainer(fls_predict_scalar_boosted, text_masker)


def _build_prob_table(labels, probs):
    rows = "".join(
        f"<tr><td style='text-align:right;'>{label}</td>"
        f"<td style='text-align:right;'>{score:.2f}</td></tr>"
        for label, score in zip(labels, probs)
    )
    return f"""
    <div style='text-align:center;'>
    <table style='margin:auto; border-collapse: collapse; width:60%;'>
    <tr><th style='text-align:right;'>Sentiment</th><th style='text-align:right;'>Probability</th></tr>
    {rows}
    </table></div>"""


def explain_finbert(text):
    try:
        shap_values = get_finbert_explainer()([text])
        html_plot = shap.plots.text(shap_values[0], display=False)
        probs = finbert_predict([text])[0]
        prob_table = _build_prob_table(["Positive", "Negative", "Neutral"], probs)
        return f"<h3>Model Explanation</h3>{html_plot}<br><h3>Prediction Confidence</h3>{prob_table}"
    except Exception as e:
        return f"<p>Error generating SHAP explanation: {e}</p>"


def explain_fls_waterfall_plot(text):
    try:
        shap_values = get_fls_boosted_explainer()([text])
        raw_probs = fls_predict([text])[0]
        boosted_probs = keyword_boost(text, raw_probs, mode="fls")
        pred_class = np.argmax(boosted_probs)
        pred_label = FLS_LABELS[pred_class]

        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)

        prob_table = _build_prob_table(FLS_LABELS, boosted_probs)
        return f"""
        <h3>FLS SHAP Waterfall Plot (Predicted: {pred_label})</h3>
        <div style='text-align:center;'>
            <img src="data:image/png;base64,{img_str}" width="800"/>
        </div>
        <h3>Prediction Confidence</h3>{prob_table}
        """
    except Exception as e:
        return f"<p>Error generating FLS waterfall plot: {e}</p>"


def _aggregate_shap_tokens(shap_values):
    token_scores = {}
    for sv in shap_values:
        tokens = sv.data
        values = sv.values
        for token, val in zip(tokens, values):
            score = float(np.sum(np.abs(val))) if isinstance(val, np.ndarray) else abs(val)
            token_scores[str(token)] = token_scores.get(str(token), 0.0) + score
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    tokens, scores = zip(*sorted_tokens)
    return tokens, scores


def global_shap_summary(text_list):
    try:
        clean_texts = [str(t) for t in text_list if isinstance(t, str) and len(t.strip()) > 0]
        if not clean_texts:
            return "No valid input texts provided."

        shap_values = get_finbert_explainer()(clean_texts)
        tokens, scores = _aggregate_shap_tokens(shap_values)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(tokens[::-1], scores[::-1], color="#009a4d")
        ax.set_title("Top 15 Influential Tokens Across Financial Texts", fontsize=14)
        ax.set_xlabel("Aggregate SHAP Value (Absolute)", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        for bar, score in zip(bars, scores[::-1]):
            ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_str}" width="700"/>'
    except Exception as e:
        return f"<p>Error generating global SHAP summary: {e}</p>"


def global_fls_shap_summary(text_list):
    try:
        clean_texts = [str(t) for t in text_list if isinstance(t, str) and len(t.strip()) > 0]
        if not clean_texts:
            return "No valid input texts provided."

        shap_values = get_fls_explainer()(clean_texts)
        tokens, scores = _aggregate_shap_tokens(shap_values)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(tokens[::-1], scores[::-1], color="#0077b6")
        ax.set_title("Top 15 Influential Tokens for Forward-Looking Detection", fontsize=14)
        ax.set_xlabel("Aggregate SHAP Value (Absolute)", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        for bar, score in zip(bars, scores[::-1]):
            ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_str}" width="700"/>'
    except Exception as e:
        return f"<p>Error generating global FLS SHAP summary: {e}</p>"
