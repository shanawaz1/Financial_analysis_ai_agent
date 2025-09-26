#!/usr/bin/env python
import argparse, os, json, csv, math
from typing import List, Dict, Tuple, Any
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Lazy imports inside functions to avoid importing heavy libs when not needed

def standardize_label(lbl: str) -> str:
    if lbl is None: return ""
    return str(lbl).strip().replace("LABEL_", "").replace("label_", "").lower()

def plot_confusion(cm, classes, out_png):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_prf_bars(report_dict, out_png):
    labels = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    p = [report_dict[k]["precision"] for k in labels]
    r = [report_dict[k]["recall"] for k in labels]
    f = [report_dict[k]["f1-score"] for k in labels]

    # Plot P/R/F as side-by-side bars
    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar([i - width for i in x], p, width, label="Precision")
    ax.bar(x, r, width, label="Recall")
    ax.bar([i + width for i in x], f, width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def eval_sentiment(data_csv: str, out_dir: str, device: int, save_raw: bool):
    from transformers import pipeline
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_csv, encoding='windows-1252')
    texts = df["text"].astype(str).tolist()
    gold = [standardize_label(x) for x in df["label"].tolist()]

    clf = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone", device=device)
    preds = []
    raw = []
    for t in tqdm(texts, desc="FinBERT sentiment"):
        out = clf(t)
        raw.append(out)
        preds.append(standardize_label(out[0]["label"]))

    classes = sorted(list(set(gold) | set(preds)))
    report = classification_report(gold, preds, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=classes)

    pd.DataFrame({"text":texts,"gold":gold,"pred":preds}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump(report, f, indent=2)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))
    plot_prf_bars(report, os.path.join(out_dir, "prf_bars.png"))
    if save_raw:
        with open(os.path.join(out_dir, "raw_outputs.jsonl"), "w") as f:
            for r in raw: f.write(json.dumps(r)+"\n")

def eval_asr(data_csv: str, out_dir: str, device: int, save_raw: bool, chunk_len=30, stride=5):
    """
    Evaluate ASR using the VoxTral-Mini-3B model.
    - data_csv: CSV with columns [audio_path, reference_text]
    - out_dir: folder where predictions + metrics are saved
    - device: -1 for CPU, 0+ for GPU
    - save_raw: whether to dump raw model outputs
    - chunk_len: audio chunk length in seconds
    - stride: overlap between chunks in seconds
    """
    import os
    import pandas as pd
    import json
    from tqdm import tqdm
    from jiwer import wer, cer
    from transformers import pipeline

    os.makedirs(out_dir, exist_ok=True)

    # Load and clean test data
    df = pd.read_csv(data_csv, encoding='windows-1252')
    paths = df["audio_path"].astype(str).apply(lambda p: p.strip().strip('"').strip("'")).tolist()
    refs = df["reference_text"].astype(str).tolist()

    # VoxTral-Mini-3B ASR pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model="fixie-ai/VoxTral-Mini-3B",
        chunk_length_s=chunk_len,
        stride_length_s=(stride, stride),
        device=device,
        return_timestamps=True
    )

    hyps = []
    raw = []

    for p in tqdm(paths, desc="VoxTral transcribe"):
        out = asr(p, return_timestamps=True)
        raw.append(out)

        if isinstance(out, dict) and "chunks" in out and isinstance(out["chunks"], list):
            text = " ".join([c.get("text", "") for c in out["chunks"]])
        else:
            text = out.get("text", "") if isinstance(out, dict) else ""
        hyps.append(text.strip())

    # Compute overall metrics
    overall_wer = wer(refs, hyps)
    overall_cer = cer(refs, hyps)

    # Per-file metrics
    per_file = []
    for r, h, p in zip(refs, hyps, paths):
        per_file.append({
            "audio_path": p,
            "wer": wer(r, h),
            "cer": cer(r, h),
        })

    # Save predictions
    pd.DataFrame({
        "audio_path": paths,
        "reference": refs,
        "hypothesis": hyps
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Save WER/CER results
    with open(os.path.join(out_dir, "wer.txt"), "w") as f:
        f.write(f"Overall WER: {overall_wer:.4f}\nOverall CER: {overall_cer:.4f}\n")
        for row in per_file:
            f.write(json.dumps(row) + "\n")

    # Optionally save raw model outputs
    if save_raw:
        with open(os.path.join(out_dir, "raw_outputs.jsonl"), "w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")

    print(f"✅ VoxTral-Mini-3B evaluation complete. Results saved to {out_dir}")
    print(f"Overall WER={overall_wer:.4f}, CER={overall_cer:.4f}")
    
def eval_ner(data_jsonl: str, out_dir: str):
    import os
    import json
    import spacy
    from tqdm import tqdm
    from spacy.scorer import Scorer
    from spacy.training import Example
    os.makedirs(out_dir, exist_ok=True)
    nlp = spacy.load("en_core_web_lg")
    labels_kept = {"ORG", "GPE", "LOC"}
    gold_texts = []
    gold_ents = []
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            raw_entities = obj["entities"]
            aligned_entities = align_entities(text, raw_entities, nlp)
            gold_texts.append(text)
            gold_ents.append(aligned_entities)


    preds = []
    examples = []

    for text, ents in tqdm(list(zip(gold_texts, gold_ents)), desc="spaCy NER"):
        doc = nlp(text)
        pred_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in labels_kept]
        preds.append(pred_spans)

        # Build gold Doc with validated spans
        doc_gold = nlp.make_doc(text)
        example = Example.from_dict(doc_gold, {"entities": ents})
        example.predicted = doc
        examples.append(example)

    # Score predictions
    scorer = Scorer()
    scores = scorer.score(examples)

    # Save predictions
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for text, spans in zip(gold_texts, preds):
            f.write(json.dumps({"text": text, "pred_entities": spans}) + "\n")

    # Save metrics
    metrics = {
        "micro": {
            "precision": scores.get("ents_p", 0.0),
            "recall": scores.get("ents_r", 0.0),
            "f1": scores.get("ents_f", 0.0)
        },
        "per_type": scores.get("ents_per_type", {})
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def eval_fls(data_csv: str, out_dir: str, device: int, save_raw: bool, peek_labels: bool):
    from transformers import pipeline
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_csv, encoding='windows-1252')
    texts = df["text"].astype(str).tolist()
    gold = [standardize_label(x) for x in df["label"].tolist()]

    clf = pipeline("text-classification", model="yiyanghkust/finbert-fls", tokenizer="yiyanghkust/finbert-fls", device=device)
    raw_label_set = set()
    preds = []
    raw = []

    for t in tqdm(texts, desc="FinBERT-FLS"):
        out = clf(t)
        raw.append(out)
        raw_label = out[0]["label"]
        raw_label_set.add(raw_label)
        preds.append(standardize_label(raw_label))

    if peek_labels:
        print("Model labels observed:", sorted(raw_label_set))

    classes = sorted(list(set(gold) | set(preds)))
    report = classification_report(gold, preds, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=classes)

    pd.DataFrame({"text":texts,"gold":gold,"pred":preds}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump(report, f, indent=2)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))
    plot_prf_bars(report, os.path.join(out_dir, "prf_bars.png"))
    if save_raw:
        with open(os.path.join(out_dir, "raw_outputs.jsonl"), "w") as f:
            for r in raw: f.write(json.dumps(r)+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment","asr","ner","fls","all"], required=True)
    parser.add_argument("--data", help="Path to data file (csv or jsonl, see README)")
    parser.add_argument("--out", required=True, help="Output folder for results")
    parser.add_argument("--device", type=int, default=-1, help="Transformers device id (-1 CPU, 0+ GPU)")
    parser.add_argument("--save-raw", action="store_true", help="Save raw model outputs")
    parser.add_argument("--peek-labels", action="store_true", help="(FLS) Print the label strings emitted by the model")
    args = parser.parse_args()

    if args.task in ("sentiment","fls","asr") and not args.data:
        raise SystemExit("--data is required for sentiment/asr/fls")

    if args.task == "sentiment":
        eval_sentiment(args.data, args.out, args.device, args.save_raw)
    elif args.task == "asr":
        eval_asr(args.data, args.out, args.device, args.save_raw)
    elif args.task == "ner":
        data_path = args.data or "data/ner.jsonl"
        eval_ner(data_path, args.out)
    elif args.task == "fls":
        eval_fls(args.data, args.out, args.device, args.save_raw, args.peek_labels)
    elif args.task == "all":
        os.makedirs(args.out, exist_ok=True)
        # Discover default data files if not provided
        eval_sentiment("data/sentiment.csv", os.path.join(args.out, "sentiment"), args.device, args.save_raw)
        eval_asr("data/asr.csv", os.path.join(args.out, "asr"), args.device, args.save_raw)
        eval_ner("data/ner.jsonl", os.path.join(args.out, "ner"))
        eval_fls("data/fls.csv", os.path.join(args.out, "fls"), args.device, args.save_raw, args.peek_labels)

if __name__ == "__main__":
    main()
