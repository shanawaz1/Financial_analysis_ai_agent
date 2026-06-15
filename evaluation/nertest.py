def eval_ner(data_jsonl: str, out_dir: str):
    import spacy, json
    from spacy.scorer import Scorer
    from spacy.training import Example
    os.makedirs(out_dir, exist_ok=True)
    nlp = spacy.load("en_core_web_lg")
    labels_kept = {"ORG","GPE","LOC"}

    gold_texts = []
    gold_ents = []
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gold_texts.append(obj["text"])
            ents = [tuple(e) for e in obj["entities"] if e[2] in labels_kept]
            gold_ents.append(ents)

    preds = []
    examples = []
    for text, ents in tqdm(list(zip(gold_texts, gold_ents)), desc="spaCy NER"):
        doc = nlp(text)
        pred_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in labels_kept]
        preds.append(pred_spans)
        # Build gold Doc for scoring
        doc_gold = nlp.make_doc(text)
        example = Example.from_dict(doc_gold, {"entities": ents})
        example.predicted = doc
        examples.append(example)

    scorer = Scorer()
    scores = scorer.score(examples)
    # Extract entity-level metrics
    ents_p = scores.get("ents_p", 0.0)
    ents_r = scores.get("ents_r", 0.0)
    ents_f = scores.get("ents_f", 0.0)
    per_type = scores.get("ents_per_type", {})

    # Save predictions
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for text, spans in zip(gold_texts, preds):
            f.write(json.dumps({"text": text, "pred_entities": spans})+"\n")

    # Save metrics
    metrics = {"micro": {"precision": ents_p, "recall": ents_r, "f1": ents_f}, "per_type": per_type}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

for gold, pred in zip(gold_ents, preds):
    print("Gold:", gold)
    print("Pred:", pred)