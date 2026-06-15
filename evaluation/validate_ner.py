import json
import sys

def validate_ner(jsonl_path):
    print(f"Validating: {jsonl_path}")
    bad = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                bad.append((i, "json_error", str(e)))
                continue

            text = obj.get("text", "")
            ents = obj.get("entities", [])
            for ent in ents:
                try:
                    start, end, label = ent
                except Exception:
                    bad.append((i, "bad_format", ent))
                    continue

                # Check offsets
                if not (0 <= start < end <= len(text)):
                    bad.append((i, "out_of_bounds", ent))
                    continue

                substring = text[start:end]
                # Warn if substring has mismatched length or looks off
                if not substring.strip():
                    bad.append((i, "empty_span", ent))
                else:
                    # Extra heuristic: if label is ORG/GPE/LOC but substring contains space at ends
                    if substring != substring.strip():
                        bad.append((i, "whitespace_span", (substring, ent)))
    if not bad:
        print("✅ All entity spans look valid!")
    else:
        print(f"⚠ Found {len(bad)} issues:")
        for i, kind, info in bad:
            print(f" Line {i}: {kind} -> {info}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ner.py data/ner.jsonl")
    else:
        validate_ner(sys.argv[1])
