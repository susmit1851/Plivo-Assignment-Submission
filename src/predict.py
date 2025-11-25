import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--min_span_conf", type=float, default=0.0, help="Minimum average token confidence for a span to be kept (0-1).")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                # compute softmax probs for confidence filtering
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            # conservative post-filtering to improve PII precision
            def is_digits_count(s, e, min_digits=8):
                sub = text[s:e]
                return sum(c.isdigit() for c in sub) >= min_digits

            NUM_WORDS = set(["zero","one","two","three","four","five","six","seven","eight","nine"])

            def is_spelled_digits(s, e, min_words=8):
                sub = text[s:e].lower().strip()
                words = [w for w in sub.split() if w]
                if len(words) < min_words:
                    return False
                return all(w in NUM_WORDS for w in words[:min_words])

            def looks_like_email(s, e):
                sub = text[s:e].lower()
                if "@" in sub:
                    return True
                if " at " in sub and " dot " in sub:
                    return True
                for d in ["gmail","yahoo","hotmail","outlook","protonmail","rediffmail"]:
                    if d in sub:
                        return True
                return False

            def looks_like_date(s, e):
                sub = text[s:e].lower()
                if any(ch.isdigit() for ch in sub) and ("/" in sub or "-" in sub or any(m in sub for m in ["january","february","march","april","may","june","july","august","september","october","november","december"])):
                    return True
                return False

            def looks_like_name(s, e):
                sub = text[s:e].strip()
                # prefer multi-token names or longer single tokens to reduce false positives
                if " " in sub:
                    return True
                return len(sub) >= 6

            ents = []
            for s, e, lab in spans:
                keep = True
                # compute average token confidence for this span
                # map char offsets to token indices via offsets
                token_confs = []
                for (tstart, tend), token_probs, tid in zip(offsets, probs, pred_ids):
                    if tstart == 0 and tend == 0:
                        continue
                    if tstart >= s and tend <= e:
                        # confidence for predicted label at this token
                        token_confs.append(token_probs[tid])
                span_conf = (sum(token_confs) / len(token_confs)) if token_confs else 0.0
                if span_conf < args.min_span_conf:
                    keep = False
                if lab == "CREDIT_CARD":
                    keep = is_digits_count(s, e, min_digits=12) or is_spelled_digits(s, e, min_words=12)
                elif lab == "PHONE":
                    keep = is_digits_count(s, e, min_digits=8) or is_spelled_digits(s, e, min_words=8)
                elif lab == "EMAIL":
                    keep = looks_like_email(s, e)
                elif lab == "PERSON_NAME":
                    keep = looks_like_name(s, e)
                elif lab == "DATE":
                    keep = looks_like_date(s, e)
                # CITY and LOCATION are non-PII; keep by default

                if keep:
                    ents.append(
                        {
                            "start": int(s),
                            "end": int(e),
                            "label": lab,
                            "pii": bool(label_is_pii(lab)),
                        }
                    )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
