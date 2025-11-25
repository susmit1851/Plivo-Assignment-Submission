import json

og_train = "data/train.jsonl"
new_train = "data/new_train.jsonl"
og_dev = "data/dev.jsonl"
new_dev = "data/new_dev.jsonl"

train = []
dev = []

with open(og_train, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        train.append(obj)

with open(new_train, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        train.append(obj)

with open(og_dev, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        dev.append(obj)

with open(new_dev, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        dev.append(obj)

with open("data/full_train.jsonl", "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")
with open("data/full_dev.jsonl", "w", encoding="utf-8") as f:
    for item in dev:
        f.write(json.dumps(item) + "\n")