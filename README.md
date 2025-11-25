# Plivo-Assignment-Submission
Susmit Neogi 22B0352 Plivo Assignment Submission

## Given Training Data
Epoch 1 average loss: 0.7689
Epoch 2 average loss: 0.0151
Epoch 3 average loss: 0.0090


## After Adding 700 train + 150 dev data (Keeping model same)
Epoch 1 average loss: 0.4637
Epoch 2 average loss: 0.0064
Epoch 3 average loss: 0.0037

#### For test dataset:
Per-entity metrics:
CITY            P=1.000 R=1.000 F1=1.000
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=0.200 R=0.200 F1=0.200
PERSON_NAME     P=0.812 R=0.975 F1=0.886
PHONE           P=0.298 R=0.350 F1=0.322

Macro-F1: 0.568

PII-only metrics: P=0.766 R=0.670 F1=0.715
Non-PII metrics: P=1.000 R=1.000 F1=1.000


Latency over 50 runs (batch_size=1):
  p50: 7.75 ms
  p95: 9.38 ms

Adding a minimum confidence value for positiv class prediction:
For current model, max F1 score obtained for min_conf = 0.7 which is 0.568






## Updates Scripts (based on new nomencalture of files)

1. Train:
```text
python src/train.py --model_name distilbert-base-uncased --train data/full_train.jsonl --dev data/full_dev.jsonl --out_dir out
```

2. Dev Set Prediction:
```text
python src/predict.py --model_dir out --input data/full_dev.jsonl --output out/dev_pred.json
```

3. Dev Set F1:
```text
python src/eval_span_f1.py \
--gold data/full_dev.jsonl \
--pred out/dev_pred.json
```

4. Stress Set Prediction:
```text
python src/predict.py \
--model_dir out \
--input data/stress.jsonl \
--output out/stress_pred.json
```

5. Stress Set F1 eval:
```text
python src/eval_span_f1.py \
--gold data/stress.jsonl \
--pred out/stress_pred.json
```

6. Latency:
```text
python src/measure_latency.py \
--model_dir out \
--input data/full_dev.jsonl \
--runs 50
```
