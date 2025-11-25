# Plivo-Assignment-Submission
Susmit Neogi 22B0352 Plivo Assignment Submission

## Given Training Data
```text
Epoch 1 average loss: 0.7689
Epoch 2 average loss: 0.0151
Epoch 3 average loss: 0.0090
```

## After Adding 700 train + 150 dev data (Keeping model same)
We use the following scripts to generates synthetic data and merge with given data to get final full data.
```text
scripts/generate_synthetic_data.py
scripts/final_data/merge.py
```

```text
Epoch 1 average loss: 0.4637
Epoch 2 average loss: 0.0064
Epoch 3 average loss: 0.0037
```
#### For test dataset:
```text
Per-entity metrics:
CITY            P=1.000 R=1.000 F1=1.000
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=0.200 R=0.200 F1=0.200
PERSON_NAME     P=0.812 R=0.975 F1=0.886
PHONE           P=0.298 R=0.350 F1=0.322
```

Macro-F1: 0.568

```text
PII-only metrics: P=0.766 R=0.670 F1=0.715
Non-PII metrics: P=1.000 R=1.000 F1=1.000
```

Latency over 50 runs (batch_size=1):
  ```text
  p50: 7.75 ms
  p95: 9.38 ms
```
Adding a minimum confidence value for positiv class prediction:
For current model, max F1 score obtained for min_conf = 0.7 which is 0.568


Using google electra small model:

Hyperparams:
```text
Batch: 32
Context Length = 256
Epochs = 6
```
```text
Per-entity metrics:
CITY            P=0.400 R=1.000 F1=0.571
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=0.950 R=0.950 F1=0.950
PERSON_NAME     P=0.765 R=0.975 F1=0.857
PHONE           P=0.105 R=0.100 F1=0.103

Macro-F1: 0.580

PII-only metrics: P=0.793 R=0.670 F1=0.726
Non-PII metrics: P=0.284 R=1.000 F1=0.442
```



## Updates Scripts (based on new nomencalture of files)

1. Train:
```text
python src/train.py   --model_name google/electra-small-discriminator   --train data/full_train.jsonl   --dev data/full_dev.jsonl   --out_dir out   --batch_size 32   --epochs 6   --max_length 256   --device cpu 
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
