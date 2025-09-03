## 🎯 Experiment Objective
----
To compare and analyze anomaly detection methods with imbalanced binary classification methods under extremely imbalanced settings.

## 🧭 Pipeline Overview
----
Data loading → train/validation/test split → (optional) adjustment of minority class size → over/under-sampling → MinMax scaling → model training and score generation → metric storage.

## ▶️ Example Command
----
```python experiment_revision_threshold.py -d 35 -m 0 -s 0 -l 3 -mi 3 -se 0 -g 0 --epochs 50 --lr 0.001 --batch_size 128 --thresholdq 0.95```

## ⚙️ Key Parameter Descriptions
----
- -d Dataset ID

- -m Model (0: mlp, 1: lr, 9: rf, 10: svm, 4–7: Anomaly methods)

- -s Sampling method (none/smote/adasyn, etc.)

- -l Loss function (focal, etc.)

- -mi Number of minority (anomaly) samples

- -se Seed preset

- -g GPU

- --epochs / --lr / --batch_size Training configuration

- --thresholdq Quantile threshold (q)

## 🏷️ Label Convention
----
Anomaly = y=1, Normal = y=0. Scores are assumed to follow “higher = more anomalous.”
