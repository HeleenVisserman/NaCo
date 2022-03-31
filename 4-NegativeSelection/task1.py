from sklearn.metrics import roc_auc_score
import numpy as np
import subprocess

lines = []

with open('./task1/pla-eng-result-r3.txt') as f:
    lines = [float(line.strip()) for line in f]


print(lines)

english_labels = np.zeros(124)
print(english_labels)
other_labels = np.ones(500)
labels = np.concatenate((english_labels, other_labels))
print(labels)

score = roc_auc_score(labels, lines)
print("ROCAUC: ", score)