import pandas as pd
import numpy as np
from .dataset import load_config
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from utils.metricsClassification import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_hamming_loss
)
import ast
import json

def evaluate(dataset):
    config_path="./textClassifier/config.json"
    config = load_config(config_path)
    data = pd.read_csv(os.path.join(config["output"]["predictions"], f"predictions_{dataset}.csv"))
    output=os.path.join(config["output"]["metrics"], f"metrics_{dataset}.csv")
    y_true = [list(map(int, ast.literal_eval(label))) for label in data["True Label"]]
    y_pred = [list(map(int, ast.literal_eval(label))) for label in data["Predicted Label"]] 
    y_true = np.array([np.array(label) for label in y_true])
    y_pred = np.array([np.array(label) for label in y_pred])
    with open(config["output"]["dynamicThreshold"], "r") as f:
        thresholds = json.load(f)
    y_pred_bin = (y_pred >= thresholds).astype(int)
    weighted_accuracy = calculate_accuracy(y_true, y_pred_bin)
    weighted_hamming = calculate_hamming_loss(y_true, y_pred_bin)
    weighted_prf_scores = calculate_precision_recall_f1(y_true, y_pred_bin)
    summary = {
        'Weighted Accuracy': weighted_accuracy,
        'Weighted Hamming Loss': weighted_hamming,
        'Weighted Precision': weighted_prf_scores['Precision'],
        'Weighted Recall': weighted_prf_scores['Recall'],
        'Weighted F1 Score': weighted_prf_scores['F1 Score'],
        "accuracy_micro": accuracy_score(y_true.flatten(), y_pred_bin.flatten()),
        "precision_micro": precision_score(y_true, y_pred_bin, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred_bin, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred_bin, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred_bin, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred_bin, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred_bin, average="macro", zero_division=0),
    }
    try:
        auc_per_class = roc_auc_score(y_true, y_pred, average=None)
        for i, auc in enumerate(auc_per_class):
            summary[f'auc_class_{i}'] = auc
        summary["auc_macro"] = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        print("Some class does not have enough positive(negative example for AUC calculation")

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output, index=False)








