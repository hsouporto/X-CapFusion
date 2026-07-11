import os
import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import fbeta_score
import torch
from transformers import (
    AutoTokenizer
)
from .model import load_model
from .dataset import get_dataloader
import yaml


def to_one_hot(label_indices, num_labels=14):
    label_indices = label_indices.long() 
    one_hot = torch.zeros(num_labels, dtype=torch.float32)
    one_hot[label_indices] = 1
    return one_hot




def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    return config




def calculate_dynamic_thresholds(outputs, labels, beta=0.5):
    thresholds = []
    preds = torch.sigmoid(outputs).cpu().numpy()
    labels = labels.cpu().numpy()  
    for i in range(preds.shape[1]):
        best_fbeta = 0
        best_th = 0.5
        for th in np.arange(0.01, 1, 0.01):
            pred_label = (preds[:, i] > th).astype(int)
            fbeta = fbeta_score(labels[:, i], pred_label, beta=beta, zero_division=0)
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_th = th
        thresholds.append(best_th)
    return thresholds





def predict(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = "./textClassifier/config.json"
    config = load_config(config_path)
    classifier_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    classifier = load_model("dmis-lab/biobert-base-cased-v1.1", 14)  # Ajuste o número de classes multilabel
    classifier.load_state_dict(torch.load(config["output"]["checkpoint_path"], map_location=device))
    classifier.eval()
    classifier=classifier.to(device)
    dataloader = get_dataloader(group_data=dataset, config_path=config_path)

    all_outputs = []
    all_labels = []
    all_image_paths = []
    all_captions = []

    with torch.no_grad():
        for input_ids, attention_mask, labels_one_hot, captions, image_paths in tqdm(dataloader, desc="Predicting"):
            outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels_one_hot.cpu())
            all_image_paths.extend(image_paths)
            all_captions.extend(captions)

          
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    dynamic_thresholds = calculate_dynamic_thresholds(all_outputs, all_labels)
    threshold_path = config["output"]["dynamicThreshold"]
    with open(threshold_path, "w") as f:
        json.dump(dynamic_thresholds, f) 
    probs = torch.sigmoid(all_outputs).numpy()
    preds_bin = (probs > np.array(dynamic_thresholds)).astype(int)

    results = []
    for img_path, true_label, pred, caption in zip(all_image_paths, all_labels.numpy(), preds_bin, all_captions):
        results.append([img_path, true_label.tolist(), pred.tolist(), caption])

    os.makedirs(config["output"]["predictions"], exist_ok=True)
    csv_filename = os.path.join(config["output"]["predictions"], f"predictions_{dataset}.csv")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "True Label", "Predicted Label", "Caption"])
        writer.writerows(results)
