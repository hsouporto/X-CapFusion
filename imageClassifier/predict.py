import torch
import csv
from tqdm import tqdm
from .model import load_model
import numpy as np
from sklearn.metrics import fbeta_score
from .dataset import get_dataloader,load_config  
import json
import os

def load_model_from_checkpoint(model, checkpoint_path, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  
    return model


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
    config_path = "./imageClassifier/config.json"
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_name="microsoft/cvt-21", num_labels=config["dataset"]["n_labels"])
    
    checkpoint_path = config["output"]["checkpoint_path"]
    model = load_model_from_checkpoint(model, checkpoint_path, device)

    dataloader = get_dataloader(group_data=dataset)
    model.eval()

    all_outputs = []
    all_labels = []
    all_image_paths = []
    all_captions = []

    with torch.no_grad():
        for images, labels, batch_captions, image_paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            all_image_paths.extend(image_paths)
            all_captions.extend(batch_captions)

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
