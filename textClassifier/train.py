
import torch
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn
import os
import json
from sklearn.metrics import f1_score, accuracy_score
from .model import load_model
from .dataset import get_dataloader, load_config
import numpy as np


def train_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    criterion = nn.BCEWithLogitsLoss()
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        preds_all.append(preds)
        labels_all.append(labels.detach().cpu().numpy())    
    avg_loss = running_loss / len(dataloader)
    preds_all = np.vstack(preds_all)
    labels_all = np.vstack(labels_all)
    avg_f1 = f1_score(labels_all, preds_all, average='micro')
    avg_accuracy = accuracy_score(labels_all, preds_all)
    return avg_loss, avg_accuracy, avg_f1


def save_metrics(train_losses, train_accuracies, train_f1scores, metricsTrain):
    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_f1scores": train_f1scores
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Métricas de treinamento salvas em 'training_metrics.json'.")


def train():
    config_path="./imageClassifier/config.json"
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model("dmis-lab/biobert-base-cased-v1.1", 14)
    model.to(device)
    learning_rate = float(config["model"]["parameters"]["learning_rate"])
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_loader = get_dataloader(config_path=config_path, group_data='train')
    num_epochs = config["model"]["parameters"]["epochs"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses = []
    train_accuracies = []
    train_f1scores = []
    print("Training Text Classifier...")

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch} ===")
        train_loss, train_accuracy, train_f1 = train_epoch(model, train_loader, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1scores.append(train_f1)
        if epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
            torch.save(model.state_dict(), checkpoint_path)
    metricsTrain= config["output"]["metricsTrain"]
    save_metrics(train_losses, train_accuracies, train_f1scores,metricsTrain)

