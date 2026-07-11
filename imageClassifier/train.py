# train.py
import torch
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn
import os
from .model import load_model
from .dataset import get_dataloader, load_config
import json

def train_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    print(f">>> Training epoch {epoch}")
    for images, labels, _, _ in tqdm(dataloader, desc=f"Training ..."):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5  
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.numel()
    avg_loss = running_loss / len(dataloader)
    avg_accuracy = correct_preds / total_preds
    return avg_loss, avg_accuracy


def save_metrics(train_losses, train_accuracies, metricsTrain):
    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }
    with open(metricsTrain, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Métricas de treinamento salvas em 'training_metrics.json'.")

def train():    
    train_losses = []
    train_accuracies = []
    config_path="./imageClassifier/config.json"
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_name="microsoft/cvt-21", num_labels=config["dataset"]["n_labels"]) 
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["model"]["parameters"]["learning_rate"])
    train_loader = get_dataloader( group_data="train")
    num_epochs = config["model"]["parameters"]["epochs"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        if epoch==num_epochs-1:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
    metricsTrain= config["output"]["metricsTrain"]
    save_metrics(train_losses, train_accuracies,metricsTrain)

