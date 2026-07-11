import os
import sys
import json
import yaml
import torch
from torch.nn import functional as nnf
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from .dataset import get_dataloader
from .model import XCapFusionNoLabel
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from textClassifier.model import load_model


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config


 
def train():
    config_path="./noLabel/config.yaml"
    config = load_config(config_path)
    prefix_length = config["imageEncoder"]["prefix_length"]
    prefix_length_clip= config["imageEncoder"]["prefix_length_clip"]
    output_dir=config["paths"]["checkpoints"]
    train_dataloader = get_dataloader( config_path=config_path, group_data="train")
    warmup_steps=config["training"]["warmup_steps"]
    lr=config["training"]["lr"]
    alpha=config["training"]["alpha"]
    epochs_number=config["training"]["epochs_number"]
    lr=float(lr)    
    prefix_dim =  config["imageEncoder"]["prefix_dimension"]
    num_layers= config["mapping"]["num_layers"]
    model = XCapFusionNoLabel(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                num_layers=num_layers )
    tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout.flush() 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs_number * len(train_dataloader)
    )
    LOSS=[]
    print("Training Late Label Conditioned Captioning...")
    for epoch in range(epochs_number):
        epoch_loss = 0
        total_caption_loss=0
        total_classifier_loss=0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="trainig")
        for idx, (image_tensor,  tokens_caption_tensor , mask) in enumerate(train_dataloader):
            model.zero_grad()
            image_tensor,  tokens_caption_tensor , mask =image_tensor.to(device), tokens_caption_tensor.to(device),mask.to(device)
            outputs = model(image_tensor,tokens_caption_tensor, mask)
            prefix_length=model.prefix_length_with_labels
            logits = outputs.logits[:, prefix_length - 1: -1]            
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens_caption_tensor.flatten().long(),  
                ignore_index=0
            )   
            total_loss = loss
            epoch_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            del image_tensor, tokens_caption_tensor, mask, logits, loss, total_loss
            torch.cuda.empty_cache()
        progress.close()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        LOSS.append(avg_epoch_loss)
        if epoch == epochs_number-1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "checkpoint.pt"),
            )
    with open(os.path.join(output_dir, "total_loss_history.json"), "w") as f:
        json.dump(LOSS, f)
    return model
