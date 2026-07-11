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
from .model import XCapFusionEarly


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

 
def train():
    config_path="./earlyLabel/config.yaml"
    config = load_config(config_path)
    prefix_length = config["imageEncoder"]["prefix_length"]
    prefix_length_clip= config["imageEncoder"]["prefix_length_clip"]
    output_dir=config["paths"]["checkpoints"]
    train_dataloader = get_dataloader( config_path=config_path, group_data="train")
    warmup_steps=config["training"]["warmup_steps"]
    lr=config["training"]["lr"]
    epochs_number=config["training"]["epochs_number"]
    lr=float(lr)    
    prefix_dim =  config["imageEncoder"]["prefix_dimension"]
    num_layers= config["mapping"]["num_layers"]
    model = XCapFusionEarly(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                num_layers=num_layers )
    print("Training Early Label Conditioned Captioning...")
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
    for epoch in range(epochs_number):
        epoch_loss = 0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="Training ....")
        for idx, (image_tensor,  tokens_caption_tensor ,one_hot_label, mask) in enumerate(train_dataloader):
            one_hot_label = one_hot_label.long()
            model.zero_grad()
            tokens_caption_tensor, mask, one_hot_label, image_tensor =tokens_caption_tensor.to(device), mask.to(device), one_hot_label.to(device), image_tensor.to(device)
            outputs = model(image_tensor,tokens_caption_tensor,one_hot_label, mask)
            prefix_length=model.prefix_length_with_labels
            logits = outputs.logits[:, prefix_length - 1: -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens_caption_tensor.flatten().long(), 
                ignore_index=0
            )   
            epoch_loss += loss.item()         
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            del image_tensor, tokens_caption_tensor, one_hot_label, mask, logits, loss
            torch.cuda.empty_cache()
        progress.close()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        LOSS.append(avg_epoch_loss)
        print(LOSS)
        if epoch == epochs_number-1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "checkpoint.pt"),
            )
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(LOSS, f)
    return model



