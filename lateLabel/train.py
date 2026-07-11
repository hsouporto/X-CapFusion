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
from .model import XCapFusionLate
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from textClassifier.model import load_model


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config


 
def train():
    config_path="./lateLabel/config.yaml"
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
    model = XCapFusionLate(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                num_layers=num_layers )
    tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
    classifier_tokenizer = AutoTokenizer.from_pretrained( "dmis-lab/biobert-base-cased-v1.1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = load_model("dmis-lab/biobert-base-cased-v1.1", 14) 
    classifier.load_state_dict(torch.load(config["paths"]["checkpoints_load_classifier"], map_location=device))
    sys.stdout.flush() 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    classifier=classifier.to(device)
    model.train()
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs_number * len(train_dataloader)
    )
    classifier_loss_fn = torch.nn.BCEWithLogitsLoss()
    LOSS=[]
    CLASSIFIER_LOSS=[]
    CAPTION_LOSS=[]
    print("Training Late Label Conditioned Captioning...")
    for epoch in range(epochs_number):
        epoch_loss = 0
        total_caption_loss=0
        total_classifier_loss=0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="trainig")
        for idx, (image_tensor,  tokens_caption_tensor , mask, true_labels) in enumerate(train_dataloader):
            model.zero_grad()
            image_tensor,  tokens_caption_tensor , mask, true_labels =image_tensor.to(device), tokens_caption_tensor.to(device),mask.to(device), true_labels.to(device)
            outputs = model(image_tensor,tokens_caption_tensor, mask)
            prefix_length=model.prefix_length_with_labels
            logits = outputs.logits[:, prefix_length - 1: -1]
            predicted_tokens = torch.argmax(logits.detach(), dim=-1)
            generated_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted_tokens]
            encoding = classifier_tokenizer(
                generated_texts,
                add_special_tokens=True,
                max_length=500,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids=encoding['input_ids'].to(device)
            attention_mask=encoding['attention_mask'].to(device)
            with torch.no_grad():
                classifier_logits = classifier(input_ids=input_ids, attention_mask=attention_mask)
                prediction_labels = torch.sigmoid(classifier_logits) 
            classifier_loss = classifier_loss_fn(prediction_labels, true_labels.squeeze(1))
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens_caption_tensor.flatten().long(),  
                ignore_index=0
            )   
            total_loss = loss + alpha * classifier_loss  # alpha = peso da loss auxiliar (ex: 0.5)
            total_caption_loss+=loss
            total_classifier_loss+=classifier_loss
            epoch_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            del image_tensor, tokens_caption_tensor,true_labels, mask, logits, loss, classifier_loss, total_loss
            torch.cuda.empty_cache()
        progress.close()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        LOSS.append(avg_epoch_loss)
        avg_epoch_loss_caption = total_caption_loss / len(train_dataloader)
        CAPTION_LOSS.append(avg_epoch_loss_caption)
        avg_epoch_loss_classifier = total_classifier_loss / len(train_dataloader)
        CLASSIFIER_LOSS.append(avg_epoch_loss_classifier)
        if epoch == epochs_number-1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "checkpoint.pt"),
            )
    with open(os.path.join(output_dir, "total_loss_history.json"), "w") as f:
        json.dump([x.item() if torch.is_tensor(x) else x for x in LOSS], f)

    with open(os.path.join(output_dir, "total_caption_loss_history.json"), "w") as f:
        json.dump([x.item() if torch.is_tensor(x) else x for x in CAPTION_LOSS], f)

    with open(os.path.join(output_dir, "total_classifier_loss_history.json"), "w") as f:
        json.dump([x.item() if torch.is_tensor(x) else x for x in CLASSIFIER_LOSS], f)
    return model
