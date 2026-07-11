import os
import sys
import re
import csv
import glob
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import skimage.io as io
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor
)
from transformers.image_processing_utils import BatchFeature
from .model import XCapFusionEarly, ConvAggregator


def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)  
    seen = set()
    filtered_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            filtered_sentences.append(sentence)
            seen.add(sentence)
    return " ".join(filtered_sentences)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

def to_one_hot(label_indices, num_labels=14):
    label_indices = label_indices.long()
    one_hot = torch.zeros(num_labels, dtype=torch.float32)
    one_hot[label_indices] = 1
    return one_hot


def predict(dataset):
    config_path="./earlyLabel/config.yaml"
    image_path=[]
    predicted_captions = []
    real_captions = []
    FOLDER_NAMES=[]
    predictor = EarlyLabelPredictor(config_path)
    config = load_config(config_path)
    data_dir = config["paths"]["data_dir"]
    dataset_raw =pd.read_csv(os.path.join(data_dir, f"raw_reports_{dataset}.csv") )
    label_path = os.path.join(data_dir, 'mimic-cxr-2.0.0-negbio.csv')
    df_labels = pd.read_csv(label_path)
    label2idx = {label: idx for idx, label in enumerate(sorted(df_labels.columns[2:]))}
    print(f">>> Predicting ....")
    for _, entry in tqdm(dataset_raw.iterrows(), desc="Predicting ...", unit="entry", total=len(dataset_raw)):
        folder = entry["REPORT_ID"]
        folder=folder.replace(".txt", "")
        folder_num = int(folder[1:])
        patient = entry["PATIENT_ID"]
        patient_num = int(patient[1:])
        caption = entry["REPORTS"]
        num_labels=len(df_labels.columns[2:])
        row_label = df_labels[(df_labels["subject_id"] == patient_num) & 
                                    (df_labels["study_id"] == folder_num)]
        if row_label.empty:
            continue 
        row_label = row_label.iloc[:, 2:].iloc[0] 
        label = row_label[row_label == 1].index.tolist()
        label_indices = [label2idx[l] for l in label]
        one_hot_labels = to_one_hot(torch.tensor(label_indices), num_labels=num_labels).unsqueeze(0)
        foldername=f"{config['paths']['data_dir']}/files/{patient[:3]}/{patient}/{folder}"
        FOLDER_NAMES.append(foldername)
        files = glob.glob(os.path.join(foldername, '*'))
        if len(files) == 0:
            print(f"No files found in folder: {folder}")
            continue
        else:
            i=FOLDER_NAMES.count(foldername)
            print(i)
            try:
                filename = random.choice(files)
                print(filename)
                if not os.path.isfile(filename):
                        continue
                try:
                    predicted_caption = predictor.predict(filename,one_hot_labels)
                    real_caption=caption
                    image_path.append(filename)
                    predicted_captions.append(predicted_caption)
                    real_captions.append(real_caption)  
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            except IndexError:
                print(f"Error: Index {i-1} is out of range for the files list.")
    output_csv_file =  os.path.join(config["paths"]["predictions_path"],f"predictions_{dataset}.csv")
    with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["img_path", "GT_caption", "Predicted_caption"])
        for path, real_caption, predicted_caption in zip(image_path, real_captions, predicted_captions):
            writer.writerow([path, real_caption, predicted_caption])
    print(f"Results saved to {output_csv_file} successfully!")
    
      
class EarlyLabelPredictor:
    def __init__(self, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = AutoModel.from_pretrained("microsoft/rad-dino").to(self.device)
        self.preprocess = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        self.cvt=ConvAggregator(input_dim=768, hidden_dim=1024, output_dim=2048, dropout=0.1)
        self.tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
        config = load_config(config_path)
        self.models = {}
        self. prefix_length = config["imageEncoder"]["prefix_length"]
        num_layers=  config["mapping"]["num_layers"]
        prefix_dim = config["imageEncoder"]["prefix_dimension"]
        prefix_length_clip=config["imageEncoder"]["prefix_length_clip"]
        self.model = XCapFusionEarly(self.prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                num_layers=num_layers)
        self.model.load_state_dict(torch.load(config["paths"]["checkpoints_load"], map_location=self.device))
        self.model = self.model.eval().to(self.device)
    def predict(self, image_path, one_hot_labels):
        one_hot_labels=one_hot_labels.to(self.device)
        image = io.imread(image_path)
        image = Image.fromarray(image)  
        image_tensor = self.preprocess(image)  
        if isinstance(image_tensor, BatchFeature):
            image_tensor = image_tensor["pixel_values"] 
        if isinstance(image_tensor, list):  
            image_tensor = np.array(image_tensor)  
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        image_tensor = image_tensor.squeeze(0).to(self.device)
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0)
            embedding_label_extended = self.model.mapppingLabel(one_hot_labels.float())
            inputs = {"pixel_values": image_tensor}
            vision_outputs = self.model.model(**inputs)
            patch_features = vision_outputs.last_hidden_state 
            patch_features = patch_features.permute(0, 2, 1)  
            conv_output = self.model.conv(patch_features)  
            conv_output = conv_output.mean(dim=2)
            conv_output = conv_output / conv_output.norm(2, -1, keepdim=True)
            prefix_projections = self.model.project(conv_output).view(-1, self.prefix_length, self.model.gpt_embedding_size)
            prefix_projections=self.model.crossAtt(embedding_label_extended, prefix_projections) 
        return generate_beam(self.model, self.tokenizer, embed= prefix_projections)[0]
   
def generate_beam(
    model,
    tokenizer,
    embed=None,
):
    config_path="./earlyLabel/config.yaml"
    config = load_config(config_path)
    beam_size=config["generation"]["beam_size"]
    entry_length=config["generation"]["entry_length"]
    temperature=config["generation"]["temperature"]
    prompt=config["generation"]["prompt"]
    model.eval()
    stop_token_index = torch.tensor([198, 198], device=next(model.parameters()).device)
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped | next_tokens.squeeze().unsqueeze(1).eq(stop_token_index).any(dim=1)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    output_texts=remove_repeated_sentences(output_texts[0])
    last_dot_index = output_texts.rfind('.')  
    if last_dot_index != -1:  
        output_texts= output_texts[:last_dot_index + 1] 
    return [output_texts]


