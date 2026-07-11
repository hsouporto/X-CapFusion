import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoImageProcessor
)
from transformers.image_processing_utils import BatchFeature
import os
import ast
import re

def parse_tensor_string(s):
    s = s.strip()
    s = re.sub(r"tensor\(", "", s)
    s = re.sub(r",?\s*device=.*\)", "", s)
    s = re.sub(r"\)$", "", s)
    return list(map(float, s.strip("[]").split(',')))

def to_one_hot(label_indices, num_labels=14):
    label_indices = label_indices.long() 
    one_hot = torch.zeros(num_labels, dtype=torch.float32)
    one_hot[label_indices] = 1
    return one_hot



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

def str_to_tensor(s):
    if isinstance(s, torch.Tensor):
        return s.detach().cpu()
    s = s.split('([')[1].split(']')[0]  
    values = list(map(float, s.split(',')))
    return torch.FloatTensor(values)



class MIMICDataset(Dataset):
    def __init__(self, group_data='train', config_path="config.yaml"):
        config = load_config(config_path)
        self.data_dir = config["paths"]["data_dir"]
        self.prefix_length = config["imageEncoder"]["prefix_length"]
        self.tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT", use_fast=False)
        self.valid_data = []
        self.labels_prob=[]
        self.dataset_raw =pd.read_csv(os.path.join(self.data_dir, f"raw_reports_{group_data}.csv") )
        self.captions_tokens = []
        self.dataAll =pd.read_csv(config["paths"]["classifier_predictions"])
        self.captions_list=[]
        self.preprocess = AutoImageProcessor.from_pretrained( "microsoft/rad-dino")
        for _, row in self.dataAll.iterrows():
            FILENAME = row['FILENAME']
            PREDICTED_LABEL = row['PREDICTED_LABEL']
            GT_LABEL = row['GT_LABEL']
            if isinstance(GT_LABEL, str): 
                GT_LABEL = parse_tensor_string(GT_LABEL)
            GT_LABEL_tensor = torch.FloatTensor(GT_LABEL)
            CAPTION = row['CAPTION']
            tokens_caption_tensor = torch.tensor(self.tokenizer.encode(CAPTION))
            self.captions_list.append(CAPTION)
            self.captions_tokens.append(tokens_caption_tensor)
            self.valid_data.append((FILENAME, tokens_caption_tensor, GT_LABEL_tensor))
        all_len_captions= torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len_captions = min(int(all_len_captions.mean() + all_len_captions.std() * 10), int(all_len_captions.max()))
    def pad_tokens(self, item: int):
        FILENAME , tokens_caption, PREDICTED_LABEL = self.valid_data[item]
        padding = self.max_seq_len_captions - tokens_caption.shape[0]
        if padding > 0:
            tokens_caption = torch.cat((tokens_caption, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens_caption = tokens_caption[:self.max_seq_len_captions]
        mask = tokens_caption.ge(0)  
        tokens_caption[~mask] = 0
        mask = mask.float()   
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens_caption,  mask, PREDICTED_LABEL,FILENAME
    def __len__(self):
        return len(self.valid_data)
    def __getitem__(self, idx):
        tokens_caption_tensor, mask, PREDICTED_LABEL, FILENAME = self.pad_tokens(idx)
        image = io.imread(FILENAME)
        image = Image.fromarray(image)  
        image_tensor = self.preprocess(image) 
        if isinstance(image_tensor, BatchFeature):
            image_tensor = image_tensor["pixel_values"] 
        if isinstance(image_tensor, list):  
             image_tensor = torch.tensor(np.array(image_tensor), dtype=torch.float32)
        image_tensor = image_tensor.squeeze(0) 
        return image_tensor,  tokens_caption_tensor, mask, PREDICTED_LABEL
def get_dataloader(config_path="config.yaml", group_data='train'):
    config = load_config(config_path)
    dataset = MIMICDataset(group_data=group_data, config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False)
    return dataloader


