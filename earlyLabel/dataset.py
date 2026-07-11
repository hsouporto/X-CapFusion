import os
import glob
import random
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoTokenizer)
from transformers.image_processing_utils import BatchFeature


def to_one_hot(label_indices, num_labels=14):
    label_indices = label_indices.long() 
    one_hot = torch.zeros(num_labels, dtype=torch.float32)
    one_hot[label_indices] = 1
    return one_hot

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

class MIMICDataset(Dataset):
    def __init__(self, group_data='train', config_path="config.yaml"):
        config = load_config(config_path)
        self.data_dir = config["paths"]["data_dir"]
        self.prefix_length = config["imageEncoder"]["prefix_length"]
        self.tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
        label_path = os.path.join(self.data_dir, 'mimic-cxr-2.0.0-negbio.csv')
        self.df_labels = pd.read_csv(label_path)
        self.valid_data = []
        self.dataset_raw =pd.read_csv(os.path.join(self.data_dir, f"raw_reports_{group_data}.csv") )
        self.captions_tokens = []
        self.label_tokens = []
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df_labels.columns[2:]))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_labels = len(self.df_labels.columns[2:])
        self.tokenizer = AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
        self.preprocess = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        for _, entry in tqdm(self.dataset_raw.iterrows(), desc="Processing dataset", unit="entry", total=len(self.dataset_raw)):
            folder = entry["REPORT_ID"]
            folder = folder.replace(".txt", "")
            folder_num = int(folder[1:])
            patient = entry["PATIENT_ID"]
            patient_num = int(patient[1:])
            caption = entry["REPORTS"]
            row_label = self.df_labels[(self.df_labels["subject_id"] == patient_num) & 
                                       (self.df_labels["study_id"] == folder_num)]
            if row_label.empty:
                continue 
            row_label = row_label.iloc[:, 2:].iloc[0]  
            label = row_label[row_label == 1].index.tolist()
            foldername = os.path.join(self.data_dir, "files", patient[:3], patient, folder)
            files = glob.glob(os.path.join(foldername, '*'))
            if not files:
                continue 
            tokens_caption_tensor = torch.tensor(self.tokenizer.encode(caption))
            self.captions_tokens.append(tokens_caption_tensor)
            label_indices = [self.label2idx[l] for l in label]
            one_hot_labels = to_one_hot(torch.tensor(label_indices), num_labels=self.num_labels)
            self.label_tokens.append(one_hot_labels)  
            self.valid_data.append((files, patient, folder))
        all_len_captions= torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len_captions = min(int(all_len_captions.mean() + all_len_captions.std() * 10), int(all_len_captions.max()))

    def pad_tokens(self, item: int):
        tokens_caption = self.captions_tokens[item]        
        padding = self.max_seq_len_captions - tokens_caption.shape[0]
        if padding > 0:
            tokens_caption = torch.cat((tokens_caption, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens_caption = tokens_caption[:self.max_seq_len_captions]
        mask = tokens_caption.ge(0)
        tokens_caption[~mask] = 0
        mask = mask.float()   
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens_caption,  mask
    def __len__(self):
        return len(self.valid_data)
    def __getitem__(self, idx):
        tokens_caption_tensor, mask = self.pad_tokens(idx)
        labels_one_hot=self.label_tokens[idx]
        files,_,_ = self.valid_data[idx]
        filename = random.choice(files)
        image = io.imread(filename)
        image = Image.fromarray(image) 
        image_tensor = self.preprocess(image) 
        if isinstance(image_tensor, BatchFeature):
            image_tensor = image_tensor["pixel_values"]  
        if isinstance(image_tensor, list):  
             image_tensor = torch.tensor(np.array(image_tensor), dtype=torch.float32)
        image_tensor = image_tensor.squeeze(0) 
        return image_tensor,  tokens_caption_tensor, labels_one_hot, mask




def get_dataloader(config_path="config.yaml", group_data='train'):
    config = load_config(config_path)
    dataset = MIMICDataset(group_data=group_data, config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False)
    return dataloader

