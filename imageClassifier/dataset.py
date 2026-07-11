import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor

def load_config(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config

class MIMICDataset(Dataset):
    def __init__(self, group_data='train'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_path="./imageClassifier/config.json"
        config = load_config(config_path)
        self.data_dir = config["dataset"]["data_dir"]
        self.dataset_raw =pd.read_csv(os.path.join(self.data_dir, f"raw_reports_{group_data}.csv") )
        label_path = os.path.join(self.data_dir, 'mimic-cxr-2.0.0-negbio.csv')
        self.df_labels = pd.read_csv(label_path)
        self.preprocess = AutoImageProcessor.from_pretrained("microsoft/cvt-21")
        self.valid_data = []
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
            label = np.zeros(len(row_label), dtype=int)
            column_indices = [row_label.index.get_loc(col) for col in row_label[row_label == 1].index]
            label[column_indices] = 1
            label = torch.tensor(label, dtype=torch.float32)
            foldername = os.path.join(self.data_dir, "files", patient[:3], patient, folder)
            files = glob.glob(os.path.join(foldername, '*'))
            if not files:
                continue 
            filename = files[0]  
            if not os.path.exists(filename):
                continue 
            self.valid_data.append((filename, label, caption))
    def __len__(self):
        return len(self.valid_data)
    def __getitem__(self, idx):
        filename, label, caption = self.valid_data[idx]
        image = Image.open(filename).convert("RGB")
        processed_image = self.preprocess(image)  
        if isinstance(processed_image['pixel_values'], list):
            image_tensor = processed_image['pixel_values'][0]  
        else:
            image_tensor = processed_image['pixel_values']
        image_tensor = torch.tensor(image_tensor).to(self.device)  
        return image_tensor, label, caption,filename

def collate_fn(batch):
    images, labels, captions, filenames = zip(*batch)  
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels, captions, filenames

def get_dataloader( group_data='train', shuffle=True):
    config_path="./imageClassifier/config.json"
    config = load_config(config_path)
    data_dir = config["dataset"]["data_dir"]
    dataset = MIMICDataset(group_data=group_data)
    dataloader = DataLoader(dataset, batch_size=config["model"]["parameters"]["batch_size"], shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

