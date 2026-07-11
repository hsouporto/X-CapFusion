import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def to_one_hot(label_indices, num_labels=14):
    label_indices = label_indices.long() 
    one_hot = torch.zeros(num_labels, dtype=torch.float32)
    one_hot[label_indices] = 1
    return one_hot

def load_config(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config



class MIMICDataset(Dataset):
    def __init__(self, group_data='train', config_path="config.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_path="./textClassifier/config.json"
        config = load_config(config_path)        
        self.data_dir = config["dataset"]["data_dir"]
        self.label2idx = {}
        self.idx2label = {}
        label_path = os.path.join(self.data_dir, 'mimic-cxr-2.0.0-negbio.csv')
        self.df_labels = pd.read_csv(label_path)  
        self.dataset_raw =pd.read_csv(os.path.join(self.data_dir, f"raw_reports_{group_data}.csv") )
        self.df_labels = self.df_labels.copy()
        self.df_labels = self.df_labels.fillna(0)
        self.valid_data = []
        self.captions = []
        self.att_masks = []
        self.label_tokens = []
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df_labels.columns[2:]))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_labels = len(self.df_labels.columns[2:])
        for _, entry in tqdm(self.dataset_raw.iterrows(), desc='Processing dataset',total=len(self.dataset_raw)):
            folder = entry["REPORT_ID"].replace(".txt", "")
            folder_num = int(folder[1:])
            patient = entry["PATIENT_ID"]
            patient_num = int(patient[1:])
            caption = entry["REPORTS"]
            row_label = self.df_labels[
                (self.df_labels["subject_id"] == patient_num) & (self.df_labels["study_id"] == folder_num)
            ]
            if row_label.empty:
                continue
            row_label = row_label.iloc[:, 2:]  
            label_indices = [self.label2idx[l] for l in row_label.columns[row_label.iloc[0] == 1]]
            one_hot_labels = to_one_hot(torch.tensor(label_indices), num_labels=self.num_labels)
            self.label_tokens.append(one_hot_labels)
            files = glob.glob(os.path.join(self.data_dir, "files", patient[:3], patient, folder, '*'))
            if not files:
                continue 
            filename = files[0]  

            self.valid_data.append((filename, patient, folder))
            self.captions.append(caption)


    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        filepath, _, _=self.valid_data[idx]
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=500,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        ) 
        labels_one_hot = self.label_tokens[idx]
        return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), labels_one_hot, caption, filepath



def collate_fn(batch):
    input_ids, attention_mask, labels, captions, filenames = zip(*batch)  
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    return input_ids,attention_mask, labels, captions, filenames


def get_dataloader(config_path='config.json', group_data='train'):
    config = load_config(config_path)
    dataset = MIMICDataset(
        group_data=group_data,
        config_path=config_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["model"]["parameters"]["batch_size"],
        shuffle=True,
        drop_last=False
    )
    return dataloader

