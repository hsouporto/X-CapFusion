import os
import yaml
import pandas as pd
import nltk
nltk.download('wordnet')
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from metricsNLP import calculate_bert_score, calculate_bleu, calculate_meteor, calculate_cider, calculate_rouge

def clean_sentences(sentences):
    return [sentence.strip().replace("\n", " ").replace("\r", "") for sentence in sentences]


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

def evaluate(dataset):
    print(f">>> Evaluating Traditional Metrics....")
    config_path="./earlyLabel/config.yaml"
    config=load_config(config_path)
    csv_file=os.path.join(config["paths"]["predictions_path"],f"predictions_{dataset}.csv")
    data = pd.read_csv(csv_file)
    output=os.path.join(config["paths"]["evaluation_path"],f"traditional_metrics_{dataset}.csv")
    bert=calculate_bert_score(clean_sentences(data["GT_caption"]), clean_sentences(data["Predicted_caption"]))
    blue=calculate_bleu(data["GT_caption"], data["Predicted_caption"])
    rouge=calculate_rouge(data["GT_caption"], data["Predicted_caption"])
    meteor=calculate_meteor(data["GT_caption"], data["Predicted_caption"])
    cyder=calculate_cider(data["GT_caption"], data["Predicted_caption"])
    summary = {
        'BLEU': blue,
        'ROUGE': rouge,
        'Bert Score': bert,
        'METEOR Score':meteor,
        'Cide Score': cyder
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output, index=False)
    print(f"Results saved to {output} successfully!")
