import statistics
from tqdm import tqdm
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from bert_score import score

def calculate_cider(predicted, true):
    assert len(predicted) == len(true), "The list must have the same size"
    references = {i: [{"caption": true[i]}] for i in range(len(true))}
    hypotheses = {i: [{"caption": predicted[i]}] for i in range(len(predicted))}
    tokenizer = PTBTokenizer()
    references = tokenizer.tokenize(references)
    hypotheses = tokenizer.tokenize(hypotheses)
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(references, hypotheses)
    return score
def calculate_meteor(candidate_list, reference_list):
    if len(candidate_list) != len(reference_list):
        raise ValueError("candidate_list and reference_list must have the same length.") 
    scores = []
    for i in range(len(candidate_list)):
        candidate_tokens = candidate_list[i].split()
        reference_tokens = reference_list[i].split()
        score = meteor_score([reference_tokens], candidate_tokens)
        scores.append(score)
    mean_score = sum(scores) / len(scores)
    return mean_score
def calculate_bleu(predicted, true):
    smoothing = SmoothingFunction().method4
    references = [[gt.split()] for gt in true]  
    candidates = [pred.split() for pred in predicted] 
    score = corpus_bleu(references, candidates, smoothing_function=smoothing)
    return score
def calculate_rouge(predicted, true):
    rouge1_precision=[]
    rouge2_precision=[]
    rougeL_precision=[]
    rouge1_recall=[]
    rouge2_recall=[]
    rougeL_recall=[]
    rouge1_f1=[]
    rouge2_f1=[]
    rougeL_f1=[]
    for pred, gt in tqdm(zip(predicted, true), total=len(predicted), desc="Calculating ROUGE"):
        reference = gt
        candidate = pred
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        rouge1_precision.append(scores["rouge1"].precision)
        rouge2_precision.append(scores["rouge2"].precision)
        rougeL_precision.append(scores["rougeL"].precision)
        rouge1_recall.append(scores["rouge1"].recall)
        rouge2_recall.append(scores["rouge2"].recall)
        rougeL_recall.append(scores["rougeL"].recall)
        rouge1_f1.append(scores["rouge1"].fmeasure)
        rouge2_f1.append(scores["rouge2"].fmeasure)
        rougeL_f1.append(scores["rougeL"].fmeasure)
    rouge_means = {
        "rouge1_precision_mean": statistics.mean(rouge1_precision),
        "rouge2_precision_mean": statistics.mean(rouge2_precision),
        "rougeL_precision_mean": statistics.mean(rougeL_precision),
        "rouge1_recall_mean": statistics.mean(rouge1_recall),
        "rouge2_recall_mean": statistics.mean(rouge2_recall),
        "rougeL_recall_mean": statistics.mean(rougeL_recall),
        "rouge1_f1_mean": statistics.mean(rouge1_f1),
        "rouge2_f1_mean": statistics.mean(rouge2_f1),
        "rougeL_f1_mean": statistics.mean(rougeL_f1),
    }
    return  rouge_means
def calculate_bert_score(predicted, true):
    P, R, F1 = score(predicted, true, lang='en')
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()
    bertscore = {
        "Bert_precision": bert_precision,
        "Bert_recall": bert_recall,
        "Bert_f1": bert_f1
    }
    return bertscore
