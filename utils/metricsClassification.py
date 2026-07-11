from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / y_true.size
    return accuracy
def calculate_hamming_loss(y_true, y_pred):
    incorrect_predictions = np.sum(y_true != y_pred)
    hamming_loss = incorrect_predictions / y_true.size
    return hamming_loss


def calculate_precision_recall_f1(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }