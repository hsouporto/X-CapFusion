from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TextClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def load_model(model_name, num_labels):
    return TextClassifier(model_name, num_labels)

