from transformers import CvtForImageClassification
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ImageClassifier, self).__init__()
        self.model = CvtForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True  
        )
    def forward(self, x):
        return self.model(x).logits 



def load_model(model_name, num_labels):
    return ImageClassifier(model_name, num_labels)
