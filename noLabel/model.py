import torch
import torch.nn as nn
import yaml
from typing import Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils.modules import ConvAggregator, TransformerMapper


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config


class XCapFusionNoLabel(nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8, num_attention_heads: int = 8):
        super(XCapFusionNoLabel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained("Rabbiaaa/MedGPT", output_attentions=True, return_dict_in_generate=True,attn_implementation="eager")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.tokenizer= AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
        self.model = AutoModel.from_pretrained("microsoft/rad-dino")
        self.conv = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv = ConvAggregator(input_dim=768, hidden_dim=1024, output_dim=2048, dropout=0.1)
        self.project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = True
        for param in self.project.parameters():
            param.requires_grad = True
        for param in self.conv.parameters():
            param.requires_grad = True
    def forward(self, image_tensor: torch.Tensor, tokens_caption_tensor: torch.Tensor,
                mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens_caption_tensor)
        inputs = {"pixel_values": image_tensor}
        vision_outputs = self.model(**inputs)
        patch_features = vision_outputs.last_hidden_state
        patch_features = patch_features.permute(0, 2, 1) 
        conv_output = self.conv(patch_features) 
        conv_output = conv_output.mean(dim=2) 
        conv_output = conv_output / conv_output.norm(2, -1, keepdim=True)
        prefix_projections = self.project(conv_output).view(-1, self.prefix_length, self.gpt_embedding_size)
        self.prefix_length_with_labels = prefix_projections.shape[1]
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

