from typing import Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer, AutoModelForCausalLM
)
from utils.modules import ConvAggregator, TransformerMapper, CrossAttentionLayer



class XCapFusionIntermediate(nn.Module):
    def reconstruction_loss(embedding_label, reconstructed_label):
        loss = torch.nn.functional.mse_loss(embedding_label, reconstructed_label)
        return loss
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8, num_attention_heads: int = 8):
        super(XCapFusionIntermediate, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained("Rabbiaaa/MedGPT")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.tokenizer= AutoTokenizer.from_pretrained("Rabbiaaa/MedGPT")
        repo = "microsoft/rad-dino"
        self.model = AutoModel.from_pretrained(repo)
        self.crossAtt=CrossAttentionLayer(768,8)
        # Learnable convolutional layer to aggregate patch features
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
        for param in self.crossAtt.parameters():
            param.requires_grad = True 
    def forward(self, image_tensor: torch.Tensor, tokens_caption_tensor: torch.Tensor, tokens_label_tensor: torch.Tensor,
                mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens_caption_tensor)
        embedding_label = self.gpt.transformer.wte(tokens_label_tensor)
        inputs = {"pixel_values": image_tensor}
        vision_outputs = self.model(**inputs)
        patch_features = vision_outputs.last_hidden_state 
        patch_features = patch_features.permute(0, 2, 1) 
        conv_output = self.conv(patch_features) 
        conv_output = conv_output.mean(dim=2)
        conv_output = conv_output / conv_output.norm(2, -1, keepdim=True)
        prefix_projections = self.project(conv_output).view(-1, self.prefix_length, self.gpt_embedding_size)
        self.prefix_length_with_labels = prefix_projections.shape[1]
        prefix_projections=self.crossAtt(embedding_label, prefix_projections)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out
