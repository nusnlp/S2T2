import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm, trange
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, AdamW

class SMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, input):
        x = self.layer(input)

        return x

class PretrainedEmbedder(nn.Module):
    def __init__(self, model_name='xlm-roberta-large'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.config.output_hidden_states = True
        self.dropout = 0.3

    def forward(self, inp_ids, attn_mask):
        out = self.model(input_ids=inp_ids, attention_mask=attn_mask)
        return out