import os, sys, json, pickle, random
from tqdm.notebook import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers import (AdamW,
                          get_linear_schedule_with_warmup,
                          BartForConditionalGeneration,
                          BartTokenizer,
                          AutoModel, 
                          AutoConfig)


class u2p_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device

        # load model and tokenizer
        self.model = AutoModel.from_pretrained(self.args.model_name_or_path).to(self.device)
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.loss_fn = CrossEntropyLoss()
        
            
    def forward(self, batch):
        input_ids, label, _ = batch

        batch_size, clust, seq_len = input_ids.shape
        num_cand = int(clust/(self.args.max_num_uttr * self.args.max_num_prsn))
        input_ids = input_ids.view(-1, seq_len)
        
        output = self.model(input_ids = input_ids.to(self.device))
        pooled_output = output[1] # [batch*clust, 768]
        logits = self.classifier(pooled_output) # [batch*clust, 1]
        logits = logits.view(-1, self.args.max_num_uttr, self.args.max_num_prsn)
        logits = torch.max(logits, -1)[0]
        logits = torch.sum(logits, -1)
        reshaped_logits = logits.view(-1, num_cand)
        loss = self.loss_fn(reshaped_logits, label.to(self.device))
        
        return loss, reshaped_logits

