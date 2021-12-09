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


data_dir = {"train": './data_PMPC/train_both_revised.txt',
           "valid": './data_PMPC/valid_both_revised_cand_10.txt',
           "test": './data_PMPC/test_both_revised_cand_10.txt'}

def pad_ids(text, max_len, pad_id):
    if len(text)< max_len:
        res = text + [pad_id] * (max_len-len(text))
    else:
        res = text[:max_len]
    return res

class SPDDataset(Dataset):
    def __init__(self, args, split, tokenizer=None):
        super().__init__()
        self.args = args
        self.split = split
        if not tokenizer:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            self.tok = tokenizer
        self.pad_id = self.tok.pad_token_id
        self.load_data()

    def convert_to_inst(self, data):
        idx, context, prsn, cands = data.split("\t")
        context = [ele.strip() for ele in context.split("_eos_")]
        prsn = [ele.strip() for ele in prsn.split("_eos_")]
        cand_out = []
        for ele in cands.split("|"):
            cand_out.append(ele.split("_eos_"))
        return {"id": idx,
               "dialogue": context,
               "persona": prsn,
               "candidate": cand_out}
    
    def load_data(self):
        self.examples = []
        with open(data_dir[self.split]) as fout:
            for line in fout:
                try: 
                    self.examples.append(self.convert_to_inst(line))
                except:
                    pass
        

    def __len__(self):
        return len(self.examples)

    def generate_inst(self, dialogue, persona):
        num_utt = len(dialogue)
        num_prsn = len(persona)

        inst = []
        if num_utt > self.args.max_num_uttr:
            dialogue = dialogue[:self.args.max_num_uttr]
        if num_prsn > self.args.max_num_prsn:
            persona = persona[:self.args.max_num_prsn]

        for utt in dialogue:
            utt_ids =  self.tok.encode(utt)
            utt_ids = pad_ids(utt_ids, self.args.max_utt_len,  self.tok.pad_token_id)
            for prsn in persona:
                prsn_ids =  self.tok.encode(prsn)[1:]
                prsn_ids = pad_ids(prsn_ids, self.args.max_prsn_len,  self.tok.pad_token_id)
                inst.append(utt_ids+prsn_ids)
            if num_prsn < self.args.max_num_prsn:
                for i in range(self.args.max_num_prsn-num_prsn):
                    inst.append([ self.tok.pad_token_id]*(self.args.max_utt_len+self.args.max_prsn_len))

        if num_utt < self.args.max_num_uttr:
            for i in range(self.args.max_num_uttr-num_utt):
                for _ in range(self.args.max_num_prsn):
                    inst.append([ self.tok.pad_token_id]*(self.args.max_utt_len+self.args.max_prsn_len))

        return inst
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        this_inst = {"dialog_id": example["id"]}
        this_inst["input_ids"] = []
        this_inst["input_ids"].extend(self.generate_inst(example["dialogue"], example["persona"]))
        for cand in example["candidate"]:
            this_inst["input_ids"].extend(self.generate_inst(example["dialogue"], cand))
        this_inst["label_idx"] = 0
        
        return this_inst

    def collate_fn(self, batch):
    
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = int(len(batch[0]["input_ids"])/ (self.args.max_num_uttr*self.args.max_num_prsn))
        input_ids = torch.tensor(input_ids).view(batch_size, -1, self.args.max_utt_len+self.args.max_prsn_len)
        label_idx = torch.tensor(label_idx)
        return input_ids, label_idx, data_info
