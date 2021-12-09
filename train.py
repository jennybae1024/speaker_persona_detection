import os, sys, json, pickle, random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (AdamW,
                          get_linear_schedule_with_warmup,
                          BartForConditionalGeneration,
                          BartTokenizer,
                          AutoModel,
                          AutoConfig)
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import SPDDataset
from model import u2p_model


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        # load train and dev sets
        self.train_dataset = SPDDataset(args, split="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size,
                                  collate_fn=self.train_dataset.collate_fn, shuffle=True, drop_last=True)
        self.dev_set  = SPDDataset(args, split="valid")
        self.dev_loader = DataLoader(self.dev_set , batch_size=args.batch_size,
                                  collate_fn=self.dev_set .collate_fn, drop_last=True)

        # load model
        self.model = u2p_model.to(self.args.device)
        self.model_best_params = {}

    def save(self, step):
        ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        model_to_save.save_pretrained(ckpt_path)
        torch.save(self.args, os.path.join(ckpt_path, "training_args.bin"))

        print(f"*** Model checkpoint saved to {ckpt_path} ***")


    def update_best(self, epoch, step, train_loss, global_step, epoch_end = None):
        dev_loss, dev_acc = self.dev()
        if epoch_end:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
                  f"train loss {train_loss :.4f}, dev loss {dev_loss:.4f}, dev acc {dev_acc:.4f} ***")
        else:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
              f"train loss {train_loss / step:.4f}, dev loss {dev_loss:.4f}, dev acc {dev_acc:.4f} ***")
        self.model_best_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if dev_loss < self.best_dev:
            self.best_dev = dev_loss
        self.save(global_step)

        return dev_loss, dev_acc

    def train(self):
        self.best_dev = 999999, 0

        tb_writer = SummaryWriter(self.args.output_dir)
        params = []
        params.append({'params': self.model.parameters(), 'lr': self.args.lr})
        # optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        t_total = len(self.train_loader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        optimizer = AdamW(params, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        self.best_dev, init_acc = self.dev()
        print(f"*** Initial dev loss: {self.best_dev:.4f}, dev acc {init_acc:.4f} ***")

        global_step = 0
        self.model.zero_grad()

        for epoch in range(self.args.num_epochs):
            train_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader), start=1):
                self.model.train()
                model_outputs = self.model(batch)
                loss = model_outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()
                global_step += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()

                # evaluate dev loss every 500 steps
                if step % (self.args.dev_at_step) == 0:
                    dev_loss, dev_acc = self.update_best(epoch, step, train_loss, global_step)
                    tb_writer.add_scalar("eval_loss", dev_loss, global_step)
                    tb_writer.add_scalar("eval_acc", dev_acc, global_step)
                    tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train/loss", train_loss/step, global_step)

            # scheduler.step()
            train_loss /= step
            tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("train/loss", train_loss, global_step)

            dev_loss, dev_acc = self.update_best(epoch, step, train_loss, global_step, epoch_end=True)
            tb_writer.add_scalar("eval_loss", dev_loss, global_step)
            tb_writer.add_scalar("eval_acc", dev_acc, global_step)

    # calculate dev loss and print intermediate outputs
    def dev(self):
        self.model.eval()
        tot_loss = 0
        gold_labels = []
        preds = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dev_loader), start=1):
                labels = batch[1]
                model_outputs = self.model(batch)
                loss = model_outputs[0]
                gold_labels.extend(labels.detach().cpu().numpy())
                preds.extend(np.argmax(model_outputs[1].detach().cpu().numpy(), 1))
                tot_loss += loss.item()

            eval_loss = tot_loss / step
            eval_acc = sum(np.array(gold_labels) == np.array(preds)) / len(gold_labels)

            return eval_loss, eval_acc


def main():
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument("--max_num_uttr", type=int, default=4)
    parser.add_argument("--max_num_prsn", type=int, default=5)
    parser.add_argument("--max_utt_len", type=int, default=20)
    parser.add_argument("--max_prsn_len", type=int, default=15)
    
    # basic settings
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased')
    parser.add_argument("--output_dir", type=str, default='outputs')

    # training params
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--dev_at_step", type=float, default=0.0)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Trainer.train()

if __name__ == '__main__':
    main()
