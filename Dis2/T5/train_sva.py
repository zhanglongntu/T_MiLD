# coding: utf-8
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

class Example:
    def __init__(self, idx, desc, code, sev_label):
        self.idx = idx
        self.desc = desc
        self.code = code
        self.sev_label = sev_label

def read_examples(excel_path):
    df = pd.read_excel(excel_path)
    exs = []
    for i, r in df.iterrows():
        desc = str(r.get("description", ""))
        code = str(r.get("abstract_func_before", ""))
        sev  = int(r.get("severity"))
        exs.append(Example(i, desc, code, sev))
    return exs

class VulDataset(Dataset):
    def __init__(self, examples, tokenizer, window=384, stride=192, tok_desc="[<DESC>]", tok_code="[<CODE>]"):
        self.examples = examples
        self.tk = tokenizer
        self.W = window
        self.S = stride
        self.td = tok_desc
        self.tc = tok_code

    def __len__(self):
        return len(self.examples)

    def _windows(self, ids):
        if len(ids) <= self.W:
            return [ids]
        segs, i = [], 0
        while True:
            segs.append(ids[i:i+self.W])
            if i + self.W >= len(ids):
                break
            i += self.S
        return segs

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = f"{self.td} {ex.desc} {self.tc} {ex.code}"
        ids = self.tk(seq, add_special_tokens=True, truncation=False)["input_ids"]
        segments = self._windows(ids)
        return {"segments": segments, "label": ex.sev_label, "sid": ex.idx}

def collate_fn(batch, pad_id):
    segs, owners, labels, sids = [], [], [], []
    for i, b in enumerate(batch):
        for s in b["segments"]:
            segs.append(s)
            owners.append(i)
        labels.append(b["label"])
        sids.append(b["sid"])
    max_len = max(len(s) for s in segs) if segs else 0
    if segs:
        input_ids = []
        for s in segs:
            if len(s) < max_len:
                s = s + [pad_id] * (max_len - len(s))
            input_ids.append(s)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        owners = torch.tensor(owners, dtype=torch.long)
    else:
        input_ids = torch.empty(0, dtype=torch.long)
        owners    = torch.empty(0, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    sids   = torch.tensor(sids, dtype=torch.long)
    return {"input_ids": input_ids, "owners": owners, "labels": labels, "sids": sids}

class SegmentAggregator(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj  = torch.nn.Linear(hidden, hidden)
        self.score = torch.nn.Linear(hidden, 1)
    def forward(self, seg_embs, owners):
        H = torch.tanh(self.proj(seg_embs))
        u = self.score(H).squeeze(-1)
        uniq = owners.unique()
        z_list = []
        for i in uniq:
            m = owners == i
            a = torch.softmax(u[m], dim=0)
            z_i = (a.unsqueeze(1) * seg_embs[m]).sum(dim=0)
            z_list.append(z_i)
        return torch.stack(z_list, dim=0)

class T5Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(name)
    def forward(self, ids, attn=None):
        return self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state

class SVAHead(torch.nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(hidden, num_classes)
    def forward(self, z):
        return self.fc(z)

class SVA_System(torch.nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.enc = T5Encoder(name)
        h = self.enc.encoder.config.d_model
        self.agg = SegmentAggregator(h)
        self.head = SVAHead(h, num_classes)

    def seg_embed(self, input_ids, pad_id):
        if input_ids.numel() == 0:
            return torch.empty(0, self.enc.encoder.config.d_model, device=input_ids.device)
        attn = (input_ids != pad_id).long()
        h = self.enc(input_ids, attn)
        denom = attn.sum(dim=1).clamp_min(1).unsqueeze(-1)
        hk = (h * attn.unsqueeze(-1)).sum(dim=1) / denom
        return hk

    def forward(self, input_ids, owners, pad_id):
        seg_embs = self.seg_embed(input_ids, pad_id)
        z = self.agg(seg_embs, owners)
        logits = self.head(z)
        return logits

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, dl, device, pad_id):
    model.eval()
    all_y, all_p = [], []
    for b in tqdm(dl, desc="Eval"):
        ids = b["input_ids"].to(device)
        owners = b["owners"].to(device)
        y = b["labels"].to(device)
        logits = model(ids, owners, pad_id)
        pred = logits.argmax(-1)
        all_y.extend(y.cpu().numpy().tolist())
        all_p.extend(pred.cpu().numpy().tolist())
    acc = accuracy_score(all_y, all_p)
    p, r, f1, _ = precision_recall_fscore_support(all_y, all_p, average="macro", zero_division=0)
    mcc = matthews_corrcoef(all_y, all_p)
    return {"acc": acc, "precision": p, "recall": r, "f1": f1, "mcc": mcc}

def freeze_encoder_all(model):
    for p in model.enc.parameters():
        p.requires_grad = False

def unfreeze_top_n_blocks(model, n):
    blocks = list(model.enc.encoder.encoder.block)
    for p in model.enc.parameters():
        p.requires_grad = False
    if n <= 0:
        return
    for b in blocks[-n:]:
        for p in b.parameters():
            p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="t5-base")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs_t5")
    ap.add_argument("--severity_classes", type=int, default=4)
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sva_unfreeze_top", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tok = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tok.add_special_tokens({"additional_special_tokens": ["[<DESC>]", "[<CODE>]"]})
    pad_id = tok.pad_token_id

    train_ex = read_examples(args.train_file)
    val_ex   = read_examples(args.val_file)

    train_ds = VulDataset(train_ex, tok, window=args.window, stride=args.stride)
    val_ds   = VulDataset(val_ex, tok, window=args.window, stride=args.stride)

    train_dl = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_dl   = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SVA_System(args.model_name_or_path, args.severity_classes).to(device)
    model.enc.encoder.resize_token_embeddings(len(tok))

    freeze_encoder_all(model)
    unfreeze_top_n_blocks(model, args.sva_unfreeze_top)

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    tot_steps = len(train_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, 0, tot_steps)
    ce = torch.nn.CrossEntropyLoss()

    best_f1, bad = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bt in tqdm(train_dl, desc=f"SVA Epoch {epoch}"):
            ids = bt["input_ids"].to(device)
            owners = bt["owners"].to(device)
            y = bt["labels"].to(device)
            logits = model(ids, owners, pad_id)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()

        metrics = evaluate(model, val_dl, device, pad_id)
        if metrics["f1"] > best_f1:
            best_f1, bad = metrics["f1"], 0
            torch.save(model.enc.state_dict(), os.path.join(args.output_dir, "sva_encoder_t5.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

if __name__ == "__main__":
    main()
