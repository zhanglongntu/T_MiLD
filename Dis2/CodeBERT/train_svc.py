# coding: utf-8
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

class Example:
    def __init__(self, idx, desc, code, cwe_label):
        self.idx = idx
        self.desc = desc
        self.code = code
        self.cwe_label = cwe_label

def read_examples(excel_path):
    df = pd.read_excel(excel_path)
    out = []
    for i, r in df.iterrows():
        desc = str(r.get("description",""))
        code = str(r.get("abstract_func_before",""))
        cwe  = int(r.get("cwe_id"))
        out.append(Example(i, desc, code, cwe))
    return out

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
        enc = self.tk(seq, add_special_tokens=True, truncation=False, return_attention_mask=False)
        ids = enc["input_ids"]
        segments = self._windows(ids)
        return {"segments": segments, "label": ex.cwe_label, "sid": ex.idx}

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
        input_ids, attn = [], []
        for s in segs:
            a = [1]*len(s)
            if len(s) < max_len:
                pad = [pad_id]*(max_len-len(s))
                s = s + pad
                a = a + [0]*len(pad)
            input_ids.append(s)
            attn.append(a)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention = torch.tensor(attn, dtype=torch.long)
        owners = torch.tensor(owners, dtype=torch.long)
    else:
        input_ids = torch.empty(0, dtype=torch.long)
        attention = torch.empty(0, dtype=torch.long)
        owners    = torch.empty(0, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    sids   = torch.tensor(sids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "owners": owners, "labels": labels, "sids": sids}

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

class CodeBERTEncoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = RobertaModel.from_pretrained(name)
    def forward(self, ids, attn):
        return self.model(input_ids=ids, attention_mask=attn).last_hidden_state

class SVCHead(torch.nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(hidden, num_classes)
    def forward(self, z):
        return self.fc(z)

class SVC_System(torch.nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.enc = CodeBERTEncoder(name)
        h = self.enc.model.config.hidden_size
        self.agg = SegmentAggregator(h)
        self.head = SVCHead(h, num_classes)

    def seg_embed(self, ids, attn):
        if ids.numel() == 0:
            h = self.enc.model.config.hidden_size
            return torch.empty(0, h, device=ids.device)
        last = self.enc(ids, attn)
        denom = attn.sum(dim=1).clamp_min(1).unsqueeze(-1)
        hk = (last * attn.unsqueeze(-1)).sum(dim=1) / denom
        return hk

    def forward(self, ids, attn, owners):
        seg_embs = self.seg_embed(ids, attn)
        z = self.agg(seg_embs, owners)
        return self.head(z)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def set_trainable(module, flag=True):
    for p in module.parameters():
        p.requires_grad = flag

def group_roberta_layers(model):
    return list(model.enc.model.encoder.layer)

def focal_loss_from_probs(p, y_onehot, gamma=2.0, eps=1e-8):
    return -(((1 - p).clamp_min(0)**gamma) * y_onehot * (p.clamp_min(eps)).log()).sum(dim=-1).mean()

def lsce_from_probs(p, y_onehot, eps_smooth=0.1, eps=1e-8):
    y_smooth = (1 - eps_smooth) * y_onehot + eps_smooth / y_onehot.size(-1)
    return -(y_smooth * (p.clamp_min(eps)).log()).sum(dim=-1).mean()

def composite_loss(logits, y, epoch, E_max, gamma=2.0, eps_smooth=0.1):
    p = torch.softmax(logits, dim=-1)
    y_onehot = torch.nn.functional.one_hot(y, num_classes=logits.size(-1)).float()
    T = 1.0 - (epoch / float(E_max))**2
    fl = focal_loss_from_probs(p, y_onehot, gamma=gamma)
    ls = lsce_from_probs(p, y_onehot, eps_smooth=eps_smooth)
    return T * fl + (1 - T) * ls, {"T": T, "FL": float(fl.item()), "LS": float(ls.item())}

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    all_y, all_p = [], []
    for b in tqdm(dl, desc="Eval"):
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        owners = b["owners"].to(device)
        y = b["labels"].to(device)
        logits = model(ids, attn, owners)
        pred = logits.argmax(-1)
        all_y.extend(y.cpu().numpy().tolist())
        all_p.extend(pred.cpu().numpy().tolist())
    acc = accuracy_score(all_y, all_p)
    p, r, f1, _ = precision_recall_fscore_support(all_y, all_p, average="macro", zero_division=0)
    mcc = matthews_corrcoef(all_y, all_p)
    return {"acc": acc, "precision": p, "recall": r, "f1": f1, "mcc": mcc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs_cb")
    ap.add_argument("--encoder_ckpt", type=str, default="./outputs_cb/sva_encoder_codebert.pt")
    ap.add_argument("--cwe_classes", type=int, default=32)
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase1_epochs", type=int, default=6)
    ap.add_argument("--epochs_per_block", type=int, default=3)
    ap.add_argument("--max_unfreeze_layers", type=int, default=6)
    ap.add_argument("--enable_phase3", action="store_true", default=False)
    ap.add_argument("--phase3_epochs", type=int, default=5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--top_lr", type=float, default=2e-5)
    ap.add_argument("--head_top_lr", type=float, default=5e-5)
    ap.add_argument("--full_lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--E_max", type=int, default=40)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--eps_smooth", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    tok.add_special_tokens({"additional_special_tokens": ["[<DESC>]", "[<CODE>]"]})
    pad_id = tok.pad_token_id

    train_ex = read_examples(args.train_file)
    val_ex   = read_examples(args.val_file)

    train_ds = VulDataset(train_ex, tok, window=args.window, stride=args.stride)
    val_ds   = VulDataset(val_ex, tok, window=args.window, stride=args.stride)

    train_dl = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_dl   = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SVC_System(args.model_name_or_path, args.cwe_classes).to(device)
    model.enc.model.resize_token_embeddings(len(tok))

    if os.path.exists(args.encoder_ckpt):
        sd = torch.load(args.encoder_ckpt, map_location=device)
        model.enc.load_state_dict(sd, strict=False)

    for p in model.enc.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    opt = AdamW(model.head.parameters(), lr=args.head_lr)
    ce = torch.nn.CrossEntropyLoss()
    best_f1, bad = 0.0, 0

    for e in range(1, args.phase1_epochs + 1):
        model.train()
        for bt in tqdm(train_dl, desc=f"SVC P1 Epoch {e}"):
            ids = bt["input_ids"].to(device)
            attn = bt["attention_mask"].to(device)
            owners = bt["owners"].to(device)
            y = bt["labels"].to(device)
            logits = model(ids, attn, owners)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        metrics = evaluate(model, val_dl, device)
        if metrics["f1"] > best_f1:
            best_f1, bad = metrics["f1"], 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "svc_best_codebert.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    if os.path.exists(os.path.join(args.output_dir, "svc_best_codebert.pt")):
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "svc_best_codebert.pt"), map_location=device))

    layers = list(reversed(group_roberta_layers(model)))
    best_f1, bad = 0.0, 0
    total_epochs = 0

    for l_idx, layer in enumerate(layers[:args.max_unfreeze_layers]):
        set_trainable(layer, True)
        params = [
            {"params": layer.parameters(), "lr": args.top_lr},
            {"params": model.head.parameters(), "lr": args.head_top_lr}
        ]
        opt = AdamW(params)
        for ep in range(1, args.epochs_per_block + 1):
            total_epochs += 1
            model.train()
            for bt in tqdm(train_dl, desc=f"SVC P2 Layer{l_idx+1} Ep{ep}"):
                ids = bt["input_ids"].to(device)
                attn = bt["attention_mask"].to(device)
                owners = bt["owners"].to(device)
                y = bt["labels"].to(device)
                logits = model(ids, attn, owners)
                loss, _ = composite_loss(logits, y, epoch=total_epochs, E_max=args.E_max,
                                         gamma=args.gamma, eps_smooth=args.eps_smooth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            metrics = evaluate(model, val_dl, device)
            if metrics["f1"] > best_f1:
                best_f1, bad = metrics["f1"], 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "svc_best_codebert.pt"))
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if bad >= args.patience:
            break

    if args.enable_phase3:
        if os.path.exists(os.path.join(args.output_dir, "svc_best_codebert.pt")):
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "svc_best_codebert.pt"), map_location=device))
        for p in model.enc.parameters():
            p.requires_grad = True
        opt = AdamW(model.parameters(), lr=args.full_lr)
        best_f1, bad = 0.0, 0
        for ep in range(1, args.phase3_epochs + 1):
            total_epochs += 1
            model.train()
            for bt in tqdm(train_dl, desc=f"SVC P3 Epoch {ep}"):
                ids = bt["input_ids"].to(device)
                attn = bt["attention_mask"].to(device)
                owners = bt["owners"].to(device)
                y = bt["labels"].to(device)
                logits = model(ids, attn, owners)
                loss, _ = composite_loss(logits, y, epoch=total_epochs, E_max=args.E_max,
                                         gamma=args.gamma, eps_smooth=args.eps_smooth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            metrics = evaluate(model, val_dl, device)
            if metrics["f1"] > best_f1:
                best_f1, bad = metrics["f1"], 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "svc_best_codebert.pt"))
            else:
                bad += 1
                if bad >= args.patience:
                    break

if __name__ == "__main__":
    main()
