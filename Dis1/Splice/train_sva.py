# splice_sva.py
import os, argparse, random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

class Example:
    def __init__(self, idx, desc, code, sev):
        self.idx, self.desc, self.code, self.sev = idx, desc, code, sev

def read_examples(path):
    df = pd.read_excel(path)
    out = []
    for i, r in df.iterrows():
        desc = str(r.get("description","")); code = str(r.get("abstract_func_before",""))
        sev  = int(r.get("severity"))
        out.append(Example(i, desc, code, sev))
    return out

class SpliceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=384, tok_desc="[<DESC>]", tok_code="[<CODE>]"):
        self.examples, self.tk, self.L, self.td, self.tc = examples, tokenizer, max_len, tok_desc, tok_code
    def __len__(self): return len(self.examples)
    def _segments(self, ids):
        if len(ids)<=self.L: return [ids]
        segs=[]; i=0
        while i < len(ids):
            segs.append(ids[i:i+self.L]); i += self.L
        return segs
    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = f"{self.td} {ex.desc} {self.tc} {ex.code}"
        ids = self.tk(seq, add_special_tokens=True, truncation=False)["input_ids"]
        segs = self._segments(ids)
        return {"segments": segs, "y": ex.sev, "sid": ex.idx}

def collate(batch, pad_id):
    segs, owners, y, sid = [], [], [], []
    for i,b in enumerate(batch):
        for s in b["segments"]:
            segs.append(s); owners.append(i)
        y.append(b["y"]); sid.append(b["sid"])
    max_len = max(len(s) for s in segs) if segs else 0
    if segs:
        ids, attn = [], []
        for s in segs:
            L=len(s)
            ids.append(s + [pad_id]*(max_len-L))
            attn.append([1]*L + [0]*(max_len-L))
        ids  = torch.tensor(ids, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)
        owners = torch.tensor(owners, dtype=torch.long)
    else:
        ids  = torch.empty(0, dtype=torch.long)
        attn = torch.empty(0, dtype=torch.long)
        owners = torch.empty(0, dtype=torch.long)
    return {
        "seg_ids": ids,
        "seg_attn": attn,
        "owners": owners,
        "y": torch.tensor(y, dtype=torch.long),
        "sid": torch.tensor(sid, dtype=torch.long),
    }

class CodeT5Encoder(torch.nn.Module):
    def __init__(self, name): super().__init__(); self.encoder = T5EncoderModel.from_pretrained(name)
    def forward(self, ids, attn): return self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state

class SVAHead(torch.nn.Module):
    def __init__(self, hidden, C): super().__init__(); self.fc = torch.nn.Linear(hidden, C)
    def forward(self, z): return self.fc(z)

class Model(torch.nn.Module):
    def __init__(self, name, C):
        super().__init__()
        self.enc = CodeT5Encoder(name)
        H = self.enc.encoder.config.d_model
        self.head = SVAHead(H, C)
    def seg_embed(self, seg_ids, seg_attn):
        if seg_ids.numel()==0:
            return torch.empty(0, self.enc.encoder.config.d_model, device=seg_ids.device)
        h = self.enc(seg_ids, seg_attn)
        denom = seg_attn.sum(dim=1).clamp_min(1).unsqueeze(-1)
        hk = (h * seg_attn.unsqueeze(-1)).sum(dim=1) / denom
        return hk
    def forward(self, seg_ids, seg_attn, owners):
        hk = self.seg_embed(seg_ids, seg_attn)
        uniq = owners.unique()
        z_list = []
        for i in uniq:
            m = owners==i
            z_i = hk[m].mean(dim=0)     # 简单平均（不使用注意力）
            z_list.append(z_i)
        z = torch.stack(z_list, dim=0)
        return self.head(z)

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    Y,P=[],[]
    for b in tqdm(dl, desc="Eval"):
        ids, attn, owners, y = b["seg_ids"].to(device), b["seg_attn"].to(device), b["owners"].to(device), b["y"].to(device)
        logits = model(ids, attn, owners)
        pred = logits.argmax(-1)
        Y += y.cpu().tolist(); P += pred.cpu().tolist()
    acc = accuracy_score(Y,P); p,r,f1,_ = precision_recall_fscore_support(Y,P,average="macro",zero_division=0); mcc = matthews_corrcoef(Y,P)
    return {"acc":acc,"precision":p,"recall":r,"f1":f1,"mcc":mcc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5-base")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs_sva_splice")
    ap.add_argument("--severity_classes", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tok = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tok.add_special_tokens({"additional_special_tokens": ["[<DESC>]","[<CODE>]"]})
    pad_id = tok.pad_token_id

    train_ex = read_examples(args.train_file)
    val_ex   = read_examples(args.val_file)

    train_ds = SpliceDataset(train_ex, tok, max_len=args.max_len)
    val_ds   = SpliceDataset(val_ex, tok, max_len=args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    val_dl   = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(args.model_name_or_path, args.severity_classes).to(device)
    model.enc.encoder.resize_token_embeddings(len(tok))

    opt = AdamW(model.parameters(), lr=args.lr)
    tot_steps = len(train_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, 0, tot_steps)
    ce = torch.nn.CrossEntropyLoss()

    best_f1, bad = 0.0, 0
    for e in range(1, args.epochs+1):
        model.train()
        for bt in tqdm(train_dl, desc=f"SVA-Splice Ep{e}"):
            ids, attn, owners, y = bt["seg_ids"].to(device), bt["seg_attn"].to(device), bt["owners"].to(device), bt["y"].to(device)
            loss = ce(model(ids, attn, owners), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); opt.zero_grad()
        m = evaluate(model, val_dl, device)
        if m["f1"] > best_f1:
            best_f1, bad = m["f1"], 0
            torch.save(model.enc.state_dict(), os.path.join(args.output_dir, "sva_encoder.pt"))
        else:
            bad += 1
            if bad >= args.patience: break

if __name__ == "__main__":
    main()
