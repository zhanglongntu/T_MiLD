# truncation_sva.py
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

class TruncDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=384, tok_desc="[<DESC>]", tok_code="[<CODE>]"):
        self.examples, self.tk, self.L, self.td, self.tc = examples, tokenizer, max_len, tok_desc, tok_code
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = f"{self.td} {ex.desc} {self.tc} {ex.code}"
        enc = self.tk(seq, add_special_tokens=True, truncation=True, max_length=self.L)
        return {
            "input_ids": enc["input_ids"],
            "attn": enc["attention_mask"],
            "y": ex.sev,
            "sid": ex.idx
        }

def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, attn, y, sid = [], [], [], []
    for b in batch:
        L = len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id]*(max_len-L))
        attn.append(b["attn"] + [0]*(max_len-L))
        y.append(b["y"]); sid.append(b["sid"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attn": torch.tensor(attn, dtype=torch.long),
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
    def forward(self, ids, attn):
        h = self.enc(ids, attn)
        denom = attn.sum(dim=1).clamp_min(1).unsqueeze(-1)
        z = (h * attn.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(z)

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    Y, P = [], []
    for b in tqdm(dl, desc="Eval"):
        ids, attn, y = b["input_ids"].to(device), b["attn"].to(device), b["y"].to(device)
        logits = model(ids, attn)
        pred = logits.argmax(-1)
        Y += y.cpu().tolist(); P += pred.cpu().tolist()
    acc = accuracy_score(Y,P); p,r,f1,_ = precision_recall_fscore_support(Y,P,average="macro",zero_division=0); mcc = matthews_corrcoef(Y,P)
    return {"acc":acc,"precision":p,"recall":r,"f1":f1,"mcc":mcc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5-base")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs_sva_trunc")
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

    train_ds = TruncDataset(train_ex, tok, max_len=args.max_len)
    val_ds   = TruncDataset(val_ex, tok, max_len=args.max_len)

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
        for bt in tqdm(train_dl, desc=f"SVA-Trunc Ep{e}"):
            ids, attn, y = bt["input_ids"].to(device), bt["attn"].to(device), bt["y"].to(device)
            loss = ce(model(ids, attn), y)
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
