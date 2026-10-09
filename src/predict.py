import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from transformers import AutoTokenizer, EsmModel
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = (ROOT / "models--facebook--esm2_t33_650M_UR50D" / "snapshots"
             / "08e4846e537177426273712802403f7ba8261b6c")
ESM_REPO = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
LOCAL_ESM = MODEL_DIR.exists()
ESM_SRC = str(MODEL_DIR) if LOCAL_ESM else ESM_REPO
ESM_KW = {"local_files_only": True} if LOCAL_ESM else {"revision": ESM_REVISION}

FM = ROOT / "final_model"
VALID = set("ACDEFGHIKLMNPQRSTVWY")

def read_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None: recs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name is not None: recs.append((name, "".join(buf)))
    return recs

def clean(seq):
    return "".join(seq.split()).upper().rstrip("*")

@torch.no_grad()
def embed(seqs, max_tok=6000):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ESM_SRC, **ESM_KW)
    model = EsmModel.from_pretrained(ESM_SRC, add_pooling_layer=False, **ESM_KW)
    model = (model.half() if dev == "cuda" else model).to(dev).eval()
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), 1280), dtype=np.float32)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]]) + 2) <= max_tok:
            j += 1
        idx = order[i:j]
        b = tok([seqs[k] for k in idx], return_tensors="pt", padding=True).to(dev)
        h = model(**b).last_hidden_state.float()
        mask = b["attention_mask"].clone()
        mask[:, 0] = 0
        mask[torch.arange(len(idx)), b["attention_mask"].sum(1) - 1] = 0
        m = mask.unsqueeze(-1).float()
        out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        i = j
    return out

def main():
    ap = argparse.ArgumentParser(description="Predict organism growth temperature from protein sequence(s)")
    ap.add_argument("--seq", nargs="*", default=[], help="one or more raw sequences")
    ap.add_argument("--fasta", help="FASTA file")
    ap.add_argument("--out", help="save results to CSV")
    ap.add_argument("--aggregate", action="store_true",
                    help="treat all input sequences as the SAME organism and report their mean")
    args = ap.parse_args()

    recs = [(f"seq{i+1}", s) for i, s in enumerate(args.seq)]
    if args.fasta: recs += read_fasta(args.fasta)
    if not recs: ap.error("give --seq or --fasta")

    names, seqs = [], []
    for n, s in recs:
        s = clean(s)
        if not s or set(s) - VALID:
            print(f"WARNING: skipping {n} (empty or non-standard amino acids)"); continue
        if len(s) > 1022:
            print(f"WARNING: {n} truncated to 1022 aa"); s = s[:1022]
        elif not (100 <= len(s) <= 1000):
            print(f"WARNING: {n} length {len(s)} is outside the training range (100-1000 aa)")
        names.append(n); seqs.append(s)
    if not seqs: raise SystemExit("no valid sequences")

    X = embed(seqs)
    ridge = joblib.load(FM / "ridge.joblib")
    svr = joblib.load(FM / "svr_pca.joblib")
    xgb = XGBRegressor(); xgb.load_model(str(FM / "xgb.json")); xgb.set_params(device="cpu")

    res = pd.DataFrame({"id": names, "length": [len(s) for s in seqs],
                        "ridge": ridge.predict(X), "xgboost": xgb.predict(X),
                        "svr": svr.predict(X)})
    res["pred_OGT"] = res[["ridge", "xgboost", "svr"]].mean(axis=1)
    print(res.round(1).to_string(index=False))
    if args.aggregate and len(res) > 1:
        print(f"\nOrganism-level estimate (mean of {len(res)} proteins): "
              f"{res.pred_OGT.mean():.1f} C")
    if args.out:
        res.to_csv(args.out, index=False); print("saved", args.out)

if __name__ == "__main__":
    main()