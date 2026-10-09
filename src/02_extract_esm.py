import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = (ROOT / "models--facebook--esm2_t33_650M_UR50D" / "snapshots"
             / "08e4846e537177426273712802403f7ba8261b6c")
ESM_REPO = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
LOCAL_ESM = MODEL_DIR.exists()
ESM_SRC = str(MODEL_DIR) if LOCAL_ESM else ESM_REPO
ESM_KW = {"local_files_only": True} if LOCAL_ESM else {"revision": ESM_REVISION}

N = int(sys.argv[1]) if len(sys.argv) > 1 else None   
MAX_TOK = 6000                                         

assert torch.cuda.is_available(), "CUDA not available: install a CUDA build of torch"
print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
if N: df = df.iloc[:N]
seqs = df.sequence.tolist()

tok = AutoTokenizer.from_pretrained(ESM_SRC, **ESM_KW)
model = EsmModel.from_pretrained(ESM_SRC, add_pooling_layer=False, **ESM_KW).half().cuda().eval()

order = np.argsort([len(s) for s in seqs])
out = np.zeros((len(seqs), 1280), dtype=np.float32)

i = 0
pbar = tqdm(total=len(seqs))
with torch.no_grad():
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]]) + 2) <= MAX_TOK:
            j += 1
        idx = order[i:j]
        b = tok([seqs[k] for k in idx], return_tensors="pt", padding=True).to("cuda")
        h = model(**b).last_hidden_state.float()
        mask = b["attention_mask"].clone()
        mask[:, 0] = 0
        mask[torch.arange(len(idx)), b["attention_mask"].sum(1) - 1] = 0
        m = mask.unsqueeze(-1).float()
        out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        pbar.update(len(idx)); i = j

suffix = f"_test{N}" if N else ""
np.save(ROOT / "data" / f"X_esm_ogt{suffix}.npy", out)
print(out.shape, "NaN:", int(np.isnan(out).sum()))