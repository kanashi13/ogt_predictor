from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
df["pred"] = np.load(ROOT / "data" / "oof_ensemble.npy")

org = df.groupby("organism").agg(ogt=("ogt", "first"), pred=("pred", "mean"),
                                 n=("pred", "size")).reset_index()

def rep(name, d):
    hot = d.ogt >= 45
    print(f"{name:22s} n_org={len(d):5d} MAE={mean_absolute_error(d.ogt, d.pred):.2f} "
          f"RMSE={np.sqrt(mean_squared_error(d.ogt, d.pred)):.2f} R2={r2_score(d.ogt, d.pred):.3f} "
          f"MAE(OGT>=45)={mean_absolute_error(d.ogt[hot], d.pred[hot]):.2f}")

rep("protein-level (per-org avg)", df.assign(pred=df.pred).rename(columns={}))
rep("organism-level (all)", org)
rep("organism-level (n>=3)", org[org.n >= 3])
rep("organism-level (n>=6)", org[org.n >= 6])

plt.figure(figsize=(6, 6))
plt.scatter(org.ogt, org.pred, s=8, alpha=0.4)
plt.plot([org.ogt.min(), org.ogt.max()], [org.ogt.min(), org.ogt.max()], "r--")
plt.xlabel("True OGT (°C)"); plt.ylabel("Predicted OGT (°C), organism mean")
plt.tight_layout(); plt.savefig(ROOT / "data" / "scatter_organism.png", dpi=300)