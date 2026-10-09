from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
X = np.load(ROOT / "data" / "X_classical_ogt.npy")
y, groups = df["ogt"].values, df["genus"].values

AA = "ACDEFGHIKLMNPQRSTVWY"
phys = ["log_length", "GRAVY", "IVYWREL", "(E+K)/(Q+H)", "CvP", "aromatic(FWY)", "net_charge(KR-DE)"]
names = [f"AAC_{a}" for a in AA] + [f"DP_{a}{b}" for a in AA for b in AA] + phys
assert len(names) == X.shape[1]

tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, groups))
a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=1).split(X[tr], y[tr], groups[tr]))
xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.8,
                   colsample_bytree=0.3, min_child_weight=5, reg_lambda=5,
                   tree_method="hist", device="cuda", early_stopping_rounds=50,
                   importance_type="gain", random_state=42)
xgb.fit(X[tr][a], y[tr][a], eval_set=[(X[tr][b], y[tr][b])], verbose=False)
p = xgb.predict(X[te])
print(f"classical-only XGBoost (genus split): MAE={mean_absolute_error(y[te], p):.2f} R2={r2_score(y[te], p):.3f}")

imp = pd.Series(xgb.feature_importances_, index=names).sort_values(ascending=False)
imp = imp / imp.sum()
print("\nTop 20 features (share of total gain):")
print(imp.head(20).round(4).to_string())
imp.to_csv(ROOT / "data" / "feature_importance_classical.csv")

top = imp.head(20)[::-1]
plt.figure(figsize=(6, 6))
plt.barh(top.index, top.values)
plt.xlabel("Relative importance (gain)")
plt.tight_layout(); plt.savefig(ROOT / "data" / "feature_importance.png", dpi=300)

idx = list(range(20)) + list(range(420, 427))
rows = [(names[i], spearmanr(X[:, i], y)[0]) for i in idx]
corr = pd.DataFrame(rows, columns=["feature", "spearman_rho"]).sort_values("spearman_rho")
print("\nSpearman correlation with OGT (AAC + physicochemical):")
print(corr.round(3).to_string(index=False))
corr.to_csv(ROOT / "data" / "feature_correlation.csv", index=False)