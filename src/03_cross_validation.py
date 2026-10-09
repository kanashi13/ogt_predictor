from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

USE_SVR = True

ROOT = Path(__file__).resolve().parent.parent
X = np.load(ROOT / "data" / "X_esm_ogt.npy")
df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
y, groups = df["ogt"].values, df["genus"].values

# برچسب طبقه‌بندی فقط برای متعادل کردن foldها
strata = pd.cut(y, [0, 25, 35, 45, 60, 200], labels=False)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def metrics(yt, yp):
    hot = yt >= 45
    return dict(MAE=mean_absolute_error(yt, yp),
                RMSE=np.sqrt(mean_squared_error(yt, yp)),
                R2=r2_score(yt, yp),
                MAE_hot=mean_absolute_error(yt[hot], yp[hot]))

names = ["Ridge", "XGBoost"] + (["SVR-PCA"] if USE_SVR else []) + ["Ensemble"]
rows = []
oof = {n: np.zeros(len(y)) for n in names}

for k, (tr, te) in enumerate(cv.split(X, strata, groups), 1):
    assert not set(groups[tr]) & set(groups[te])          
    preds = {}

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 20)))
    ridge.fit(X[tr], y[tr]); preds["Ridge"] = ridge.predict(X[te])

    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=1)
                .split(X[tr], y[tr], groups[tr]))
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=5,
                       subsample=0.8, colsample_bytree=0.3, min_child_weight=5,
                       reg_lambda=5, tree_method="hist", device="cuda",
                       early_stopping_rounds=50, random_state=42)
    xgb.fit(X[tr][a], y[tr][a], eval_set=[(X[tr][b], y[tr][b])], verbose=False)
    preds["XGBoost"] = xgb.predict(X[te])

    if USE_SVR:
        svr = make_pipeline(StandardScaler(), PCA(128, random_state=0),
                            SVR(kernel="rbf", C=10, epsilon=1.0))
        svr.fit(X[tr], y[tr]); preds["SVR-PCA"] = svr.predict(X[te])

    preds["Ensemble"] = np.mean([preds[n] for n in names if n != "Ensemble"], axis=0)

    for n in names:
        oof[n][te] = preds[n]
        rows.append(dict(fold=k, model=n, n_test=len(te),
                         hot_frac=(y[te] >= 45).mean(), **metrics(y[te], preds[n])))
    e = rows[-1]
    print(f"fold {k}: n_test={len(te)} | Ensemble MAE={e['MAE']:.2f} R2={e['R2']:.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "data" / "cv5_results.csv", index=False)

print("\n=== mean ± std over 5 folds ===")
for n in names:
    r = res[res.model == n]
    print(f"{n:9s} MAE={r.MAE.mean():.2f}±{r.MAE.std():.2f}  "
          f"RMSE={r.RMSE.mean():.2f}±{r.RMSE.std():.2f}  "
          f"R2={r.R2.mean():.3f}±{r.R2.std():.3f}  "
          f"MAE(OGT>=45)={r.MAE_hot.mean():.2f}±{r.MAE_hot.std():.2f}")

np.save(ROOT / "data" / "oof_ensemble.npy", oof["Ensemble"])
plt.figure(figsize=(6, 6))
plt.scatter(y, oof["Ensemble"], s=5, alpha=0.25)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("True OGT (°C)"); plt.ylabel("Predicted OGT (°C), out-of-fold")
plt.tight_layout(); plt.savefig(ROOT / "data" / "scatter_cv5.png", dpi=300)
print("saved")