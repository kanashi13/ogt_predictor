from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
X_esm = np.load(ROOT / "data" / "X_esm_ogt.npy")
y, groups = df["ogt"].values, df["genus"].values

AA = "ACDEFGHIKLMNPQRSTVWY"
idx = {a: i for i, a in enumerate(AA)}
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))  # Kyte-Doolittle

def classical(seq):
    ids = np.fromiter((idx[c] for c in seq), dtype=np.int64, count=len(seq))
    L = len(seq)
    aac = np.bincount(ids, minlength=20) / L
    dip = np.bincount(ids[:-1] * 20 + ids[1:], minlength=400) / (L - 1)
    f = lambda s: sum(aac[idx[c]] for c in s)
    phys = [np.log(L),
            np.mean([KD[c] for c in seq]),                      
            f("IVYWREL"),
            (f("EK") + 1e-6) / (f("QH") + 1e-6),
            f("DEKR") - f("NQST"),                                
            f("FWY"),
            f("KR") - f("DE")]                                    
    return np.concatenate([aac, dip, phys])

X_cls = np.vstack([classical(s) for s in df.sequence]).astype(np.float32)
np.save(ROOT / "data" / "X_classical_ogt.npy", X_cls)
print("classical:", X_cls.shape, flush=True)

sets = {"ESM only": X_esm,
        "Classical only": X_cls,
        "ESM + Classical": np.hstack([X_esm, X_cls])}

strata = pd.cut(y, [0, 25, 35, 45, 60, 200], labels=False)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_esm, strata, groups))

def m(yt, yp):
    hot = yt >= 45
    return (mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp)),
            r2_score(yt, yp), mean_absolute_error(yt[hot], yp[hot]))

rows = []
for name, X in sets.items():
    for k, (tr, te) in enumerate(folds, 1):
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 20)))
        ridge.fit(X[tr], y[tr]); p_r = ridge.predict(X[te])

        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=1)
                    .split(X[tr], y[tr], groups[tr]))
        xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=5,
                           subsample=0.8, colsample_bytree=0.3, min_child_weight=5,
                           reg_lambda=5, tree_method="hist", device="cuda",
                           early_stopping_rounds=50, random_state=42)
        xgb.fit(X[tr][a], y[tr][a], eval_set=[(X[tr][b], y[tr][b])], verbose=False)
        p_x = xgb.predict(X[te])

        for mn, p in [("Ridge", p_r), ("XGBoost", p_x), ("Mean(R+X)", (p_r + p_x) / 2)]:
            mae, rmse, r2, mh = m(y[te], p)
            rows.append(dict(features=name, model=mn, fold=k, MAE=mae, RMSE=rmse, R2=r2, MAE_hot=mh))
        print(name, "fold", k, "done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "data" / "ablation_results.csv", index=False)
g = res.groupby(["features", "model"])[["MAE", "RMSE", "R2", "MAE_hot"]].agg(["mean", "std"]).round(3)
print(g.to_string())