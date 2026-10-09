from pathlib import Path
import json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "final_model"; out.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "dataset_ogt.csv")
X = np.load(ROOT / "data" / "X_esm_ogt.npy")
y, groups = df["ogt"].values, df["genus"].values

ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 20)))
ridge.fit(X, y); joblib.dump(ridge, out / "ridge.joblib"); print("ridge done", flush=True)

params = dict(learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.3,
              min_child_weight=5, reg_lambda=5, tree_method="hist", device="cuda",
              random_state=42)
a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=1).split(X, y, groups))
probe = XGBRegressor(n_estimators=2000, early_stopping_rounds=50, **params)
probe.fit(X[a], y[a], eval_set=[(X[b], y[b])], verbose=False)
n_trees = int((probe.best_iteration + 1) * 1.1)
xgb = XGBRegressor(n_estimators=n_trees, **params)
xgb.fit(X, y); xgb.save_model(out / "xgb.json"); print("xgb done, trees:", n_trees, flush=True)

svr = make_pipeline(StandardScaler(), PCA(128, random_state=0),
                    SVR(kernel="rbf", C=10, epsilon=1.0))
svr.fit(X, y); joblib.dump(svr, out / "svr_pca.joblib"); print("svr done", flush=True)

json.dump({"features": "ESM-2 650M mean-pooled (1280-d)", "n_train": int(len(y)),
           "xgb_n_trees": n_trees,
           "final_prediction": "mean(Ridge, XGBoost, SVR-PCA)",
           "target": "organism growth temperature (OGT, deg C)"},
          open(out / "meta.json", "w"), indent=2)
print("saved to", out)