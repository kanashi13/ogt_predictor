# Predicting Growth Temperature of Microbes from a Protein Sequence

Give this tool the amino acid sequence of **one protein** from a bacterium or an archaeon.
It predicts the **optimal growth temperature (OGT, in °C)** of the organism that protein comes from.

> **Important:** the prediction is the temperature at which the *organism* grows best.
> It is **not** the optimal temperature of the individual enzyme (Topt).
> Every protein from the same organism has the same label.

---

## 1. Results at a glance

On average, the predicted temperature is about **5.6 °C** away from the true one (MAE),
and the model explains about **72%** of the variation in temperature (R²).

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Ridge | 6.06 ± 0.13 | 8.18 | 0.696 |
| XGBoost | 5.66 ± 0.19 | 7.89 | 0.717 |
| SVR (on PCA) | 5.52 ± 0.11 | 7.93 | 0.715 |
| **Final model (average of the three)** | **5.56 ± 0.13** | **7.78 ± 0.15** | **0.725 ± 0.030** |

*5-fold cross-validation, mean ± standard deviation over the folds.*

How to read the numbers:

- **MAE** = average size of the error in °C (smaller is better).
- **RMSE** = like MAE, but big errors count more (smaller is better).
- **R²** = share of the variation explained by the model (1 is perfect, 0 is no better than always predicting the average).
- For comparison, always predicting the average temperature gives an MAE of about 10.8 °C (measured on one test split).

Other findings (all tables and figures are in [`results/`](results/)):

- **Averaging over proteins helps.** If you average the predictions of several proteins from the *same organism*, the error drops to MAE 4.16 °C (RMSE 5.59, R² 0.694, 6,249 organisms).
- **ESM-2 is enough.** Adding 427 hand-made features (amino acid composition, dipeptides, hydrophobicity, charge, etc.) did not improve the result (R² 0.718 with and without them). The hand-made features alone reach R² 0.461.
- **Hot organisms are harder.** For organisms that grow at 45 °C or above, the error is about 10.7 °C, roughly twice the overall error.
- **Which simple features matter?** In the hand-made feature set, the most useful ones were the ratio (E+K)/(Q+H), the "CvP" bias, and the fraction of the amino acids I, V, Y, W, R, E, L (IVYWREL). Higher fractions of K, E, I and Y go with higher growth temperature; higher fractions of A, Q, T and H go with lower temperature. These are associations, not proof of cause.

---

## 2. How it works

1. **Data:** protein sequences with a known (experimentally measured) organism growth temperature.
2. **Embedding:** each sequence is turned into 1,280 numbers by the protein language model
   [ESM-2](https://github.com/facebookresearch/esm) (`esm2_t33_650M_UR50D`), averaged over the sequence.
3. **Three simple regression models** (Ridge, XGBoost, SVR) are trained on these numbers.
   The final prediction is the average of the three.
4. **Fair testing:** proteins from the same *genus* are never in both the training and the test set,
   so the model cannot just recognise close relatives.

---

## 3. Data

- Enzyme and UniProt information comes from BRENDA (release 2018.2, doi:10.1093/nar/gky1048).
- The annotated tables with growth temperatures and sequences (`annotated_brenda.tsv`, `brenda.sql`) come from
  Li, G. & Engqvist, M. K. M. (2019). *Enzymes from the BRENDA and CAZy databases annotated with organism growth temperatures and predicted Topt.* Zenodo.
  [doi:10.5281/zenodo.3578468](https://doi.org/10.5281/zenodo.3578468) (license: CC BY 4.0).
- Only rows with `ogt_note = experimental` were used as labels. The `topt` column in that table is
  computed by another machine-learning model, so it was **not** used.
- Final dataset: 37,058 proteins from 6,249 organisms (about 1,500 genera), bacteria and archaea only,
  length 100–1000 amino acids, at most 6 proteins per organism (20 for organisms with OGT ≥ 45 °C,
  to keep enough hot organisms). About 19% of the proteins have OGT ≥ 45 °C.
- The raw data are **not** stored in this repository. Download `annotated_brenda.tsv` and `brenda.sql`
  from the Zenodo link above and put them in `data/raw/`.

---

## 4. Project structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_build_dataset.py        build the dataset from the raw database
│   ├── 02_extract_esm.py          compute ESM-2 embeddings (needs an NVIDIA GPU)
│   ├── 03_cross_validation.py     5-fold cross-validation, main results
│   ├── 04_organism_level.py       average predictions per organism
│   ├── 05_ablation.py             compare ESM vs hand-made features
│   ├── 06_feature_importance.py   which hand-made features matter
│   ├── 07_train_final.py          train and save the final model
│   └── predict.py                 predict from your own sequences
├── results/                       result tables and figures
├── examples/example_input.fasta   example input file
├── data/                          NOT in the repo (see section 3)
└── final_model/                   NOT in the repo (download from Releases)
```

---

## 5. Installation

```bash
git clone https://github.com/kanashi13/ogt_predictor
cd ogt_predictor
pip install -r requirements.txt
```

- For GPU support, install PyTorch with CUDA following the instructions on [pytorch.org](https://pytorch.org/get-started/locally/).
- Developed and tested on Windows with an NVIDIA RTX 3050 (about 45 minutes for the embeddings of all 37,058 proteins).
- The ESM-2 weights (about 2.6 GB) are downloaded from Hugging Face the first time they are needed.

---

## 6. Quick start: predict from your own sequences

1. Go to the **Releases** page of this repository, download `final_model_v1.zip`,
   and unzip it so that the folder `final_model/` contains
   `ridge.joblib`, `svr_pca.joblib`, `xgb.json` and `meta.json`.
2. Run:

```bash
# a FASTA file with one or more proteins
python src/predict.py --fasta examples/example_input.fasta --out predictions.csv

# a single sequence typed directly
python src/predict.py --seq MKRVNAFNDLKRIGDDKVTAIGMGTWGIGGRETPDYSR...

# several proteins that all come from the SAME organism: also print their average
python src/predict.py --fasta my_organism_proteins.fasta --aggregate
```

The output table has one row per protein: the prediction of each of the three models
(`ridge`, `xgboost`, `svr`) and the final value `pred_OGT`.

Tips:

- Use proteins between 100 and 1000 amino acids (a warning appears otherwise).
- One protein gives a noisy answer. If you have several proteins from one organism, use `--aggregate`.
- The example sequences in `examples/` come from the training data, so their predictions look better than
  what you should expect for new proteins.
- Only load `.joblib` files from sources you trust (loading them can run code).
- The saved models were created with scikit-learn 1.8.0 and xgboost 3.4.1. Other versions may still run,
  but can print a version warning.

---

## 7. Reproduce everything from scratch

Run the scripts in this order from the repository folder:

```bash
python src/01_build_dataset.py      # needs data/raw/brenda.sql; creates data/dataset_ogt.csv
python src/02_extract_esm.py        # about 45 min on an RTX 3050; creates data/X_esm_ogt.npy
python src/03_cross_validation.py   # main results table and figure
python src/04_organism_level.py     # needs the output of step 03
python src/05_ablation.py           # also builds the hand-made features
python src/06_feature_importance.py # needs the output of step 05
python src/07_train_final.py        # creates final_model/
```

Tips:

- `python src/02_extract_esm.py 200` runs a quick test on the first 200 proteins only
  (the output file gets the suffix `_test200` so nothing is overwritten).
- `01_build_dataset.py` will not overwrite an existing dataset unless you add `--force`.
- Output files are written to `data/`. Copy the ones you want to keep into `results/`.

---

## 8. Limitations

- **Close relatives may still leak.** Splitting by genus helps, but different genera can have very similar proteins.
  A stricter check by sequence similarity (for example with CD-HIT) has not been done yet.
- **Predictions are pulled toward the average.** Cold-loving organisms are predicted too warm
  and extreme heat-lovers (above about 90 °C) too cool. The model separates "cool" from "hot" much better than it ranks
  organisms inside the 20–40 °C range.
- **Labels belong to organisms, not proteins.** All proteins of one organism share one temperature, so single-protein predictions are noisy.
- **Bacteria and archaea only.** Do not use it for eukaryotes.
- **Old database release.** The data are based on BRENDA 2018.2.

---

## 9. Citation and credits

If you use this work, please also cite the data source:

> Li, G. & Engqvist, M. K. M. (2019). Enzymes from the BRENDA and CAZy databases annotated with organism growth temperatures and predicted Topt. Zenodo. https://doi.org/10.5281/zenodo.3578468

and BRENDA (doi:10.1093/nar/gky1048).
The protein language model is ESM-2 from Meta AI (`facebook/esm2_t33_650M_UR50D`).

## 10. License

The code is released under the MIT License (see `LICENSE`).
The input data are distributed under CC BY 4.0 by their authors (see section 3)
