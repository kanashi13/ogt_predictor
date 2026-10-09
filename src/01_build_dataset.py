import sqlite3, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "raw" / "brenda.sql"         
OUT = ROOT / "data" / "dataset_ogt.csv"
if OUT.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{OUT} already exists; pass --force to overwrite")

con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)

meta = pd.read_sql("""
    SELECT MIN(id) AS id, uniprot_id, MIN(organism) AS organism,
           MIN(domain) AS domain, MIN(ogt) AS ogt, MIN(LENGTH(sequence)) AS len
    FROM annotation
    WHERE ogt_note='experimental' AND domain IN ('Bacteria','Archaea')
    GROUP BY uniprot_id
""", con)

meta = meta[(meta.len >= 100) & (meta.len <= 1000)]
meta["genus"] = meta["organism"].str.split("_").str[0]

meta = meta.sample(frac=1, random_state=42)
cap = (meta["ogt"] >= 45).map({True: 20, False: 6})
meta["rank"] = meta.groupby("organism").cumcount()
sel = meta[meta["rank"] < cap].drop(columns="rank").copy()

ids = sel["id"].tolist()
seqs = []
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    q = f"SELECT id, sequence FROM annotation WHERE id IN ({','.join('?' * len(chunk))})"
    seqs.append(pd.read_sql(q, con, params=chunk))
sel = sel.merge(pd.concat(seqs), on="id")
sel = sel[sel.sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")]

sel.to_csv(OUT, index=False)
print(sel.shape, "organisms:", sel.organism.nunique(), "genera:", sel.genus.nunique(),
      "OGT>=45:", round((sel.ogt >= 45).mean(), 3))