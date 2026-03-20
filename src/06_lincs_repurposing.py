"""
06_lincs_repurposing.py  —  AIHA scRNA-seq pipeline
LINCS L1000 transcriptomic reversal scoring via Enrichr.
Queries: Diagnosis vs Remission (primary), RR vs Remission (secondary),
         Diagnosis vs RR (tertiary), cluster-level, T cell subset,
         erythroid subset, pro-AIHA clusters.
"""
import os, time, re
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp

sc.settings.verbosity = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

OUT_PATH = os.path.join(PROCESSED, "06_lincs.csv")
if os.path.exists(OUT_PATH):
    print("06_lincs.csv already exists — skipping Script 06.")
    import sys; sys.exit(0)

LINCS_LIB  = "LINCS_L1000_Chem_Pert_Consensus_Sigs"
TOP_N      = 150    # genes per query
SLEEP      = 0.4    # seconds between API calls

# ── Compound name cleaning ────────────────────────────────────────────────────
CLEAN_RE = re.compile(r'^LJP\d+\s+\S+\s+\S+?-(.+)-[\d.]+$')
def clean_compound(name):
    m = CLEAN_RE.match(str(name))
    return m.group(1) if m else str(name)

# ── Enrichr query ─────────────────────────────────────────────────────────────
def enrichr_query(gene_list, label):
    if not gene_list:
        return pd.DataFrame()
    try:
        enr = gp.enrichr(gene_list=gene_list, gene_sets=LINCS_LIB,
                         outdir=None, verbose=False)
        df = enr.results.copy()
        df["query"] = label
        time.sleep(SLEEP)
        return df
    except Exception as e:
        print(f"    Enrichr error ({label}): {e}")
        return pd.DataFrame()

# ── Load DE gene lists ────────────────────────────────────────────────────────
print("Loading DE gene lists...")
adata = sc.read_h5ad(os.path.join(PROCESSED, "05_de.h5ad"))
print(f"  {adata.n_obs:,} cells | {adata.obs['leiden'].nunique()} clusters")

def load_de(fname, group_col="names", fc_col="logfoldchanges", top=TOP_N):
    path = os.path.join(PROCESSED, fname)
    if not os.path.exists(path):
        return [], []
    df = pd.read_csv(path)
    if group_col not in df.columns or fc_col not in df.columns:
        return [], []
    up   = df[df[fc_col] > 0].nlargest(top, fc_col)[group_col].tolist()
    down = df[df[fc_col] < 0].nsmallest(top, fc_col)[group_col].tolist()
    return up, down

all_results = []

# Primary
up, dn = load_de("de_diagnosis_vs_remission.csv")
print(f"\nPrimary query: Diagnosis vs Remission...")
r = enrichr_query(up + dn, "diagnosis_vs_remission")
if not r.empty:
    print(f"  {len(r)} reversal hits")
    all_results.append(r)

# Secondary
up, dn = load_de("de_rr_vs_remission.csv")
print("Secondary query: RR vs Remission...")
r = enrichr_query(up + dn, "rr_vs_remission")
if not r.empty:
    print(f"  {len(r)} reversal hits")
    all_results.append(r)

# Tertiary
up, dn = load_de("de_diagnosis_vs_rr.csv")
print("Tertiary query: Diagnosis vs RR...")
r = enrichr_query(up + dn, "diagnosis_vs_rr")
if not r.empty:
    print(f"  {len(r)} reversal hits")
    all_results.append(r)

# Subset queries
for fname, label in [
    ("de_tcell_subset.csv",       "tcell_subset"),
    ("de_erythroid_subset.csv",   "erythroid_subset"),
    ("de_pro_aiha_clusters.csv",  "pro_aiha_clusters"),
]:
    up, dn = load_de(fname)
    if up or dn:
        print(f"  {label}...")
        r = enrichr_query(up + dn, label)
        if not r.empty:
            print(f"    {len(r)} hits")
            all_results.append(r)

# Cluster-level queries
df_cluster = pd.read_csv(os.path.join(PROCESSED, "de_cluster_vs_rest.csv"))
clusters = df_cluster["group"].unique() if "group" in df_cluster.columns else []
for i, cl in enumerate(clusters):
    df_cl = df_cluster[df_cluster["group"] == cl] if "group" in df_cluster.columns else pd.DataFrame()
    up   = df_cl.nlargest(TOP_N, "logfoldchanges")["names"].tolist() if not df_cl.empty else []
    dn   = df_cl.nsmallest(TOP_N, "logfoldchanges")["names"].tolist() if not df_cl.empty else []
    ct   = adata.obs.loc[adata.obs["leiden"] == str(cl), "cell_type"].iloc[0] \
           if str(cl) in adata.obs["leiden"].values else "unknown"
    print(f"  Cluster {cl} ({i+1}/{len(clusters)})...", end=" ")
    r = enrichr_query(up + dn, f"cluster_{cl}")
    if not r.empty:
        print(f"{len(r)} hits")
        all_results.append(r)
    else:
        print("no hits")

# ── Aggregate results ─────────────────────────────────────────────────────────
if not all_results:
    print("No LINCS results — empty output.")
    pd.DataFrame().to_csv(OUT_PATH, index=False)
    import sys; sys.exit(0)

raw = pd.concat(all_results, ignore_index=True)
print(f"\nRaw results: {len(raw):,} rows")

# Clean compound names
raw["compound"] = raw["Term"].apply(clean_compound)

# Reversal score = negative Combined Score (reversal = compound opposes disease)
raw["reversal_score"] = pd.to_numeric(raw.get("Combined Score", raw.get("combined_score", 0)),
                                       errors="coerce").fillna(0)

# Deduplicate: max reversal score + count queries
summary = (raw.groupby("compound")
             .agg(max_reversal_score=("reversal_score", "max"),
                  n_queries=("query", "nunique"))
             .reset_index()
             .query("max_reversal_score > 12")
             .sort_values("max_reversal_score", ascending=False)
             .reset_index(drop=True))

print(f"Unique compounds: {len(summary)}")
print(summary.head(15).to_string(index=False))

summary.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print("Script 06 complete")
