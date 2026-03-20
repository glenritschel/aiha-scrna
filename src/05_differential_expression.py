"""
05_differential_expression.py  —  AIHA scRNA-seq pipeline
Three DE comparisons:
  Primary   : Diagnosis vs Remission   (active AIHA vs resolved)
  Secondary : Relapse_Refractory vs Remission  (treatment-resistant vs resolved)
  Tertiary  : Diagnosis vs Relapse_Refractory  (onset vs chronic refractory)
Plus cluster-vs-rest, erythroid subset DE, and T cell subset DE.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

OUT_PATH = os.path.join(PROCESSED, "05_de.h5ad")
if os.path.exists(OUT_PATH):
    print("05_de.h5ad already exists — skipping Script 05.")
    import sys; sys.exit(0)

print("Loading 04_scored.h5ad...")
adata = sc.read_h5ad(os.path.join(PROCESSED, "04_scored.h5ad"))
n_clusters = adata.obs["leiden"].nunique()
print(f"  {adata.n_obs:,} cells, {n_clusters} clusters")
print(f"  Conditions: {adata.obs['condition'].value_counts().to_dict()}")

DE_PARAMS = dict(method="wilcoxon", use_raw=True)

# ── Primary: Diagnosis vs Remission ──────────────────────────────────────────
print("\nDiagnosis vs Remission DE (primary)...")
mask_primary = adata.obs["condition"].isin(["Diagnosis", "Remission"])
adata_primary = adata[mask_primary].copy()
sc.tl.rank_genes_groups(adata_primary, "condition", groups=["Diagnosis"],
                         reference="Remission", **DE_PARAMS)
df_primary = sc.get.rank_genes_groups_df(adata_primary, group="Diagnosis",
                                          pval_cutoff=0.05, log2fc_min=0.25)
df_primary.to_csv(os.path.join(PROCESSED, "de_diagnosis_vs_remission.csv"), index=False)
top_up   = df_primary.nlargest(5, "logfoldchanges")["names"].tolist()
top_down = df_primary.nsmallest(5, "logfoldchanges")["names"].tolist()
print(f"  Top up:   {top_up}")
print(f"  Top down: {top_down}")

# ── Secondary: Relapse_Refractory vs Remission ────────────────────────────────
print("\nRelapse_Refractory vs Remission DE (secondary)...")
mask_rr = adata.obs["condition"].isin(["Relapse_Refractory", "Remission"])
adata_rr = adata[mask_rr].copy()
sc.tl.rank_genes_groups(adata_rr, "condition", groups=["Relapse_Refractory"],
                         reference="Remission", **DE_PARAMS)
df_rr = sc.get.rank_genes_groups_df(adata_rr, group="Relapse_Refractory",
                                     pval_cutoff=0.05, log2fc_min=0.25)
df_rr.to_csv(os.path.join(PROCESSED, "de_rr_vs_remission.csv"), index=False)
print(f"  Top up (RR vs Remission): {df_rr.nlargest(5, 'logfoldchanges')['names'].tolist()}")

# ── Tertiary: Diagnosis vs Relapse_Refractory ────────────────────────────────
print("\nDiagnosis vs Relapse_Refractory DE (tertiary)...")
mask_diag_rr = adata.obs["condition"].isin(["Diagnosis", "Relapse_Refractory"])
adata_diag_rr = adata[mask_diag_rr].copy()
sc.tl.rank_genes_groups(adata_diag_rr, "condition", groups=["Diagnosis"],
                         reference="Relapse_Refractory", **DE_PARAMS)
df_diag_rr = sc.get.rank_genes_groups_df(adata_diag_rr, group="Diagnosis",
                                          pval_cutoff=0.05, log2fc_min=0.25)
df_diag_rr.to_csv(os.path.join(PROCESSED, "de_diagnosis_vs_rr.csv"), index=False)
print(f"  Top up (Diag vs RR): {df_diag_rr.nlargest(5, 'logfoldchanges')['names'].tolist()}")

# ── Cluster-vs-rest ───────────────────────────────────────────────────────────
print("\nCluster-vs-rest DE...")
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=True)
df_cluster = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.05)
df_cluster.to_csv(os.path.join(PROCESSED, "de_cluster_vs_rest.csv"), index=False)
print(f"  {len(df_cluster):,} gene-cluster pairs")

# ── Erythroid subset DE ───────────────────────────────────────────────────────
ery_types = ["erythroid_progenitor", "mature_erythroid", "erythroid_progenitor_mixed",
             "mature_erythroid_mixed", "hsc_progenitor"]
mask_ery = adata.obs["cell_type"].isin(ery_types)
n_ery = mask_ery.sum()
print(f"\nErythroid subset: {n_ery:,} cells")
if n_ery >= 30:
    adata_ery = adata[mask_ery].copy()
    conds_ery = adata_ery.obs["condition"].unique().tolist()
    if "Diagnosis" in conds_ery and "Remission" in conds_ery:
        adata_ery_filt = adata_ery[adata_ery.obs["condition"].isin(["Diagnosis","Remission"])].copy()
        sc.tl.rank_genes_groups(adata_ery_filt, "condition", groups=["Diagnosis"],
                                 reference="Remission", **DE_PARAMS)
        df_ery = sc.get.rank_genes_groups_df(adata_ery_filt, group="Diagnosis",
                                              pval_cutoff=0.05, log2fc_min=0.25)
        df_ery.to_csv(os.path.join(PROCESSED, "de_erythroid_subset.csv"), index=False)
        print(f"  Erythroid DE saved ({len(df_ery)} genes)")
else:
    print("  Insufficient erythroid cells — skipping subset DE")

# ── T cell subset DE ──────────────────────────────────────────────────────────
t_types = ["t_cell", "cytotoxic_t", "exhausted_t", "regulatory_t",
           "t_cell_mixed", "cytotoxic_t_mixed", "exhausted_t_mixed"]
mask_t = adata.obs["cell_type"].isin(t_types)
n_t = mask_t.sum()
print(f"\nT cell subset: {n_t:,} cells")
if n_t >= 30:
    adata_t = adata[mask_t].copy()
    conds_t = adata_t.obs["condition"].unique().tolist()
    if "Diagnosis" in conds_t and "Remission" in conds_t:
        adata_t_filt = adata_t[adata_t.obs["condition"].isin(["Diagnosis","Remission"])].copy()
        sc.tl.rank_genes_groups(adata_t_filt, "condition", groups=["Diagnosis"],
                                 reference="Remission", **DE_PARAMS)
        df_t = sc.get.rank_genes_groups_df(adata_t_filt, group="Diagnosis",
                                            pval_cutoff=0.05, log2fc_min=0.25)
        df_t.to_csv(os.path.join(PROCESSED, "de_tcell_subset.csv"), index=False)
        print(f"  T cell DE saved ({len(df_t)} genes)")
else:
    print("  Insufficient T cells — skipping subset DE")

# ── Pro-AIHA cluster DE ───────────────────────────────────────────────────────
cluster_scores = adata.obs.groupby("leiden")[
    ["erythrophagocytosis_stress", "t_cell_exhaustion_dysregulation",
     "type_I_interferon_response"]].mean()
cluster_scores["composite"] = cluster_scores.sum(axis=1)
top_aiha_clusters = cluster_scores.nlargest(3, "composite").index.tolist()
print(f"\nPro-AIHA clusters: {top_aiha_clusters}")
mask_pro = adata.obs["leiden"].isin(top_aiha_clusters)
if mask_pro.sum() >= 30:
    adata_pro = adata[mask_pro].copy()
    conds_pro = adata_pro.obs["condition"].unique().tolist()
    if "Diagnosis" in conds_pro and "Remission" in conds_pro:
        adata_pro_filt = adata_pro[adata_pro.obs["condition"].isin(["Diagnosis","Remission"])].copy()
        sc.tl.rank_genes_groups(adata_pro_filt, "condition", groups=["Diagnosis"],
                                 reference="Remission", **DE_PARAMS)
        df_pro = sc.get.rank_genes_groups_df(adata_pro_filt, group="Diagnosis",
                                              pval_cutoff=0.05, log2fc_min=0.25)
        df_pro.to_csv(os.path.join(PROCESSED, "de_pro_aiha_clusters.csv"), index=False)
        print(f"  Pro-AIHA cluster DE saved ({len(df_pro)} genes)")

adata.write_h5ad(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print("Script 05 complete")
