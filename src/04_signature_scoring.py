"""
04_signature_scoring.py  —  AIHA scRNA-seq pipeline
Five AIHA-specific gene signatures scored per cell.
Signatures cover: erythrophagocytosis stress, complement/antibody activation,
T cell dysregulation, inflammatory cytokine axis, and type I IFN response.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

OUT_PATH = os.path.join(PROCESSED, "04_scored.h5ad")
if os.path.exists(OUT_PATH):
    print("04_scored.h5ad already exists — skipping Script 04.")
    import sys; sys.exit(0)

print("Loading 03_annotated.h5ad...")
adata = sc.read_h5ad(os.path.join(PROCESSED, "03_annotated.h5ad"))
print(f"  {adata.n_obs:,} cells × {adata.n_vars} genes")

# ── AIHA-specific gene signatures ─────────────────────────────────────────────
SIGNATURES = {
    # Erythrophagocytosis and erythroid stress: macrophages engulfing IgG-coated RBCs;
    # erythroid progenitor stress response
    "erythrophagocytosis_stress": [
        "TFRC", "HMOX1", "SLC40A1", "CYBRD1", "HAMP", "FTH1", "FTL",
        "EPOR", "KLF1", "GATA1", "HBA1", "HBB", "ALAS2", "EPB42"
    ],
    # Complement and antibody-mediated destruction: classical complement pathway,
    # Fc receptor activation — core AIHA effector pathway
    "complement_fcr_activation": [
        "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CR1", "CR3A",
        "FCGR1A", "FCGR2A", "FCGR2B", "FCGR3A", "FCGR3B", "CD64",
        "CD16", "CD32", "MRC1"
    ],
    # T cell exhaustion and dysregulation: central to AIHA pathogenesis;
    # cytotoxic T cells with exhaustion features predominate in active disease
    "t_cell_exhaustion_dysregulation": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1",
        "CD38", "GZMB", "PRF1", "IFNG", "CD8A", "KLRG1", "CD244"
    ],
    # Inflammatory cytokine axis: pro-inflammatory signaling driving AIHA flares
    "inflammatory_cytokine_axis": [
        "IL6", "TNF", "IL1B", "IL10", "IL2", "IL17A", "IFNG",
        "CXCL8", "CXCL10", "CCL2", "CCL5", "IL21", "IL4", "CSF1"
    ],
    # Type I interferon response: elevated in active AIHA; predicts disease activity
    "type_I_interferon_response": [
        "ISG15", "IFI44L", "IFIT1", "IFIT3", "MX1", "OAS1", "OAS2",
        "IRF7", "RSAD2", "IFI6", "OASL", "IFITM1", "IFITM3", "HERC5"
    ]
}

# ── Score per cell ─────────────────────────────────────────────────────────────
print("Scoring AIHA signatures:")
for sig_name, gene_list in SIGNATURES.items():
    genes_found   = [g for g in gene_list if g in adata.var_names]
    genes_missing = [g for g in gene_list if g not in adata.var_names]
    if genes_found:
        sc.tl.score_genes(adata, genes_found, score_name=sig_name)
        if genes_missing:
            print(f"  {sig_name}: {len(genes_found)}/{len(gene_list)} genes "
                  f"(missing: {genes_missing[:3]}{'...' if len(genes_missing) > 3 else ''})")
        else:
            print(f"  {sig_name}: {len(genes_found)}/{len(gene_list)} genes")
    else:
        print(f"  {sig_name}: no genes found — assigning 0")
        adata.obs[sig_name] = 0.0

sig_cols = list(SIGNATURES.keys())

# ── Pro-AIHA cluster identification ───────────────────────────────────────────
print("\nTop 3 pro-AIHA clusters:")
cluster_scores = adata.obs.groupby("leiden")[sig_cols].mean()
cluster_scores["composite"] = (
    cluster_scores["erythrophagocytosis_stress"] +
    cluster_scores["t_cell_exhaustion_dysregulation"] +
    cluster_scores["type_I_interferon_response"]
)
top3 = cluster_scores.nlargest(3, "composite")
for cl, row in top3.iterrows():
    ct = adata.obs.loc[adata.obs["leiden"] == cl, "cell_type"].iloc[0]
    print(f"  Cluster {cl} ({ct}): erythro={row['erythrophagocytosis_stress']:.4f}, "
          f"t_exhaust={row['t_cell_exhaustion_dysregulation']:.4f}")

# ── Condition comparison ──────────────────────────────────────────────────────
print("\nScores by condition:")
cond_scores = adata.obs.groupby("condition")[["n_cells"] + sig_cols] \
    .agg(lambda x: x.mean() if x.name != "n_cells" else x.count())
cond_scores["n_cells"] = adata.obs.groupby("condition").size()
print(cond_scores.to_string())

adata.write_h5ad(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print("Script 04 complete")
