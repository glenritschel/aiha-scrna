"""
03_annotate_clusters.py  —  AIHA scRNA-seq pipeline
Cell type annotation for bone marrow aspirate using canonical marker genes.
BM-specific cell types: HSC/progenitors, erythroid lineage, myeloid, lymphoid, plasma.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

OUT_PATH = os.path.join(PROCESSED, "03_annotated.h5ad")
if os.path.exists(OUT_PATH):
    print("03_annotated.h5ad already exists — skipping Script 03.")
    import sys; sys.exit(0)

print("Loading 02_scvi.h5ad...")
adata = sc.read_h5ad(os.path.join(PROCESSED, "02_scvi.h5ad"))
n_clusters = adata.obs["leiden"].nunique()
print(f"  {adata.n_obs:,} cells, {n_clusters} clusters")

# ── Bone marrow cell type markers ─────────────────────────────────────────────
CELL_TYPE_MARKERS = {
    "hsc_progenitor":       ["CD34", "CD38", "CKIT", "FLT3", "HOXA9", "MEIS1", "MLLT3"],
    "erythroid_progenitor": ["GATA1", "KLF1", "EPB42", "TFRC", "GYPA", "ALAS2", "TAL1"],
    "mature_erythroid":     ["HBB", "HBA1", "HBA2", "HBD", "AHSP", "ANK1", "SLC4A1"],
    "myeloid_progenitor":   ["MPO", "ELANE", "CEBPA", "IRF8", "AZU1", "PRTN3", "CTSG"],
    "monocyte":             ["CD14", "LYZ", "S100A9", "FCGR3A", "VCAN", "CD16", "CSF1R"],
    "t_cell":               ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "IL7R", "CD27"],
    "nk_cell":              ["NCAM1", "GNLY", "NKG7", "KLRD1", "KLRB1", "FCGR3A"],
    "b_cell":               ["CD79A", "MS4A1", "CD19", "PAX5", "IGHM", "CD22", "BANK1"],
    "plasma_cell":          ["MZB1", "JCHAIN", "IGHG1", "IGHA1", "XBP1", "SDC1", "PRDM1"],
    "cytotoxic_t":          ["GZMB", "PRF1", "GNLY", "NKG7", "GZMA", "GZMH", "CD8A"],
    "exhausted_t":          ["LAG3", "HAVCR2", "TIGIT", "PDCD1", "ENTPD1", "TOX", "CTLA4"],
    "regulatory_t":         ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF9", "TIGIT"],
}

CONFIDENCE_THRESHOLD = 0.4  # margin between top-2 scores for high-confidence call

# ── Score each cluster ────────────────────────────────────────────────────────
annotations = {}
for cl in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
    mask  = adata.obs["leiden"] == cl
    sub   = adata[mask]
    means = np.asarray(sub.X.mean(axis=0)).flatten()
    gene_mean = dict(zip(adata.var_names, means))

    scores = {}
    for ct, markers in CELL_TYPE_MARKERS.items():
        g_in = [g for g in markers if g in gene_mean]
        scores[ct] = np.mean([gene_mean[g] for g in g_in]) if g_in else 0.0

    top_types  = sorted(scores, key=scores.get, reverse=True)
    top1, top2 = top_types[0], top_types[1]
    margin     = scores[top1] - scores[top2]
    confidence = "high" if margin >= CONFIDENCE_THRESHOLD else "low"
    if confidence == "low":
        label = f"{top1}_mixed"
    else:
        label = top1

    annotations[cl] = {
        "cell_type": label, "confidence": confidence,
        "top_score": round(scores[top1], 4), "margin": round(margin, 4),
        "n_cells": int(mask.sum())
    }

# ── Assign and report ─────────────────────────────────────────────────────────
adata.obs["cell_type"] = adata.obs["leiden"].map(
    {cl: v["cell_type"] for cl, v in annotations.items()}
)

print("\nCluster annotations:")
print("-" * 70)
for cl, info in sorted(annotations.items(), key=lambda x: -x[1]["n_cells"]):
    flag = "high" if info["confidence"] == "high" else "low "
    print(f"  Cluster {cl:>3} | {info['cell_type']:<30} | {flag} | {info['n_cells']:>6} cells")

adata.write_h5ad(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print("Script 03 complete")
