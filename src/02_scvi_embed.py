"""
02_scvi_embed.py  —  AIHA scRNA-seq pipeline
scVI dimensionality reduction, UMAP, multi-resolution Leiden clustering.
"""
import os, gc, random
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

sc.settings.verbosity = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

OUT_PATH  = os.path.join(PROCESSED, "02_scvi.h5ad")
CKPT_PATH = os.path.join(PROCESSED, "02_scvi_ckpt.h5ad")

if os.path.exists(OUT_PATH):
    print("02_scvi.h5ad already exists — skipping Script 02.")
    import sys; sys.exit(0)

# ── Params ────────────────────────────────────────────────────────────────────
RANDOM_SEED        = 0
N_LATENT           = 30
N_LAYERS           = 2
N_HIDDEN           = 128
N_NEIGHBORS        = 15
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

# ── Cell type markers for resolution scoring ──────────────────────────────────
CELL_TYPE_MARKERS = {
    "hsc_progenitor":       ["CD34", "CD38", "CKIT", "FLT3", "HOXA9", "MEIS1"],
    "erythroid_progenitor": ["GATA1", "KLF1", "EPB42", "TFRC", "GYPA", "ALAS2"],
    "mature_erythroid":     ["HBB", "HBA1", "HBA2", "HBD", "AHSP", "ANK1"],
    "myeloid_progenitor":   ["MPO", "ELANE", "CEBPA", "IRF8", "AZU1", "PRTN3"],
    "monocyte":             ["CD14", "LYZ", "S100A9", "FCGR3A", "VCAN", "CD16"],
    "t_cell":               ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "IL7R"],
    "nk_cell":              ["NCAM1", "GNLY", "NKG7", "KLRD1", "KLRB1", "FCGR3A"],
    "b_cell":               ["CD79A", "MS4A1", "CD19", "PAX5", "IGHM", "CD22"],
    "plasma_cell":          ["MZB1", "JCHAIN", "IGHG1", "IGHA1", "XBP1", "SDC1"],
    "cytotoxic_t":          ["GZMB", "PRF1", "GNLY", "NKG7", "GZMA", "GZMH"],
    "exhausted_t":          ["LAG3", "HAVCR2", "TIGIT", "PDCD1", "ENTPD1", "TOX"],
    "regulatory_t":         ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF9"],
}

# ── Load ──────────────────────────────────────────────────────────────────────
if os.path.exists(CKPT_PATH):
    print(f"Loading scVI checkpoint: {CKPT_PATH}")
    adata = sc.read_h5ad(CKPT_PATH)
    print(f"  {adata.n_obs:,} cells, X_scVI present: {'X_scVI' in adata.obsm}")
    skip_training = "X_scVI" in adata.obsm
else:
    print("Loading 01_loaded.h5ad...")
    adata = sc.read_h5ad(os.path.join(PROCESSED, "01_loaded.h5ad"))
    print(f"  {adata.n_obs:,} cells × {adata.n_vars} HVGs")
    skip_training = False

# ── scVI training ─────────────────────────────────────────────────────────────
if not skip_training:
    scvi.model.SCVI.setup_anndata(adata, batch_key="sample")
    model = scvi.model.SCVI(
        adata, n_latent=N_LATENT, n_layers=N_LAYERS, n_hidden=N_HIDDEN
    )
    print("Training scVI (200 epochs, early stopping)...")
    model.train(max_epochs=200, early_stopping=True, early_stopping_patience=20)
    try:
        key = "train_loss_epoch" if "train_loss_epoch" in model.history else "train_loss"
        final_loss = float(np.array(model.history[key].values[-1]).flat[0])
        print(f"Training complete. Final loss: {final_loss:.2f}")
    except Exception:
        print("Training complete. (Loss history key unavailable)")

    adata.obsm["X_scVI"] = model.get_latent_representation()
    print(f"Latent shape: {adata.obsm['X_scVI'].shape}")

    # Checkpoint immediately after training
    adata.write_h5ad(CKPT_PATH)
    print(f"Checkpoint saved: {CKPT_PATH}")

# ── UMAP + Leiden ─────────────────────────────────────────────────────────────
print("Building neighbour graph and UMAP...")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGHBORS)
sc.tl.umap(adata)
print("UMAP complete")

print("Leiden clustering...")
for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key,
                 random_state=RANDOM_SEED, flavor="igraph",
                 n_iterations=2, directed=False)
    n = adata.obs[key].nunique()
    print(f"  Resolution {res}: {n} clusters")

# ── Resolution selection ──────────────────────────────────────────────────────
print("\nResolution comparison:")
res_results = []
for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_{res}"
    n_clusters = adata.obs[key].nunique()
    marker_sets = list(CELL_TYPE_MARKERS.values())
    genes_present = [g for markers in marker_sets for g in markers if g in adata.var_names]
    cluster_top = {}
    for cl in adata.obs[key].unique():
        mask = adata.obs[key] == cl
        means = adata[mask].X.mean(axis=0)
        if hasattr(means, "A1"):
            means = means.A1
        else:
            means = np.asarray(means).flatten()
        gene_mean = dict(zip(adata.var_names, means))
        scores = {}
        for ct, markers in CELL_TYPE_MARKERS.items():
            g_in = [g for g in markers if g in gene_mean]
            scores[ct] = np.mean([gene_mean[g] for g in g_in]) if g_in else 0
        cluster_top[cl] = max(scores, key=scores.get)
    n_types   = len(set(cluster_top.values()))
    max_types = len(CELL_TYPE_MARKERS)
    best_cls  = sum(1 for cl in cluster_top if
                    sorted({ct: sum(gene_mean.get(g,0) for g in ms if g in gene_mean)
                             for ct, ms in CELL_TYPE_MARKERS.items()}.values(),
                           reverse=True)[0] > 0)
    sep_score = n_types / max_types
    res_results.append({"resolution": res, "n_clusters": n_clusters,
                        "n_celltypes_resolved": n_types,
                        "separation_score": round(sep_score, 6)})

import pandas as pd
df = pd.DataFrame(res_results).sort_values("separation_score", ascending=False)
print(df.to_string(index=False))
best_res = float(df.iloc[0]["resolution"])
print(f"\nRecommended: {best_res} ({int(df.iloc[0]['n_clusters'])} clusters)")

adata.obs["leiden"] = adata.obs[f"leiden_{best_res}"].copy()
adata.uns["leiden_resolution"] = best_res

# Cluster composition table
adata.obs.groupby(["leiden", "condition"]).size().unstack(fill_value=0).to_csv(
    os.path.join(PROCESSED, "cluster_condition_counts.csv"))

adata.write_h5ad(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print(f"Clusters: {adata.obs['leiden'].nunique()}")
print("Script 02 complete")
