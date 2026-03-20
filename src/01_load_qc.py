"""
01_load_qc.py  —  AIHA scRNA-seq pipeline
Dataset   : GSE301528 (Bone marrow, AIHA: Diagnosis / Remission / Relapse_Refractory)
Tissue    : Bone marrow aspirate
Condition : Diagnosis (n=5), Remission (n=3), Relapse_Refractory (n=4)
Format    : CellRanger v3 flat (features.tsv.gz, 3-column)
Note      : Sorted subpopulation samples (GSM9084848-853) are EXCLUDED —
            only whole-BM samples are loaded.
"""
import os, gc, re
import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import scanpy as sc

sc.settings.verbosity = 1

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE   = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
RAW_DIR      = os.path.join(DRIVE_BASE, "raw")
PROCESSED    = os.path.join(DRIVE_BASE, "processed")
os.makedirs(PROCESSED, exist_ok=True)

# ── Skip guard ────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(PROCESSED, "01_loaded.h5ad")
if os.path.exists(OUT_PATH):
    print(f"01_loaded.h5ad already exists — skipping Script 01.")
    import sys; sys.exit(0)

# ── QC thresholds ─────────────────────────────────────────────────────────────
MIN_GENES_PREFILTER = 500   # empty-droplet pre-filter (CSR nnz per row)
MIN_GENES           = 200
MAX_GENES           = 6000
MAX_MT_PCT          = 20

# ── Condition map (whole-BM samples only; sorted subpops excluded) ─────────────
# Sorted subpops excluded: GSM9084848-GSM9084853 (bf.hdmi/hd44/hd45 ±pos)
CONDITION_MAP = {
    # Diagnosis — active AIHA at onset
    "GSM9084840": ("Diagnosis",          "P1"),
    "GSM9084841": ("Diagnosis",          "P2"),
    "GSM9084842": ("Diagnosis",          "P3"),
    "GSM9084843": ("Diagnosis",          "P4"),
    "GSM9084844": ("Diagnosis",          "P5"),
    # Remission — disease resolved (internal control)
    "GSM9084845": ("Remission",          "P2r"),
    "GSM9084846": ("Remission",          "P1r"),
    "GSM9084847": ("Remission",          "P4r"),
    # Relapse/Refractory — treatment-resistant active disease
    "GSM9084836": ("Relapse_Refractory", "RR1"),
    "GSM9084837": ("Relapse_Refractory", "RR2"),
    "GSM9084838": ("Relapse_Refractory", "RR3"),
    "GSM9084839": ("Relapse_Refractory", "RR4"),
}

# ── Sample discovery ──────────────────────────────────────────────────────────
def discover_samples(raw_dir):
    """Discover CellRanger v3 flat triplets. Returns {gsm: {barcodes, features, matrix}}."""
    samples = {}
    for fname in os.listdir(raw_dir):
        m = re.match(r'^(GSM\d+)_.*_(barcodes|features|matrix)\.(tsv|mtx)\.gz$', fname)
        if not m:
            continue
        gsm, ftype = m.group(1), m.group(2)
        if gsm not in CONDITION_MAP:
            continue   # skip sorted subpopulation samples
        if gsm not in samples:
            samples[gsm] = {}
        samples[gsm][ftype] = os.path.join(raw_dir, fname)
    return samples

# ── Per-sample loader (CellRanger v3, 3-column features) ─────────────────────
def load_sample_v3(gsm, paths, condition, patient):
    mat      = scipy.io.mmread(paths["matrix"]).T.tocsr()  # cells × genes
    barcodes = pd.read_csv(paths["barcodes"], header=None, sep="\t")[0].tolist()
    features = pd.read_csv(paths["features"], header=None, sep="\t")

    # Empty-droplet pre-filter
    keep     = np.diff(mat.indptr) >= MIN_GENES_PREFILTER
    mat      = mat[keep, :]
    barcodes = [bc for bc, k in zip(barcodes, keep) if k]

    adata = ad.AnnData(X=mat)
    adata.obs_names            = [f"{gsm}_{bc}" for bc in barcodes]
    adata.var_names            = features[1].tolist()
    adata.var["gene_ids"]      = features[0].tolist()
    adata.var["feature_types"] = features[2].tolist() if features.shape[1] > 2 else "Gene Expression"
    adata.obs["condition"]     = condition
    adata.obs["patient"]       = patient
    adata.obs["sample"]        = gsm
    adata.var_names_make_unique()
    return adata

# ── Main load loop ─────────────────────────────────────────────────────────────
print("Discovering samples...")
all_samples = discover_samples(RAW_DIR)
print(f"Found {len(all_samples)} whole-BM GSM IDs")

# Load per condition with Drive checkpoints to manage RAM
CONDITION_ORDER = ["Diagnosis", "Remission", "Relapse_Refractory"]
cond_ckpt_paths = []

for cond in CONDITION_ORDER:
    ckpt = os.path.join(PROCESSED, f"_ckpt_{cond}.h5ad")
    if os.path.exists(ckpt):
        print(f"{cond}: checkpoint found, loading...")
        cond_ckpt_paths.append(ckpt)
        continue

    print(f"\nLoading {cond}...")
    adatas = []
    for gsm in sorted(all_samples):
        c, patient = CONDITION_MAP[gsm]
        if c != cond:
            continue
        paths = all_samples[gsm]
        if not all(k in paths for k in ["barcodes", "features", "matrix"]):
            print(f"  {gsm} — missing files, skipping")
            continue
        try:
            a = load_sample_v3(gsm, paths, cond, patient)
            print(f"  {gsm} ({patient}) — {a.n_obs:,} cells")
            adatas.append(a)
        except Exception as e:
            print(f"  {gsm} — ERROR: {e}")

    if not adatas:
        print(f"  WARNING: no {cond} samples loaded")
        continue

    batch = sc.concat(adatas, join="outer", fill_value=0)
    del adatas; gc.collect()
    print(f"  {cond}: {batch.n_obs:,} cells — saving checkpoint...")
    batch.write_h5ad(ckpt)
    del batch; gc.collect()
    cond_ckpt_paths.append(ckpt)
    print(f"  Checkpoint saved.")

# ── QC filter each condition checkpoint separately ────────────────────────────
print("\nQC filtering per condition...")
filt_ckpt_paths = []

for ckpt in cond_ckpt_paths:
    cond = os.path.basename(ckpt).replace("_ckpt_", "").replace(".h5ad", "")
    filt = ckpt.replace(".h5ad", "_filt.h5ad")
    if os.path.exists(filt):
        print(f"{cond}: filtered checkpoint found.")
        filt_ckpt_paths.append(filt)
        continue

    a = sc.read_h5ad(ckpt)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
    n_before = a.n_obs
    a = a[a.obs["n_genes_by_counts"] >= MIN_GENES].copy()
    a = a[a.obs["n_genes_by_counts"] <= MAX_GENES].copy()
    a = a[a.obs["pct_counts_mt"]     <= MAX_MT_PCT].copy()
    print(f"  {cond}: {n_before:,} → {a.n_obs:,} cells ({n_before - a.n_obs:,} removed)")
    a.write_h5ad(filt)
    del a; gc.collect()
    filt_ckpt_paths.append(filt)

# ── Merge filtered checkpoints ────────────────────────────────────────────────
print("\nMerging filtered conditions...")
adatas_all = [sc.read_h5ad(p) for p in filt_ckpt_paths]
adata = sc.concat(adatas_all, join="outer", fill_value=0)
del adatas_all; gc.collect()

print(f"Combined: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(adata.obs["condition"].value_counts().to_string())
print(f"  Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"  Median MT%:        {adata.obs['pct_counts_mt'].median():.1f}%")

# ── Normalise, HVG ────────────────────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="sample", subset=True)
print(f"HVGs: {adata.n_vars}")

adata.write_h5ad(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print("Script 01 complete")
