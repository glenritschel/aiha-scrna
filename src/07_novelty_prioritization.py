"""
07_novelty_prioritization.py  —  AIHA scRNA-seq pipeline
PubMed novelty assessment with AIHA-specific search terms.
Three tiers: NOVEL_ALL, NOVEL_AIHA, KNOWN.
Priority score = max_reversal_score × n_queries × novelty_multiplier.
"""
import os, time, json
import urllib.request, urllib.parse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/aiha_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

# ── Skip guard ─────────────────────────────────────────────────────────────────
if (os.path.exists(os.path.join(PROCESSED, "priority_candidates.csv")) and
        os.path.exists(os.path.join(PROCESSED, "patent_watch.csv"))):
    print("Outputs already exist — skipping Script 07.")
    import sys; sys.exit(0)

SLEEP = 0.4

# ── MOA reference ─────────────────────────────────────────────────────────────
MOA_REFERENCE = {
    "QL-XII-47":     "MELK/FLT3 inhibitor",
    "WZ-3105":       "SRC/ABL inhibitor",
    "JNK-9L":        "JNK inhibitor",
    "AS-601245":     "JNK inhibitor",
    "BMS-387032":    "CDK inhibitor",
    "AZD-7762":      "CHK1/2 inhibitor",
    "PF-431396":     "FAK/PYK2 inhibitor",
    "WZ-4-145":      "CDK8 inhibitor",
    "CGP-60474":     "CDK1/2 inhibitor",
    "alvocidib":     "CDK1/2/4/6/9 inhibitor",
    "dinaciclib":    "CDK inhibitor",
    "nilotinib":     "BCR-ABL inhibitor",
    "dasatinib":     "BCR-ABL/SRC inhibitor",
    "ruxolitinib":   "JAK1/2 inhibitor",
    "BI-2536":       "PLK1 inhibitor",
    "radicicol":     "HSP90 inhibitor",
    "withaferin-a":  "HSP90/NF-κB inhibitor",
    "celastrol":     "HSP90/NF-κB inhibitor",
    "LDN-193189":    "BMP/ALK2 inhibitor",
    "SB590885":      "BRAF inhibitor",
    "trametinib":    "MEK1/2 inhibitor",
    "PD-0325901":    "MEK1/2 inhibitor",
    "sirolimus":     "mTOR inhibitor",
    "selumetinib":   "MEK1/2 inhibitor",
    "fostamatinib":  "SYK inhibitor",
    "canertinib":    "Pan-EGFR inhibitor",
    "pelitinib":     "Pan-EGFR inhibitor",
}

# ── AIHA-specific PubMed search terms ─────────────────────────────────────────
# Set A: AIHA-specific (no prior work = NOVEL_ALL)
SEARCH_AIHA = (
    '("{compound}"[Title/Abstract]) AND '
    '("autoimmune hemolytic anemia"[Title/Abstract] OR '
    '"AIHA"[Title/Abstract] OR '
    '"warm autoimmune hemolytic"[Title/Abstract])'
)
# Set B: Hematologic autoimmune (broader)
SEARCH_HEME_AUTO = (
    '("{compound}"[Title/Abstract]) AND '
    '("hemolytic anemia"[Title/Abstract] OR '
    '"autoimmune hemolysis"[Title/Abstract] OR '
    '"immune hemolysis"[Title/Abstract])'
)
# Set C: Complement/Fc receptor targeting (mechanistic relevance)
SEARCH_COMPLEMENT = (
    '("{compound}"[Title/Abstract]) AND '
    '("complement"[Title/Abstract] OR '
    '"Fc receptor"[Title/Abstract] OR '
    '"FcgR"[Title/Abstract] OR '
    '"erythrophagocytosis"[Title/Abstract])'
)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

def pubmed_count(query):
    params = urllib.parse.urlencode({"db": "pubmed", "term": query,
                                     "retmode": "json", "retmax": "0"})
    try:
        with urllib.request.urlopen(f"{PUBMED_BASE}?{params}", timeout=10) as r:
            data = json.loads(r.read().decode())
            return int(data["esearchresult"]["count"])
    except Exception:
        return -1

# ── Load LINCS candidates ─────────────────────────────────────────────────────
lincs_path = os.path.join(PROCESSED, "06_lincs.csv")
if not os.path.exists(lincs_path):
    print("06_lincs.csv not found — cannot run Script 07.")
    import sys; sys.exit(1)

candidates = pd.read_csv(lincs_path)
print(f"Loading LINCS candidates... {len(candidates)} candidates")

# ── Novelty assessment ────────────────────────────────────────────────────────
results = []
for i, row in candidates.iterrows():
    compound = row["compound"]
    print(f"[{i+1}/{len(candidates)}] {compound}...", end=" ")

    cnt_a = pubmed_count(SEARCH_AIHA.format(compound=compound))
    time.sleep(SLEEP)
    cnt_b = pubmed_count(SEARCH_HEME_AUTO.format(compound=compound))
    time.sleep(SLEEP)
    cnt_c = pubmed_count(SEARCH_COMPLEMENT.format(compound=compound))
    time.sleep(SLEEP)

    if cnt_a == 0 and cnt_b == 0 and cnt_c == 0:
        tier = "NOVEL_ALL"
    elif cnt_a == 0:
        tier = "NOVEL_AIHA"
    else:
        tier = "KNOWN"

    print(f"{tier} (AIHA:{cnt_a}, HemeAuto:{cnt_b}, Complement:{cnt_c})")

    results.append({
        "compound":           compound,
        "moa":                MOA_REFERENCE.get(compound, "unknown"),
        "novelty_tier":       tier,
        "max_reversal_score": row["max_reversal_score"],
        "n_queries":          row["n_queries"],
        "aiha_pubs":          cnt_a,
        "heme_auto_pubs":     cnt_b,
        "complement_pubs":    cnt_c,
    })

df = pd.DataFrame(results)

# ── Priority scoring ──────────────────────────────────────────────────────────
MULT = {"NOVEL_ALL": 3.0, "NOVEL_AIHA": 1.5, "KNOWN": 1.0}
df["novelty_multiplier"] = df["novelty_tier"].map(MULT)
df["priority_score"] = (df["max_reversal_score"] *
                        df["n_queries"] *
                        df["novelty_multiplier"]).round(1)
df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

# ── Novelty summary ───────────────────────────────────────────────────────────
print(f"\nNovelty breakdown:")
print(df["novelty_tier"].value_counts().to_string())

print("\nTop 20 priority candidates:")
print(df.head(20)[["compound","moa","novelty_tier",
                    "max_reversal_score","n_queries","priority_score"]].to_string(index=False))

# ── Patent watch ──────────────────────────────────────────────────────────────
patent_watch = df[df["novelty_tier"] == "NOVEL_ALL"].copy()
print(f"\nPatent watch: {len(patent_watch)} NOVEL_ALL compounds")
print(patent_watch[["compound","moa","novelty_tier",
                     "max_reversal_score","n_queries","priority_score"]].to_string(index=False))

df.to_csv(os.path.join(PROCESSED, "priority_candidates.csv"), index=False)
patent_watch.to_csv(os.path.join(PROCESSED, "patent_watch.csv"), index=False)
print(f"\nSaved: priority_candidates.csv ({len(df)} compounds)")
print(f"Saved: patent_watch.csv ({len(patent_watch)} compounds)")
print("\nScript 07 complete")
print("=" * 60)
print("PIPELINE COMPLETE — review priority_candidates.csv")
print("=" * 60)
