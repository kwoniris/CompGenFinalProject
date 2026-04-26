
---

## Script 1 — `01_Preprocessing_scRNA-seq.ipynb`

**Aim**
Convert raw sequencing data into a clean, annotated dataset ready for analysis. This is the foundation everything else builds on.

**What it does**
```
Raw 10x files (12 samples)
        ↓
Load + merge all samples into one object
        ↓
QC filtering (remove dead cells, doublets, high MT%)
        ↓
Normalize + log-transform expression values
        ↓
Cluster cells (Leiden algorithm)
        ↓
Assign cell type labels using marker genes
        ↓
PDAC_ADJ_Final_Annotated.h5ad
```

**How to interpret results**
- QC plots tell you which samples had quality issues — PDAC2 and PDAC5 had doublet signals, ADJ3 had high mitochondrial reads, all filtered appropriately
- UMAP colored by cell type should show clean separated clusters — if clusters are mixed it means annotation failed
- UMAP colored by sample should show no single sample dominating a cluster — if it does there's a batch effect
- 14 cell types successfully annotated using marker-based scoring means the biological labels are grounded in known gene signatures, not just cluster numbers

---

## Script 2 — `02_embeddings.py`

**Aim**
Convert each cell into three different mathematical representations so they can be fairly compared downstream.

**What it does**
```
PDAC_ADJ_Final_Annotated.h5ad
        ↓
scGPT transformer → X_scgpt   (512-d, learned biological space)
HVG selection + PCA → X_pca50  (50-d,  maximum variance space)
HVG selection only  → X_raw_hvg (3000-d, raw expression space)
        ↓
PDAC_ADJ_embeddings.h5ad  (all three in one file)
        ↓
UMAP comparison plot
```

**How to interpret results**

The UMAP comparison plot has two rows:

Row 1 (cell type) — tighter, more separated clusters = better representation. scGPT should show the cleanest separation because it was trained specifically to encode cell identity

Row 2 (condition) — you want ADJ and PDAC cells to mix within clusters, not separate into two completely different clouds. If they separate completely it means the representation is capturing batch/condition effects more than cell type biology. The fact that they mix but show subtle spatial separation within clusters is the ideal result — it means cell type dominates but tumor effects are still detectable

---

## Script 3 — `03_shift_analysis.py`

**Aim**
Quantify how much each cell type changes between normal and tumor tissue in each representation. This is the core scientific contribution — directly answering RQ1 and RQ2.

**What it does**
```
For each cell type × representation:

  ADJ cells of that type  → mean vector μ_adj
  PDAC cells of that type → mean vector μ_pdac

  shift = distance(μ_adj, μ_pdac)
        ↓
Ranking, heatmap, scatter, UMAP arrows
```

**How to interpret each output**

`shift_ranking_barplot.png` — ranked bar chart of cell type shifts, one panel per representation. Read this to answer RQ1. The ordering should be consistent across panels if all three representations are capturing the same biology. Ductal at the top is the expected and correct biological result.

`shift_heatmap.png` — left panel shows raw magnitudes (ignore scale differences between representations), right panel shows normalised 0-1 values which is the correct panel for comparison. Dark red = most shifted relative to other cell types in that representation. Consistent dark red across all three columns for the same cell type = robust finding.

`shift_scatter_scgpt_vs_pca.png` — each point is a cell type. Points above the diagonal mean scGPT detected a larger shift than PCA for that cell type. r=0.868 means the two representations largely agree on which cell types shift most — this validates both approaches are capturing real biology rather than noise.

`umap_shift_vectors.png` — the most intuitive plot. Arrow length = shift magnitude, arrow direction = where the cell type moves in transcriptional space under tumor conditions. Long arrows crossing into other cluster regions (like Neutrophils) indicate dramatic phenotypic switching. Short arrows indicate stable cell types.

**Condition-exclusive cell types**
```
Lost in PDAC  : Acinar, Endocrine, NK
Gained in PDAC: Plasma, Schwann
```
These can't have a shift measured because they only exist in one condition — but their complete absence is itself the most extreme form of shift. Acinar cell loss is particularly meaningful as it reflects acinar-to-ductal metaplasia, a key step in PDAC development.

---

## Script 4 — `04_classification_eval.py`

**Aim**
Test whether scGPT embeddings produce more robust classifiers when generalizing from normal tissue to tumor tissue. This directly answers RQ3.

**What it does**
```
Experiment A — Within condition
  Train: 80% of all cells (ADJ + PDAC mixed)
  Test:  20% of all cells (ADJ + PDAC mixed)
  → establishes baseline accuracy

Experiment B — Cross condition  ← KEY EXPERIMENT
  Train: all ADJ cells
  Test:  all PDAC cells
  → tests generalization under distribution shift
```

**How to interpret each output**

`f1_heatmaps.png` — two heatmaps side by side. Left = Experiment A (within), Right = Experiment B (cross). Green = high F1, Red = low F1. You expect all representations to do reasonably well in Experiment A. The interesting story is in Experiment B — which representation stays green?

`cross_condition_degradation.png` — this is the most important plot for your paper. Shows how much F1 drops going from Experiment A to B for each representation and model. 
```
Small bar = robust to distribution shift (good)
Large bar = performance collapses under shift (bad)
```
If scGPT bars are consistently shorter than PCA-50 and Raw-HVG bars across LR, KNN and MLP — that's your answer to RQ3. scGPT generalizes better because its learned representation captures stable biological cell identity rather than condition-specific variance.

`confusion_matrices/` — one per model per representation for the cross-condition experiment. Look at the diagonal — strong diagonal = correct predictions. Off-diagonal errors tell you which cell types get confused when the model is transferred from normal to tumor tissue. If scGPT confusion matrices have stronger diagonals than PCA ones, it's further evidence of better generalization.

---

## The overall narrative across all four scripts

```
Script 1  "Here is the biology"
           → 14 cell types, 12 samples, clean annotations

Script 2  "Here are three ways to encode it mathematically"
           → scGPT, PCA-50, Raw-HVG all extracted and saved

Script 3  "Here is how cancer reshapes each cell type"
           → shift rankings answer RQ1 and RQ2
           → Ductal, Neutrophils, Fibroblasts most affected
           → 5 cell types lost or gained entirely

Script 4  "Here is whether foundation models handle that better"
           → cross-condition F1 degradation answers RQ3
           → if scGPT degrades less, the hypothesis is supported
```

The project as a whole is asking a fundamental question about AI in biology — not just "can a model classify cell types" but "does a foundation model trained on millions of cells understand disease-driven biological change better than classical statistics?" That's a meaningful scientific contribution regardless of which way the results go.