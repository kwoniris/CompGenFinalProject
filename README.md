# CompGenFinalProject

Computational Genomics Final Project — comparing scGPT foundation-model embeddings against PCA and raw HVG representations for cell-type classification and condition-shift analysis on PDAC vs. matched adjacent (ADJ) tissue.

The project answers three questions:
- **RQ1 / RQ2**: Which cell types shift most between normal and tumor tissue, and do different embedding spaces agree on the ranking?
- **RQ3**: Do scGPT embeddings produce classifiers that generalize better from normal to tumor tissue than classical representations?

---

## File Structure

```
.
├── README.md
├── requirements.txt             pip dependencies
├── .gitignore
│
├── notebooks/                   Main analysis pipeline (run in order)
│   ├── 01_Preprocessing_scRNA-seq.ipynb     QC, normalize, cluster, annotate
│   ├── 02_Embeddings.ipynb                  Compute scGPT / PCA-50 / Raw-HVG
│   ├── 03_ShiftAnalysis.ipynb               Cell-type shift quantification (RQ1/RQ2)
│   ├── 04_classification_eval.ipynb         Within- vs cross-condition F1 (RQ3)
│   ├── scGPT_PDAC_classifier_v2.ipynb       Zero-shot scGPT classifier baseline
│   ├── finetuning_colab.ipynb               scGPT fine-tuning (run on Colab GPU)
│   └── Finetuned_scgpt_LR_MLP.ipynb         Downstream LR/MLP on fine-tuned embeddings
│
├── config/
│   └── environment.py           Shared paths / constants used by notebooks
│
├── results/                     Generated figures and tables
│   ├── 02/                      UMAP comparison across embeddings
│   ├── 03/                      Shift rankings, heatmaps, scatter, UMAP arrows
│   └── validation_preprocessing/   QC plots for the validation cohort
│
└── misc/                        Older / exploratory notebooks not part of the
                                 main pipeline (kept for reference)
```

---

## Data

Primary cohort (used by `notebooks/01_Preprocessing_scRNA-seq.ipynb`):
- **GSE212966** — PDAC and matched adjacent normal pancreas, 10x scRNA-seq.
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212966

Validation cohort (used by notebooks under `misc/`):
- **GSE155698** — additional PDAC scRNA-seq cohort.
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155698

Pretrained foundation model:
- **scGPT-human** — the whole-human pretrained checkpoint from the official scGPT release.
  https://github.com/bowang-lab/scGPT

After downloading the GEO archive, expand the per-sample 10x triplets (`matrix.mtx`, `barcodes.tsv`, `features.tsv`) under a local `data/` directory; paths are configured in `config/environment.py`. Large `.h5ad` outputs are intentionally excluded from git via `.gitignore`. The intermediate file `PDAC_ADJ_cleaned_full_raw.h5ad` (output of notebook 01) is the entry point for notebooks 02–04 if you want to skip preprocessing.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m ipykernel install --user --name pdac-scgpt --display-name "PDAC scGPT"
```

scGPT's CUDA wheels are sensitive to the torch / flash-attn versions; if `pip install scgpt` fails, follow the install instructions in the scGPT repo above.

---

## How to Run the Pipeline

Run the notebooks under `notebooks/` in numeric order. Each notebook saves its outputs so the next one can pick up without re-running upstream work.

### 1. `01_Preprocessing_scRNA-seq.ipynb`

Loads raw 10x files for the 12 GSE212966 samples, applies QC filters (mitochondrial %, doublet score, gene/count thresholds), normalizes and log-transforms, clusters with Leiden, and assigns 14 cell-type labels via marker-gene scoring. Writes `PDAC_ADJ_cleaned_full_raw.h5ad`.

How to read the outputs:
- QC plots flag samples with quality issues (e.g. PDAC2/PDAC5 doublets, ADJ3 high-MT).
- UMAP coloured by cell type should show clean separated clusters; UMAP coloured by sample should show no single sample dominating a cluster (no batch effect).

### 2. `02_Embeddings.ipynb`

Produces three representations of every cell so they can be compared on equal footing:
- `X_scgpt` — 512-d scGPT embedding (learned biological space)
- `X_pca50` — 50-d PCA on HVGs (max-variance space)
- `X_raw_hvg` — 3000-d raw HVG expression

How to read the outputs:
- Row 1 (cell type) of the UMAP comparison: tighter clusters = better representation. scGPT should be cleanest.
- Row 2 (condition): ADJ and PDAC should mix within clusters. Complete separation would mean the representation is dominated by condition/batch effects rather than cell-type biology.

### 3. `03_ShiftAnalysis.ipynb`

For each (cell type × representation), computes the centroid of ADJ cells and PDAC cells and measures their distance — this quantifies how much each cell type is reshaped under tumor conditions. Outputs:
- `shift_ranking_barplot.png` — answers RQ1. Consistent ordering across the three panels means all representations agree on which cell types shift most. Ductal at the top is the expected biological result.
- `shift_heatmap.png` — right panel (normalised) is the correct one for cross-representation comparison. Dark red across all three columns for the same cell type = a robust finding.
- `shift_scatter_scgpt_vs_pca.png` — points above the diagonal mean scGPT detected a larger shift than PCA. High Pearson r validates that the two representations capture the same biology.
- `umap_shift_vectors.png` — arrow length = shift magnitude, direction = where a cell type moves in transcriptional space under tumor conditions.

Condition-exclusive cell types (lost in PDAC: Acinar, Endocrine, NK; gained in PDAC: Plasma, Schwann) cannot have a shift measured, but their absence is itself the most extreme form of shift — the loss of Acinar reflects acinar-to-ductal metaplasia, a known step in PDAC development.

### 4. `04_classification_eval.ipynb`

Trains LR / KNN / MLP classifiers on each representation under two splits — this is the experiment that answers RQ3.

- **Experiment A (within-condition):** 80/20 split over all cells, ADJ + PDAC mixed. Establishes baseline accuracy.
- **Experiment B (cross-condition):** train on ADJ, test on PDAC. Tests generalization under distribution shift.

How to read the outputs:
- `f1_heatmaps.png` — left = Experiment A, right = Experiment B. The interesting story is which representation stays green in B.
- `cross_condition_degradation.png` — F1 drop from A to B per representation × model. Short bars = robust to shift, tall bars = collapse under shift. If scGPT bars are consistently shorter than PCA-50 and Raw-HVG across all three classifiers, RQ3 is supported.
- `confusion_matrices/` — strong diagonals on the cross-condition matrices = better generalization.

### Optional add-ons

- `scGPT_PDAC_classifier_v2.ipynb` — zero-shot scGPT cell-type classifier as a baseline reference.
- `finetuning_colab.ipynb` — fine-tunes scGPT on this cohort. Designed to run on Google Colab with a GPU; download the resulting checkpoint locally before running the next notebook.
- `Finetuned_scgpt_LR_MLP.ipynb` — re-runs the LR / MLP heads on the fine-tuned embeddings for comparison against the frozen-scGPT results from notebook 04.

---

## Overall Narrative

```
Notebook 1   "Here is the biology"
             14 cell types, 12 samples, clean annotations.

Notebook 2   "Here are three ways to encode it mathematically"
             scGPT, PCA-50, Raw-HVG all extracted and saved.

Notebook 3   "Here is how cancer reshapes each cell type"
             Shift rankings answer RQ1 and RQ2.
             Ductal, Neutrophils, Fibroblasts most affected.
             5 cell types lost or gained entirely.

Notebook 4   "Here is whether foundation models handle that better"
             Cross-condition F1 degradation answers RQ3.
             If scGPT degrades less, the hypothesis is supported.
```
