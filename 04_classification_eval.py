"""
Notebook 4 -- Classification & Cross-Condition Evaluation
==========================================================
Trains cell type classifiers on each representation and evaluates
robustness to disease-driven distribution shift (ADJ -> PDAC).

Representations:
    X_scgpt    (n_cells, 512)
    X_pca50    (n_cells,  50)
    X_raw_hvg  (n_cells, 3000)

Experiments:
    A) Within-condition  -- train/test split within mixed ADJ+PDAC
       (standard eval, establishes baseline accuracy)
    B) Cross-condition   -- train on ADJ only, test on PDAC only
       (key experiment -- answers RQ3)

Models:
    - Logistic Regression (LR)
    - MLP (PyTorch)
    - KNN

Inputs:
    PDAC_ADJ_embeddings.h5ad

Outputs:
    within_condition_results.csv
    cross_condition_results.csv
    within_condition_f1_heatmap.png
    cross_condition_f1_heatmap.png
    cross_condition_degradation.png   -- within vs cross F1 drop per rep
    confusion_matrices/               -- per rep per experiment

Answers:
    RQ3 -- do scGPT-based models generalize better from ADJ -> PDAC?
"""

# ======================================================================
# CELL 1 - Imports
# ======================================================================

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(f"Scanpy  {sc.__version__}")
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  {DEVICE}")

# ======================================================================
# CELL 2 - Config
# ======================================================================

from google.colab import drive
drive.mount("/content/drive")

DATA_DIR   = Path("/content/drive/MyDrive/CompGenProject/data")
EMBED_FILE = DATA_DIR / "PDAC_ADJ_embeddings.h5ad"

CELL_TYPE_KEY = "scGPT_target_label"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2      # for within-condition split

# MLP hyperparameters
MLP_EPOCHS    = 100
MLP_LR        = 1e-3
MLP_DROPOUT   = 0.3
MLP_PATIENCE  = 10       # early stopping patience
BATCH_SIZE    = 256

Path("confusion_matrices").mkdir(exist_ok=True)

print(f"Input : {EMBED_FILE}  (exists: {EMBED_FILE.exists()})")

# ======================================================================
# CELL 3 - Load data
# ======================================================================

print("\n-- Loading embeddings -----------------------------------------------")
adata = sc.read_h5ad(EMBED_FILE)
print(f"  Shape      : {adata.shape}")
print(f"  obsm keys  : {list(adata.obsm.keys())}")
print(f"  Conditions : {adata.obs['condition'].value_counts().to_dict()}")

# Encode cell type labels
le = LabelEncoder()
y_all = le.fit_transform(adata.obs[CELL_TYPE_KEY].values)
classes = le.classes_
n_classes = len(classes)
print(f"  Classes    : {n_classes}  {list(classes)}")

# Only keep cell types present in BOTH conditions (needed for cross-condition)
ct_pdac = set(adata[adata.obs["condition"] == "PDAC"].obs[CELL_TYPE_KEY].unique())
ct_adj  = set(adata[adata.obs["condition"] == "ADJ"].obs[CELL_TYPE_KEY].unique())
shared_cts = ct_pdac & ct_adj
print(f"\n  Cell types in both conditions : {len(shared_cts)}")
print(f"  ADJ-only  (excluded from cross-condition): {ct_adj - ct_pdac}")
print(f"  PDAC-only (excluded from cross-condition): {ct_pdac - ct_adj}")

# Masks
mask_adj  = (adata.obs["condition"] == "ADJ").values
mask_pdac = (adata.obs["condition"] == "PDAC").values
mask_shared = adata.obs[CELL_TYPE_KEY].isin(shared_cts).values

REPS = {
    "scGPT":   adata.obsm["X_scgpt"].astype(np.float32),
    "PCA-50":  adata.obsm["X_pca50"].astype(np.float32),
    "Raw-HVG": adata.obsm["X_raw_hvg"].astype(np.float32),
}

# ======================================================================
# CELL 4 - MLP definition
# ======================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, n_classes,
              epochs=100, lr=1e-3, dropout=0.3, patience=10,
              batch_size=256, device=DEVICE):
    """Train MLP with early stopping. Returns best model."""

    model = MLP(input_dim, n_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"      Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def predict_mlp(model, X, device=DEVICE):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return logits.argmax(dim=1).cpu().numpy()

# ======================================================================
# CELL 5 - Helper: train and evaluate all models on given split
# ======================================================================

def evaluate_split(X_train, y_train, X_test, y_test,
                   rep_name, experiment, classes, n_classes):
    """
    Trains LR, KNN, MLP on X_train/y_train and evaluates on X_test/y_test.
    Returns a list of result dicts.
    """
    # Scale features (fit on train only to avoid leakage)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    results = []

    # -- Logistic Regression ------------------------------------------
    print(f"    LR...", end=" ")
    lr = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="multinomial", n_jobs=-1, random_state=RANDOM_STATE,
    )
    lr.fit(X_tr_sc, y_train)
    y_pred_lr = lr.predict(X_te_sc)
    f1_lr  = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"macro-F1={f1_lr:.3f}")

    results.append({
        "representation": rep_name,
        "experiment":     experiment,
        "model":          "LR",
        "macro_f1":       f1_lr,
        "accuracy":       acc_lr,
        "y_pred":         y_pred_lr,
    })

    # -- KNN ----------------------------------------------------------
    print(f"    KNN...", end=" ")
    knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr_sc, y_train)
    y_pred_knn = knn.predict(X_te_sc)
    f1_knn  = f1_score(y_test, y_pred_knn, average="macro", zero_division=0)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"macro-F1={f1_knn:.3f}")

    results.append({
        "representation": rep_name,
        "experiment":     experiment,
        "model":          "KNN",
        "macro_f1":       f1_knn,
        "accuracy":       acc_knn,
        "y_pred":         y_pred_knn,
    })

    # -- MLP ----------------------------------------------------------
    print(f"    MLP...", end=" ")
    input_dim = X_tr_sc.shape[1]
    # Use a small val split from training data for early stopping
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr_sc, y_train, test_size=0.1,
        stratify=y_train, random_state=RANDOM_STATE,
    )
    mlp = train_mlp(
        X_tr2, y_tr2, X_val, y_val,
        input_dim=input_dim, n_classes=n_classes,
        epochs=MLP_EPOCHS, lr=MLP_LR, dropout=MLP_DROPOUT,
        patience=MLP_PATIENCE, batch_size=BATCH_SIZE,
    )
    y_pred_mlp = predict_mlp(mlp, X_te_sc)
    f1_mlp  = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"macro-F1={f1_mlp:.3f}")

    results.append({
        "representation": rep_name,
        "experiment":     experiment,
        "model":          "MLP",
        "macro_f1":       f1_mlp,
        "accuracy":       acc_mlp,
        "y_pred":         y_pred_mlp,
    })

    return results, y_test

# ======================================================================
# CELL 6 - Experiment A: Within-condition evaluation
# ======================================================================

print("\n-- Experiment A: Within-condition (train/test = 80/20 mixed) --------")

within_results = []

for rep_name, X_all in REPS.items():
    print(f"\n  [{rep_name}]")

    # Stratified split across all cells (both conditions mixed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    print(f"    Train: {X_tr.shape}  Test: {X_te.shape}")

    res, _ = evaluate_split(
        X_tr, y_tr, X_te, y_te,
        rep_name=rep_name,
        experiment="within_condition",
        classes=classes,
        n_classes=n_classes,
    )
    within_results.extend(res)

within_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "y_pred"}
    for r in within_results
])
within_df.to_csv("within_condition_results.csv", index=False)
print("\n  Saved -> within_condition_results.csv")
print(within_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
).round(3).to_string())

# ======================================================================
# CELL 7 - Experiment B: Cross-condition (train ADJ, test PDAC)
# ======================================================================

print("\n-- Experiment B: Cross-condition (train=ADJ, test=PDAC) -------------")
print("  Using only cell types present in both conditions")

cross_results = []

# Restrict to shared cell types only
adata_shared = adata[mask_shared]
y_shared     = le.transform(adata_shared.obs[CELL_TYPE_KEY].values)
mask_adj_sh  = (adata_shared.obs["condition"] == "ADJ").values
mask_pdac_sh = (adata_shared.obs["condition"] == "PDAC").values

# Re-encode labels using only shared classes for cleaner confusion matrices
le_shared = LabelEncoder()
y_shared  = le_shared.fit_transform(adata_shared.obs[CELL_TYPE_KEY].values)
classes_shared = le_shared.classes_
n_shared = len(classes_shared)
print(f"  Shared classes ({n_shared}): {list(classes_shared)}")

for rep_name, X_all in REPS.items():
    print(f"\n  [{rep_name}]")

    # Align X to shared cells
    shared_idx = np.where(mask_shared)[0]
    X_shared   = X_all[shared_idx]

    X_tr = X_shared[mask_adj_sh]
    y_tr = y_shared[mask_adj_sh]
    X_te = X_shared[mask_pdac_sh]
    y_te = y_shared[mask_pdac_sh]

    print(f"    Train (ADJ) : {X_tr.shape}")
    print(f"    Test  (PDAC): {X_te.shape}")

    res, y_te_out = evaluate_split(
        X_tr, y_tr, X_te, y_te,
        rep_name=rep_name,
        experiment="cross_condition",
        classes=classes_shared,
        n_classes=n_shared,
    )

    # Save confusion matrix for each model
    for r in res:
        cm = confusion_matrix(y_te_out, r["y_pred"])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes_shared,
            yticklabels=classes_shared,
            ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{rep_name} | {r['model']} | Cross-condition\n"
                     f"macro-F1={r['macro_f1']:.3f}")
        plt.tight_layout()
        fname = f"confusion_matrices/{rep_name}_{r['model']}_cross.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()

    cross_results.extend(res)

cross_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "y_pred"}
    for r in cross_results
])
cross_df.to_csv("cross_condition_results.csv", index=False)
print("\n  Saved -> cross_condition_results.csv")
print(cross_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
).round(3).to_string())

# ======================================================================
# CELL 8 - Plot: F1 heatmaps (within and cross condition)
# ======================================================================

print("\n-- Plotting F1 heatmaps ---------------------------------------------")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, df, title in zip(
    axes,
    [within_df, cross_df],
    ["Experiment A: Within-condition", "Experiment B: Cross-condition (ADJ->PDAC)"],
):
    pivot = df.pivot_table(
        index="model",
        columns="representation",
        values="macro_f1",
    )[["scGPT", "PCA-50", "Raw-HVG"]]

    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Macro F1"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Representation"); ax.set_ylabel("Model")

plt.suptitle("Cell type classification: macro-F1 across representations",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("f1_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> f1_heatmaps.png")

# ======================================================================
# CELL 9 - Plot: F1 degradation (within -> cross condition)
# ======================================================================

print("\n-- Plotting F1 degradation (within -> cross) ------------------------")

# Merge within and cross results
within_piv = within_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
).reset_index().melt(id_vars="model", var_name="representation",
                     value_name="within_f1")

cross_piv = cross_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
).reset_index().melt(id_vars="model", var_name="representation",
                     value_name="cross_f1")

deg_df = within_piv.merge(cross_piv, on=["model", "representation"])
deg_df["f1_drop"] = deg_df["within_f1"] - deg_df["cross_f1"]
deg_df["pct_drop"] = deg_df["f1_drop"] / deg_df["within_f1"] * 100

print("\n  F1 degradation (within -> cross condition):")
print(deg_df[["model", "representation", "within_f1",
              "cross_f1", "f1_drop", "pct_drop"]]
      .sort_values(["model", "representation"])
      .round(3)
      .to_string(index=False))

# Grouped bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = deg_df["model"].unique()
palette = {"scGPT": "#4C72B0", "PCA-50": "#DD8452", "Raw-HVG": "#55A868"}

for ax, model in zip(axes, ["LR", "KNN", "MLP"]):
    sub = deg_df[deg_df["model"] == model].sort_values("representation")
    bars = ax.bar(
        sub["representation"],
        sub["pct_drop"],
        color=[palette[r] for r in sub["representation"]],
        edgecolor="white", linewidth=0.5,
    )
    # Annotate bars with within/cross F1
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['within_f1']:.2f}->{row['cross_f1']:.2f}",
            ha="center", va="bottom", fontsize=8, color="#333333",
        )
    ax.set_title(model, fontsize=12, fontweight="bold")
    ax.set_xlabel("Representation")
    ax.set_ylabel("F1 drop (%)" if model == "LR" else "")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

fig.suptitle(
    "F1 degradation: within-condition -> cross-condition (ADJ->PDAC)\n"
    "(lower bar = more robust to distribution shift)",
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.savefig("cross_condition_degradation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> cross_condition_degradation.png")

# ======================================================================
# CELL 10 - Summary: answer RQ3
# ======================================================================

print("\n-- Summary: RQ3 robustness to distribution shift --------------------")

print("\nWithin-condition macro-F1:")
print(within_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
)[["scGPT", "PCA-50", "Raw-HVG"]].round(3).to_string())

print("\nCross-condition macro-F1 (train=ADJ, test=PDAC):")
print(cross_df.pivot_table(
    index="model", columns="representation", values="macro_f1"
)[["scGPT", "PCA-50", "Raw-HVG"]].round(3).to_string())

print("\nMean F1 drop per representation (lower = more robust):")
print(deg_df.groupby("representation")["pct_drop"]
      .mean()
      .sort_values()
      .round(2)
      .to_string())

print("\nConclusion:")
best_rep = deg_df.groupby("representation")["pct_drop"].mean().idxmin()
print(f"  Most robust representation: {best_rep}")
print(f"  (smallest average F1 drop from within -> cross condition)")
print("\nDone.")
