"""
╔══════════════════════════════════════════════════════════════════╗
║       ANCORA — Modul 2: Training & Validasi Model XGBoost       ║
║       Google Colab / Python 3.10+                               ║
║                                                                  ║
║  Checklist:                                                      ║
║  [✓] Training Model (XGBClassifier baseline)                    ║
║  [✓] Tuning Hyperparameter (RandomizedSearchCV + manual sweep)  ║
║  [✓] Evaluasi Matriks (Precision, Recall, F1, ROC-AUC)          ║
║  [✓] Export Artifacts (ancora_xgb_model.pkl)                    ║
╚══════════════════════════════════════════════════════════════════╝

Dependensi Modul 1 (harus ada di direktori yang sama):
    - train_test_splits.pkl
    - feature_cols.pkl
    - scaler.pkl        (sudah ada, tidak diubah di sini)

Cara pakai di Google Colab:
    1. Upload file ini + semua .pkl dari Modul 1
    2. !pip install xgboost imbalanced-learn -q
    3. !python ancora_modul2_xgboost_training.py
    4. Download: ancora_xgb_model.pkl
"""

# ── IMPORTS ────────────────────────────────────────────────────────
import pickle
import os
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

SEED       = 42
OUTPUT_DIR = "."
np.random.seed(SEED)

print("=" * 62)
print("  ANCORA — Modul 2: XGBoost Training & Validasi")
print("=" * 62)


# ══════════════════════════════════════════════════════════════════
# BAGIAN 0: LOAD ARTIFACTS MODUL 1
# ══════════════════════════════════════════════════════════════════

def load_pkl(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{filename}' tidak ditemukan.\n"
            f"Pastikan sudah menjalankan Modul 1 dan meng-upload .pkl ke direktori ini."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

print("\n[0/6] Loading artifacts dari Modul 1...")
splits       = load_pkl("train_test_splits.pkl")
FEATURE_COLS = load_pkl("feature_cols.pkl")

X_train = splits["X_train"]   # numpy array, sudah di-scale & SMOTE → shape (680, 12)
y_train = splits["y_train"]   # numpy array, balanced 340:340
X_test  = splits["X_test"]    # numpy array, original distribution → shape (100, 12)
y_test  = splits["y_test"]    # numpy array, stratified 85:15

print(f"    X_train : {X_train.shape}  |  class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"    X_test  : {X_test.shape}   |  class dist: {dict(zip(*np.unique(y_test,  return_counts=True)))}")
print(f"    Features: {FEATURE_COLS}")

# Catatan: SMOTE sudah diterapkan di Modul 1 → train set sudah balanced (340:340)
# scale_pos_weight = 1.0 karena train set sudah balanced via SMOTE
SCALE_POS_WEIGHT = 1.0


# ══════════════════════════════════════════════════════════════════
# BAGIAN 1: TRAINING MODEL BASELINE
# ══════════════════════════════════════════════════════════════════

print("\n[1/6] Training model baseline XGBClassifier...")

baseline_params = dict(
    n_estimators      = 200,
    max_depth         = 4,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = SCALE_POS_WEIGHT,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = SEED,
    n_jobs            = -1,
)

t0            = time.time()
model_base    = XGBClassifier(**baseline_params)
model_base.fit(X_train, y_train)
base_train_ms = (time.time() - t0) * 1000

y_pred_base   = model_base.predict(X_test)
y_prob_base   = model_base.predict_proba(X_test)[:, 1]

base_metrics = {
    "precision": precision_score(y_test, y_pred_base, pos_label=1),
    "recall":    recall_score(y_test, y_pred_base, pos_label=1),
    "f1":        f1_score(y_test, y_pred_base, pos_label=1),
    "roc_auc":   roc_auc_score(y_test, y_prob_base),
}

print(f"    Training time  : {base_train_ms:.0f} ms")
print(f"    Precision[1]   : {base_metrics['precision']:.4f}")
print(f"    Recall[1]      : {base_metrics['recall']:.4f}")
print(f"    F1-Score[1]    : {base_metrics['f1']:.4f}")
print(f"    ROC-AUC        : {base_metrics['roc_auc']:.4f}")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 2: HYPERPARAMETER TUNING — RandomizedSearchCV
# ══════════════════════════════════════════════════════════════════

print("\n[2/6] Hyperparameter tuning (RandomizedSearchCV, n_iter=60)...")
print("    Metric utama: F1 class Churn (pos_label=1)")
print("    Cross-val: StratifiedKFold k=5  |  Ini butuh ~30–90 detik di Colab...")

# Parameter grid — range dipilih berdasarkan best practices XGBoost tabular
param_dist = {
    "n_estimators":     [100, 200, 300, 400, 500],
    "max_depth":        [3, 4, 5, 6],           # 3–6: sweet spot untuk tabular kecil
    "learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7],        # regularisasi: cegah overfitting
    "gamma":            [0, 0.05, 0.1, 0.2, 0.3],  # min loss reduction to split
    "reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],  # L1
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0],       # L2
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

xgb_base_for_search = XGBClassifier(
    scale_pos_weight  = SCALE_POS_WEIGHT,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = SEED,
    n_jobs            = -1,
)

search = RandomizedSearchCV(
    estimator          = xgb_base_for_search,
    param_distributions= param_dist,
    n_iter             = 60,
    scoring            = "f1",          # optimasi F1 churn class
    cv                 = cv_strategy,
    refit              = True,
    verbose            = 0,
    random_state       = SEED,
    n_jobs             = -1,
    error_score        = "raise",
)

t0 = time.time()
search.fit(X_train, y_train)
tuning_ms = (time.time() - t0) * 1000

best_params = search.best_params_
print(f"\n    Selesai dalam  : {tuning_ms/1000:.1f} detik")
print(f"    Best CV F1     : {search.best_score_:.4f}")
print(f"\n    Best Parameters:")
for k, v in sorted(best_params.items()):
    print(f"      {k:<22} = {v}")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 3: TRAIN MODEL FINAL dengan best params
# ══════════════════════════════════════════════════════════════════

print("\n[3/6] Training model final dengan best parameters...")

# Tambahkan parameter yang tidak ada di search space
final_params = {
    **best_params,
    "scale_pos_weight":  SCALE_POS_WEIGHT,
    "use_label_encoder": False,
    "eval_metric":       "logloss",
    "random_state":      SEED,
    "n_jobs":            -1,
}

model_final = XGBClassifier(**final_params)
model_final.fit(X_train, y_train)

y_pred_final = model_final.predict(X_test)
y_prob_final = model_final.predict_proba(X_test)[:, 1]

final_metrics = {
    "precision": precision_score(y_test, y_pred_final, pos_label=1),
    "recall":    recall_score(y_test, y_pred_final, pos_label=1),
    "f1":        f1_score(y_test, y_pred_final, pos_label=1),
    "roc_auc":   roc_auc_score(y_test, y_prob_final),
}

# ── Cross-validation pada model final (verifikasi tidak overfitting) ──
cv_results = cross_validate(
    model_final, X_train, y_train,
    cv      = cv_strategy,
    scoring = ["precision", "recall", "f1"],
    n_jobs  = -1,
)

print(f"\n    ── Test Set Metrics (Model Final) ──")
print(f"    Precision[1]   : {final_metrics['precision']:.4f}  "
      f"(baseline: {base_metrics['precision']:.4f}  Δ{final_metrics['precision']-base_metrics['precision']:+.4f})")
print(f"    Recall[1]      : {final_metrics['recall']:.4f}  "
      f"(baseline: {base_metrics['recall']:.4f}  Δ{final_metrics['recall']-base_metrics['recall']:+.4f})")
print(f"    F1-Score[1]    : {final_metrics['f1']:.4f}  "
      f"(baseline: {base_metrics['f1']:.4f}  Δ{final_metrics['f1']-base_metrics['f1']:+.4f})")
print(f"    ROC-AUC        : {final_metrics['roc_auc']:.4f}  "
      f"(baseline: {base_metrics['roc_auc']:.4f}  Δ{final_metrics['roc_auc']-base_metrics['roc_auc']:+.4f})")

print(f"\n    ── 5-Fold CV pada Train Set (cek overfitting) ──")
print(f"    CV Precision   : {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"    CV Recall      : {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
print(f"    CV F1          : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

overfitting_gap = cv_results["test_f1"].mean() - final_metrics["f1"]
print(f"\n    CV F1 vs Test F1 gap : {overfitting_gap:+.4f}  "
      f"({'⚠️  Cek overfitting!' if abs(overfitting_gap) > 0.10 else '✓  Dalam batas wajar'})")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 4: EVALUASI MATRIKS LENGKAP
# ══════════════════════════════════════════════════════════════════

print("\n[4/6] Evaluasi matriks lengkap...")

print("\n    ── Classification Report ──")
print(classification_report(
    y_test, y_pred_final,
    target_names=["Aktif (0)", "Churn (1)"],
    digits=4
))

cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print(f"    ── Confusion Matrix ──")
print(f"    TN (Aktif benar)     : {tn}  → pelanggan aktif terdeteksi aktif ✓")
print(f"    FP (Aktif salah flag): {fp}  → pelanggan aktif salah dikirim pesan (tidak berbahaya)")
print(f"    FN (Churn terlewat)  : {fn}  → pelanggan churn TIDAK terdeteksi ← yang harus diminimalkan")
print(f"    TP (Churn terdeteksi): {tp}  → pelanggan churn berhasil ditangkap ✓")
print(f"\n    Churn Detection Rate  : {tp/(tp+fn):.1%} dari total churn tertangkap")
print(f"    False Alarm Rate      : {fp/(fp+tn):.1%} dari aktif salah flagged")

# Threshold analysis — Recall vs Precision tradeoff
thresholds   = np.arange(0.3, 0.75, 0.05)
thresh_data  = []
for t in thresholds:
    y_pred_t = (y_prob_final >= t).astype(int)
    thresh_data.append({
        "threshold": t,
        "precision": precision_score(y_test, y_pred_t, pos_label=1, zero_division=0),
        "recall":    recall_score(y_test, y_pred_t, pos_label=1, zero_division=0),
        "f1":        f1_score(y_test, y_pred_t, pos_label=1, zero_division=0),
    })
df_thresh = pd.DataFrame(thresh_data)

best_thresh_row = df_thresh.loc[df_thresh["f1"].idxmax()]
print(f"\n    ── Threshold Analysis (default=0.50) ──")
print(df_thresh.to_string(index=False, float_format="{:.4f}".format))
print(f"\n    Threshold terbaik (max F1) : {best_thresh_row['threshold']:.2f}"
      f"  →  P={best_thresh_row['precision']:.4f}  R={best_thresh_row['recall']:.4f}  F1={best_thresh_row['f1']:.4f}")

# Rekomendasi threshold untuk Ancora
# Recall lebih penting (jangan sampai pelanggan churn terlewat)
ancora_thresh = df_thresh[df_thresh["recall"] >= 0.85]["threshold"].values
if len(ancora_thresh) > 0:
    rec_thresh = ancora_thresh.max()
    row_rec    = df_thresh[df_thresh["threshold"] == rec_thresh].iloc[0]
    print(f"\n    Rekomendasi threshold untuk Ancora (Recall ≥ 85%) :")
    print(f"    threshold={rec_thresh:.2f}  "
          f"→  P={row_rec['precision']:.4f}  R={row_rec['recall']:.4f}  F1={row_rec['f1']:.4f}")
    RECOMMENDED_THRESHOLD = rec_thresh
else:
    RECOMMENDED_THRESHOLD = 0.50
    print(f"\n    Menggunakan threshold default 0.50")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 5: VISUALISASI
# ══════════════════════════════════════════════════════════════════

print("\n[5/6] Generating visualizations...")

C_NAVY   = "#1E3A5F"
C_BLUE   = "#2D5986"
C_RED    = "#DC2626"
C_GREEN  = "#0E9F6E"
C_AMBER  = "#D97706"
C_LIGHT  = "#EEF3F9"

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Ancora — Modul 2: XGBoost Model Evaluation", fontsize=15, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Confusion Matrix (baseline vs final) ──────────────────
for col_i, (model_lbl, y_pred_i, color) in enumerate([
    ("Baseline", y_pred_base, C_BLUE),
    ("Tuned (Final)", y_pred_final, C_NAVY),
]):
    ax = fig.add_subplot(gs[0, col_i])
    cm_i    = confusion_matrix(y_test, y_pred_i)
    cm_pct  = cm_i / cm_i.sum() * 100
    labels  = np.array([[f"{cm_i[r,c]}\n({cm_pct[r,c]:.1f}%)" for c in range(2)] for r in range(2)])
    sns.heatmap(cm_i, annot=labels, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Aktif", "Churn"],
                yticklabels=["Aktif", "Churn"],
                linewidths=0.5, cbar=False,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title(f"Confusion Matrix\n{model_lbl}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.tick_params(labelsize=9)

# ── Plot 2: Metrics Comparison Bar ────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
metric_names = ["Precision", "Recall", "F1", "ROC-AUC"]
base_vals    = [base_metrics["precision"], base_metrics["recall"],
                base_metrics["f1"],         base_metrics["roc_auc"]]
final_vals   = [final_metrics["precision"], final_metrics["recall"],
                final_metrics["f1"],         final_metrics["roc_auc"]]
x     = np.arange(len(metric_names))
width = 0.35
b1 = ax.bar(x - width/2, base_vals,  width, label="Baseline", color="#CBD5E1", edgecolor="white", linewidth=1.5)
b2 = ax.bar(x + width/2, final_vals, width, label="Tuned",    color=C_NAVY,    edgecolor="white", linewidth=1.5)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title("Baseline vs Tuned Model", fontweight="bold", fontsize=11)
ax.set_ylabel("Score")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 3: ROC Curve ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for model_lbl, y_prob_i, color in [
    ("Baseline", y_prob_base, "#94A3B8"),
    ("Tuned",    y_prob_final, C_NAVY),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob_i)
    auc_val     = roc_auc_score(y_test, y_prob_i)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_lbl} (AUC={auc_val:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
ax.fill_between(*roc_curve(y_test, y_prob_final)[:2], alpha=0.08, color=C_NAVY)
ax.set_xlabel("False Positive Rate", fontsize=9)
ax.set_ylabel("True Positive Rate", fontsize=9)
ax.set_title("ROC Curve", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 4: Precision-Recall Curve ────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for model_lbl, y_prob_i, color in [
    ("Baseline", y_prob_base, "#94A3B8"),
    ("Tuned",    y_prob_final, C_NAVY),
]:
    prec, rec, _ = precision_recall_curve(y_test, y_prob_i)
    ap_val       = average_precision_score(y_test, y_prob_i)
    ax.plot(rec, prec, color=color, lw=2, label=f"{model_lbl} (AP={ap_val:.3f})")
ax.axhline(y_test.mean(), color=C_RED, linestyle="--", lw=1,
           label=f"Random (prevalence={y_test.mean():.2f})")
ax.set_xlabel("Recall", fontsize=9)
ax.set_ylabel("Precision", fontsize=9)
ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 5: Threshold Tradeoff ────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.plot(df_thresh["threshold"], df_thresh["precision"], color=C_BLUE,  lw=2, marker="o", ms=5, label="Precision")
ax.plot(df_thresh["threshold"], df_thresh["recall"],    color=C_RED,   lw=2, marker="s", ms=5, label="Recall")
ax.plot(df_thresh["threshold"], df_thresh["f1"],        color=C_GREEN, lw=2, marker="^", ms=5, label="F1")
ax.axvline(RECOMMENDED_THRESHOLD, color=C_AMBER, linestyle="--", lw=1.5,
           label=f"Rekomendasi ({RECOMMENDED_THRESHOLD:.2f})")
ax.axvline(0.50, color="gray", linestyle=":", lw=1, label="Default (0.50)")
ax.set_xlabel("Threshold", fontsize=9)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("Threshold vs Metrics\n(Untuk Ancora: utamakan Recall)", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 6: Feature Importance ────────────────────────────────────
ax = fig.add_subplot(gs[2, :2])
importances = model_final.feature_importances_
feat_df     = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    .sort_values("importance", ascending=True)
)
colors_bar = [C_NAVY if imp >= np.percentile(importances, 60) else "#94A3B8"
              for imp in feat_df["importance"]]
bars = ax.barh(feat_df["feature"], feat_df["importance"], color=colors_bar, edgecolor="white")
for bar, val in zip(bars, feat_df["importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Feature Importance (XGBoost — gain)", fontweight="bold", fontsize=11)
ax.set_xlabel("Importance Score")
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 7: Churn Probability Distribution ────────────────────────
ax = fig.add_subplot(gs[2, 2])
for lbl, color, name in [(0, C_BLUE, "Aktif"), (1, C_RED, "Churn")]:
    probs = y_prob_final[y_test == lbl]
    ax.hist(probs, bins=20, alpha=0.7, color=color, label=name, edgecolor="white")
ax.axvline(RECOMMENDED_THRESHOLD, color=C_AMBER, linestyle="--", lw=2,
           label=f"Threshold ({RECOMMENDED_THRESHOLD:.2f})")
ax.set_xlabel("Predicted Probability (Churn)", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.set_title("Distribusi Probabilitas Prediksi", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR, "ancora_modul2_evaluation.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved: ancora_modul2_evaluation.png")


# ── Chart 2: Hyperparameter Search Results ────────────────────────
cv_df = pd.DataFrame(search.cv_results_)
cv_df = cv_df.sort_values("rank_test_score").head(20)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Hyperparameter Search — Top 20 Kombinasi", fontweight="bold", fontsize=13)

for ax, param, color in zip(
    axes2,
    ["param_max_depth", "param_learning_rate", "param_n_estimators"],
    [C_NAVY, C_BLUE, "#4A7AB5"],
):
    if param not in cv_df.columns:
        continue
    ax.scatter(cv_df[param].astype(float), cv_df["mean_test_score"],
               c=color, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(param.replace("param_", ""), fontsize=10)
    ax.set_ylabel("CV F1 Score")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ancora_modul2_hparam_search.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved: ancora_modul2_hparam_search.png")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 6: EXPORT MODEL
# ══════════════════════════════════════════════════════════════════

print("\n[6/6] Exporting model artifacts...")

def save_pkl(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size = os.path.getsize(path)
    print(f"    Saved: {filename} ({size:,} bytes)")

# Model utama
save_pkl(model_final, "ancora_xgb_model.pkl")

# Metadata model — dipakai FastAPI untuk self-documentation endpoint
model_meta = {
    "model_type":            "XGBClassifier",
    "feature_cols":          FEATURE_COLS,
    "n_features":            len(FEATURE_COLS),
    "best_params":           best_params,
    "scale_pos_weight":      SCALE_POS_WEIGHT,
    "recommended_threshold": float(RECOMMENDED_THRESHOLD),
    "test_metrics": {
        "precision":  round(float(final_metrics["precision"]), 4),
        "recall":     round(float(final_metrics["recall"]), 4),
        "f1":         round(float(final_metrics["f1"]), 4),
        "roc_auc":    round(float(final_metrics["roc_auc"]), 4),
    },
    "cv_f1_mean": round(float(cv_results["test_f1"].mean()), 4),
    "cv_f1_std":  round(float(cv_results["test_f1"].std()), 4),
}
save_pkl(model_meta, "ancora_model_meta.pkl")

# Simpan juga best_params sebagai text untuk referensi
meta_txt_path = os.path.join(OUTPUT_DIR, "ancora_model_meta.txt")
with open(meta_txt_path, "w") as f:
    f.write("ANCORA XGBoost Model — Best Parameters\n")
    f.write("=" * 40 + "\n")
    for k, v in sorted(best_params.items()):
        f.write(f"{k:<22} = {v}\n")
    f.write("\nTest Set Metrics (pos_label=1 / Churn)\n")
    f.write("-" * 40 + "\n")
    for k, v in model_meta["test_metrics"].items():
        f.write(f"{k:<22} = {v:.4f}\n")
    f.write(f"\nRecommended Threshold  = {RECOMMENDED_THRESHOLD:.2f}\n")
    f.write(f"CV F1 (5-fold)         = {model_meta['cv_f1_mean']:.4f} ± {model_meta['cv_f1_std']:.4f}\n")
print(f"    Saved: ancora_model_meta.txt")


# ══════════════════════════════════════════════════════════════════
# RINGKASAN AKHIR
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("  MODUL 2 SELESAI — Ringkasan Output")
print("=" * 62)
print(f"\n  Model final — Test Set Metrics (churn class):")
print(f"    Precision  : {final_metrics['precision']:.4f}")
print(f"    Recall     : {final_metrics['recall']:.4f}  ← utama untuk Ancora")
print(f"    F1         : {final_metrics['f1']:.4f}")
print(f"    ROC-AUC    : {final_metrics['roc_auc']:.4f}")
print(f"    CV F1 (5K) : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

print(f"\n  Files untuk di-download:")
print(f"    ✓  ancora_xgb_model.pkl      — model utama → taruh di FastAPI /models/")
print(f"    ✓  ancora_model_meta.pkl     — metadata model → dipakai FastAPI startup")
print(f"    ✓  ancora_model_meta.txt     — parameter referensi (human-readable)")
print(f"    ✓  ancora_modul2_evaluation.png")
print(f"    ✓  ancora_modul2_hparam_search.png")

print(f"\n  Cara load di FastAPI (Modul 4):")
print(f"    import joblib")
print(f"    model  = pickle.load(open('models/ancora_xgb_model.pkl', 'rb'))")
print(f"    scaler = pickle.load(open('models/scaler.pkl', 'rb'))")
print(f"    meta   = pickle.load(open('models/ancora_model_meta.pkl', 'rb'))")
print(f"    threshold = meta['recommended_threshold']  # {RECOMMENDED_THRESHOLD:.2f}")

print(f"\n  → Lanjut ke Modul 3: Gemini Prompt Engineering")
print("=" * 62)
