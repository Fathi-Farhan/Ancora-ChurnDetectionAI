"""
╔══════════════════════════════════════════════════════════════════╗
║        ANCORA — Modul 1: Pengumpulan & Rekayasa Data            ║
║        Google Colab / Python 3.10+                              ║
║                                                                  ║
║  Checklist:                                                      ║
║  [✓] Generate Dataset Sintetik (500 pelanggan, 85/15 split)     ║
║  [✓] Feature Engineering (RFM+: Recency, Frequency, Monetary,  ║
║       Service Diversity, Trend)                                  ║
║  [✓] Pipeline Preprocessing (StandardScaler + SMOTE)            ║
║  [✓] Export Artifacts (scaler.pkl, feature_cols.pkl, CSV)       ║
╚══════════════════════════════════════════════════════════════════╝

Cara pakai di Google Colab:
    1. Upload file ini ke Colab
    2. !pip install xgboost imbalanced-learn -q
    3. !python ancora_modul1_data_pipeline.py
    4. Download file output dari panel Files (kiri)
"""

# ── IMPORTS ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")   # Colab-safe: tidak butuh display window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── SEED & KONSTANTA ───────────────────────────────────────────────
SEED        = 42
N_TOTAL     = 500
CHURN_RATIO = 0.15
N_CHURN     = int(N_TOTAL * CHURN_RATIO)   # 75
N_ACTIVE    = N_TOTAL - N_CHURN            # 425
OUTPUT_DIR  = "."                          # ganti ke "/content" jika di Colab

np.random.seed(SEED)

print("=" * 60)
print("  ANCORA — Modul 1: Data Pipeline")
print(f"  Total: {N_TOTAL} | Active: {N_ACTIVE} | Churn: {N_CHURN}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════
# BAGIAN 1: KONFIGURASI DOMAIN
# ══════════════════════════════════════════════════════════════════

BUSINESS_TYPES   = ["Salon", "Laundry", "Bengkel"]
BUSINESS_WEIGHTS = [0.45, 0.35, 0.20]

# Harga rata-rata per kunjungan (Rupiah)
PRICE_RANGES = {
    "Salon":   (50_000,  300_000),
    "Laundry": (15_000,   80_000),
    "Bengkel": (50_000,  500_000),
}

# Frekuensi kunjungan dalam 90 hari — sesuai siklus bisnis nyata
VISIT_FREQ = {
    "Salon":   {"active": (5, 12), "churn": (1, 3)},
    "Laundry": {"active": (8, 20), "churn": (1, 4)},
    "Bengkel": {"active": (2, 6),  "churn": (1, 2)},
}

# Recency = hari sejak kunjungan terakhir
RECENCY_RANGE = {
    "Salon":   {"active": (3, 28),  "churn": (45, 90)},
    "Laundry": {"active": (2, 18),  "churn": (30, 90)},
    "Bengkel": {"active": (7, 45),  "churn": (60, 120)},
}

# Jumlah jenis layanan unik yang pernah digunakan
SERVICE_DIVERSITY = {
    "Salon":   {"active": (2, 5), "churn": (1, 2)},
    "Laundry": {"active": (1, 3), "churn": (1, 2)},
    "Bengkel": {"active": (2, 4), "churn": (1, 2)},
}


# ══════════════════════════════════════════════════════════════════
# BAGIAN 2: GENERATE DATASET SINTETIK
# ══════════════════════════════════════════════════════════════════

def _generate_group(n: int, label: int, id_offset: int = 0) -> list[dict]:
    """
    Buat n customer dengan label churn tertentu (0=aktif, 1=churn).
    Menggunakan distribusi segitiga (triangular) agar lebih natural
    daripada distribusi seragam — mirip pola kunjungan pelanggan nyata.
    """
    status   = "active" if label == 0 else "churn"
    biz_list = np.random.choice(BUSINESS_TYPES, size=n, p=BUSINESS_WEIGHTS)
    rows     = []

    for i, biz in enumerate(biz_list):
        # ── Recency ────────────────────────────────────────────
        r_lo, r_hi = RECENCY_RANGE[biz][status]
        r_mode     = r_lo + (r_hi - r_lo) * (0.25 if label == 0 else 0.65)
        recency    = int(np.clip(np.random.triangular(r_lo, r_mode, r_hi), r_lo, r_hi))

        # ── Frequency ──────────────────────────────────────────
        f_lo, f_hi = VISIT_FREQ[biz][status]
        frequency  = int(np.random.randint(f_lo, f_hi + 1))

        # ── Monetary ───────────────────────────────────────────
        p_lo, p_hi = PRICE_RANGES[biz]
        avg_price  = np.random.uniform(p_lo, p_hi)
        # Churn customers punya transaksi terakhir lebih kecil
        multiplier = np.random.uniform(0.6, 0.9) if label == 1 else 1.0
        monetary   = round(avg_price * frequency * multiplier / 1000) * 1000

        # ── Service Diversity ──────────────────────────────────
        d_lo, d_hi     = SERVICE_DIVERSITY[biz][status]
        svc_diversity  = int(np.random.randint(d_lo, d_hi + 1))

        # ── Avg Visit Gap (hari antar kunjungan) ───────────────
        avg_gap = round(90 / frequency, 1) if frequency > 0 else 90.0

        # ── Trend (rasio kunjungan 30 hari terakhir vs sebelumnya) ──
        # > 1.0 = frekuensi naik (loyal), < 1.0 = turun (risiko churn)
        if label == 0:
            trend = round(np.random.triangular(0.75, 1.20, 2.50), 3)
        else:
            trend = round(np.random.triangular(0.00, 0.30, 0.75), 3)

        rows.append({
            "customer_id":      f"CUST_{id_offset + i + 1:04d}",
            "business_type":    biz,
            "recency":          recency,
            "frequency":        frequency,
            "monetary":         monetary,
            "service_diversity": svc_diversity,
            "avg_visit_gap":    avg_gap,
            "trend":            trend,
            "churn":            label,
        })

    return rows


def inject_anomalies(active_rows: list[dict], pct: float = 0.06) -> list[dict]:
    """
    Injeksi anomali realistis ke ~6% pelanggan aktif:
    - "Dormant-but-back": sempat menghilang lama, lalu balik dengan trend positif.
    - "Loyal-high-gap": kunjungan jarang tapi konsisten (misal bengkel 2x/kuartal).
    Ini mencegah model terlalu mudah overfitting pada pola linear.
    """
    n_anomaly = max(1, int(len(active_rows) * pct))
    indices   = np.random.choice(len(active_rows), size=n_anomaly, replace=False)

    for idx in indices:
        biz = active_rows[idx]["business_type"]
        anomaly_type = np.random.choice(["dormant_back", "loyal_high_gap"])

        if anomaly_type == "dormant_back":
            # Recency tinggi tapi trend sangat positif (baru balik)
            active_rows[idx]["recency"]  = np.random.randint(40, 70)
            active_rows[idx]["trend"]    = round(np.random.uniform(1.5, 2.5), 3)
            active_rows[idx]["frequency"] = np.random.randint(2, 5)

        elif anomaly_type == "loyal_high_gap":
            # Pelanggan setia bengkel yang jarang tapi selalu balik
            if biz == "Bengkel":
                active_rows[idx]["avg_visit_gap"] = np.random.uniform(40, 60)
                active_rows[idx]["monetary"]      = round(np.random.uniform(200_000, 600_000) / 1000) * 1000
                active_rows[idx]["trend"]         = round(np.random.uniform(0.9, 1.3), 3)

    return active_rows


print("\n[1/5] Generating synthetic dataset...")
active_rows = _generate_group(N_ACTIVE, label=0, id_offset=0)
churn_rows  = _generate_group(N_CHURN,  label=1, id_offset=N_ACTIVE)

active_rows = inject_anomalies(active_rows, pct=0.06)

df_raw = (
    pd.DataFrame(active_rows + churn_rows)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print(f"    Rows generated : {len(df_raw)}")
print(f"    Churn dist     : {df_raw['churn'].value_counts().to_dict()}")
print(f"    Business dist  : {df_raw['business_type'].value_counts().to_dict()}")
print(f"    Recency (mean) : Active={df_raw[df_raw.churn==0]['recency'].mean():.1f} days"
      f"  |  Churn={df_raw[df_raw.churn==1]['recency'].mean():.1f} days")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 3: FEATURE ENGINEERING (RFM+)
# ══════════════════════════════════════════════════════════════════

print("\n[2/5] Feature Engineering (RFM+)...")

df = df_raw.copy()

# -- Feature tambahan turunan ──────────────────────────────────────

# Monetary per kunjungan (average transaction value)
df["monetary_per_visit"] = (df["monetary"] / df["frequency"].clip(lower=1)).round(0)

# Visit consistency score: frekuensi relatif terhadap gap ideal
# Semakin kecil selisih antara avg_gap dan gap ideal bisnis, semakin konsisten
GAP_IDEAL = {"Salon": 14, "Laundry": 7, "Bengkel": 30}
df["gap_ideal"] = df["business_type"].map(GAP_IDEAL)
df["visit_consistency"] = (1 / (1 + abs(df["avg_visit_gap"] - df["gap_ideal"]))).round(4)

# Recency bucket: 0=sangat baru, 1=normal, 2=lama, 3=sangat lama
def recency_bucket(row):
    thresholds = {"Salon": [14, 30, 60], "Laundry": [10, 21, 40], "Bengkel": [30, 60, 90]}
    t = thresholds[row["business_type"]]
    if row["recency"] <= t[0]:   return 0
    elif row["recency"] <= t[1]: return 1
    elif row["recency"] <= t[2]: return 2
    else:                        return 3

df["recency_bucket"] = df.apply(recency_bucket, axis=1)

# Engagement score gabungan (composite — bukan fitur utama, hanya diagnostik)
df["engagement_score"] = (
    (1 / (df["recency"] + 1)) * 0.35 +
    df["frequency"]            * 0.25 +
    df["trend"]                * 0.25 +
    df["visit_consistency"]    * 0.15
).round(4)

# One-hot encode business_type
df_enc = pd.get_dummies(df, columns=["business_type"], prefix="biz", dtype=int)

FEATURE_COLS = [
    # RFM core
    "recency", "frequency", "monetary", "service_diversity",
    # Derived
    "avg_visit_gap", "trend", "monetary_per_visit",
    "visit_consistency", "recency_bucket",
    # Business type (one-hot)
    "biz_Salon", "biz_Laundry", "biz_Bengkel",
]

# Pastikan semua kolom ada (Colab safety check)
missing_cols = [c for c in FEATURE_COLS if c not in df_enc.columns]
if missing_cols:
    raise ValueError(f"Kolom tidak ditemukan: {missing_cols}")

X_raw = df_enc[FEATURE_COLS].copy()
y     = df_enc["churn"].copy()

print(f"    Features       : {len(FEATURE_COLS)} kolom")
print(f"    X shape        : {X_raw.shape}")
print(f"    Feature list   : {FEATURE_COLS}")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 4: PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════

print("\n[3/5] Preprocessing (StandardScaler + SMOTE)...")

# Kolom numerik yang perlu di-scale (bukan binary/ordinal kecil)
NUMERIC_COLS = [
    "recency", "frequency", "monetary", "service_diversity",
    "avg_visit_gap", "trend", "monetary_per_visit", "visit_consistency",
]
PASSTHROUGH_COLS = ["recency_bucket", "biz_Salon", "biz_Laundry", "biz_Bengkel"]

# ── StandardScaler ─────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = X_raw.copy()
X_scaled[NUMERIC_COLS] = scaler.fit_transform(X_raw[NUMERIC_COLS])

print(f"    Scaler fitted on : {NUMERIC_COLS}")
print(f"    Scale (mean) sample : recency={scaler.mean_[0]:.1f}d, "
      f"frequency={scaler.mean_[1]:.1f}, monetary=Rp{scaler.mean_[2]:,.0f}")

# ── Train / Test Split SEBELUM SMOTE (penting: SMOTE hanya pada train set) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"\n    Train size     : {len(X_train)} | Test size: {len(X_test)}")
print(f"    Train churn    : {y_train.value_counts().to_dict()}")
print(f"    Test  churn    : {y_test.value_counts().to_dict()}")

# ── SMOTE hanya pada TRAIN set ─────────────────────────────────────
smote = SMOTE(
    random_state=SEED,
    k_neighbors=5,
    sampling_strategy=1.0,  # ratio akhir active:churn = 1:1 di train set
)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\n    Before SMOTE   : {y_train.value_counts().to_dict()}")
print(f"    After  SMOTE   : {pd.Series(y_train_sm).value_counts().to_dict()}")
print(f"    Train size after SMOTE: {len(X_train_sm)}")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 5: EXPORT ARTIFACTS
# ══════════════════════════════════════════════════════════════════

print("\n[4/5] Exporting artifacts...")

def save_pkl(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size = os.path.getsize(path)
    print(f"    Saved: {filename} ({size:,} bytes)")

# Scaler
save_pkl(scaler, "scaler.pkl")

# Feature column names — WAJIB konsisten dengan endpoint FastAPI
save_pkl(FEATURE_COLS, "feature_cols.pkl")

# Numeric cols list — dipakai ulang saat inferensi
save_pkl(NUMERIC_COLS, "numeric_cols.pkl")

# Train/test splits (siap untuk Modul 2)
splits = {
    "X_train": X_train_sm,
    "y_train": y_train_sm,
    "X_test":  X_test.values,   # numpy array untuk XGBoost
    "y_test":  y_test.values,
}
save_pkl(splits, "train_test_splits.pkl")

# Raw dataset CSV
raw_path = os.path.join(OUTPUT_DIR, "ancora_dataset_raw.csv")
df_raw.to_csv(raw_path, index=False)
print(f"    Saved: ancora_dataset_raw.csv ({len(df_raw)} rows)")

# Preprocessed dataset CSV
X_train_df          = pd.DataFrame(X_train_sm, columns=FEATURE_COLS)
X_train_df["churn"] = y_train_sm
prep_path           = os.path.join(OUTPUT_DIR, "ancora_dataset_preprocessed.csv")
X_train_df.to_csv(prep_path, index=False)
print(f"    Saved: ancora_dataset_preprocessed.csv ({len(X_train_df)} rows after SMOTE)")


# ══════════════════════════════════════════════════════════════════
# BAGIAN 6: VISUALISASI
# ══════════════════════════════════════════════════════════════════

print("\n[5/5] Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Ancora — Modul 1: Data Exploration", fontsize=15, fontweight="bold", y=1.01)
palette = {0: "#2D5986", 1: "#DC2626"}
labels  = {0: "Aktif", 1: "Churn"}

# 1. Churn distribution
ax = axes[0, 0]
counts   = df_raw["churn"].value_counts().sort_index()
colors   = [palette[i] for i in counts.index]
bars     = ax.bar([labels[i] for i in counts.index], counts.values, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val}\n({val/N_TOTAL:.0%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Distribusi Churn", fontweight="bold")
ax.set_ylabel("Jumlah Pelanggan")
ax.set_ylim(0, max(counts.values) * 1.2)
ax.spines[["top", "right"]].set_visible(False)

# 2. Recency distribution by churn label
ax = axes[0, 1]
for label_val, color in palette.items():
    subset = df_raw[df_raw["churn"] == label_val]["recency"]
    ax.hist(subset, bins=20, alpha=0.7, color=color, label=labels[label_val], edgecolor="white")
ax.set_title("Distribusi Recency", fontweight="bold")
ax.set_xlabel("Hari Sejak Kunjungan Terakhir")
ax.set_ylabel("Frekuensi")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# 3. Frequency distribution by churn label
ax = axes[0, 2]
for label_val, color in palette.items():
    subset = df_raw[df_raw["churn"] == label_val]["frequency"]
    ax.hist(subset, bins=15, alpha=0.7, color=color, label=labels[label_val], edgecolor="white")
ax.set_title("Distribusi Frequency (90 hari)", fontweight="bold")
ax.set_xlabel("Jumlah Kunjungan")
ax.set_ylabel("Frekuensi")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# 4. Trend distribution by churn label
ax = axes[1, 0]
for label_val, color in palette.items():
    subset = df_raw[df_raw["churn"] == label_val]["trend"]
    ax.hist(subset, bins=20, alpha=0.7, color=color, label=labels[label_val], edgecolor="white")
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Trend = 1.0 (stabil)")
ax.set_title("Distribusi Trend Score", fontweight="bold")
ax.set_xlabel("Rasio Kunjungan 30d Terakhir vs Sebelumnya")
ax.set_ylabel("Frekuensi")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# 5. Monetary boxplot by business type
ax = axes[1, 1]
biz_order  = ["Salon", "Laundry", "Bengkel"]
biz_colors = ["#4A7AB5", "#0E9F6E", "#D97706"]
data_to_plot = [df_raw[df_raw["business_type"] == b]["monetary"].values / 1000 for b in biz_order]
bp = ax.boxplot(data_to_plot, labels=biz_order, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], biz_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title("Monetary per Jenis Bisnis", fontweight="bold")
ax.set_ylabel("Total Spend (Rp ribu, 90 hari)")
ax.spines[["top", "right"]].set_visible(False)

# 6. Before/after SMOTE
ax = axes[1, 2]
before = pd.Series(y_train).value_counts().sort_index()
after  = pd.Series(y_train_sm).value_counts().sort_index()
x      = np.arange(2)
width  = 0.35
b1 = ax.bar(x - width/2, before.values, width, label="Before SMOTE",
            color=["#CBD5E1", "#FCA5A5"], edgecolor="white")
b2 = ax.bar(x + width/2, after.values,  width, label="After SMOTE",
            color=[palette[0], palette[1]], edgecolor="white", alpha=0.85)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Aktif (0)", "Churn (1)"])
ax.set_title("Before vs After SMOTE (Train Set)", fontweight="bold")
ax.set_ylabel("Jumlah Sampel")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "ancora_modul1_eda.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Saved: ancora_modul1_eda.png")

# ── Correlation heatmap (features vs churn) ────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
numeric_features = ["recency", "frequency", "monetary", "service_diversity",
                    "avg_visit_gap", "trend", "engagement_score", "churn"]
corr_matrix = df[numeric_features].corr()
corr_with_churn = corr_matrix["churn"].drop("churn").sort_values()

colors_bar = ["#DC2626" if v > 0 else "#2D5986" for v in corr_with_churn.values]
ax.barh(corr_with_churn.index, corr_with_churn.values, color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
for i, (name, val) in enumerate(corr_with_churn.items()):
    ax.text(val + (0.01 if val >= 0 else -0.01), i,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
ax.set_title("Korelasi Fitur terhadap Label Churn", fontweight="bold", fontsize=12)
ax.set_xlabel("Pearson Correlation Coefficient")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, "ancora_modul1_correlation.png")
plt.savefig(corr_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Saved: ancora_modul1_correlation.png")


# ══════════════════════════════════════════════════════════════════
# RINGKASAN AKHIR
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  MODUL 1 SELESAI — Ringkasan Output")
print("=" * 60)

summary = {
    "ancora_dataset_raw.csv":           f"{len(df_raw)} baris, {df_raw.shape[1]} kolom",
    "ancora_dataset_preprocessed.csv":  f"{len(X_train_df)} baris setelah SMOTE",
    "scaler.pkl":                       f"StandardScaler fitted — {len(NUMERIC_COLS)} fitur numerik",
    "feature_cols.pkl":                 f"{len(FEATURE_COLS)} nama kolom fitur",
    "numeric_cols.pkl":                 f"{len(NUMERIC_COLS)} kolom numeric",
    "train_test_splits.pkl":            f"X_train {X_train_sm.shape} | X_test {X_test.shape}",
    "ancora_modul1_eda.png":            "6 chart eksplorasi data",
    "ancora_modul1_correlation.png":    "Feature correlation vs churn",
}

for fname, desc in summary.items():
    print(f"  {'✓':2s} {fname:<42} {desc}")

print(f"\n  Total fitur untuk Modul 2: {len(FEATURE_COLS)}")
print(f"  Train samples (post-SMOTE): {len(X_train_sm)}")
print(f"  Test  samples (stratified): {len(X_test)}")

print("\n  → Lanjut ke Modul 2: Training XGBClassifier")
print("     Load: train_test_splits.pkl, feature_cols.pkl, scaler.pkl")
print("=" * 60)
