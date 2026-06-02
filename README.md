<div align="center">

<img src="docs/images/ancora-banner.png?v=2" alt="Ancora Banner" width="100%" />

# Ancora — AI-Powered Customer Retention for Indonesian MSMEs

**Detect silent churn before it costs you a customer.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-Google_AI-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?style=flat-square&logo=flutter&logoColor=white)](https://flutter.dev)
[![Firestore](https://img.shields.io/badge/Firestore-Firebase-FFCA28?style=flat-square&logo=firebase&logoColor=black)](https://firebase.google.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> Dibuat untuk **Gunadarma Code Week 2.0** oleh Tim **MakeNoMistake**

</div>

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Solusi](#solusi)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Tech Stack](#tech-stack)
- [Struktur Repository](#struktur-repository)
- [Cara Setup](#cara-setup)
- [API Reference](#api-reference)
- [Tim](#tim)

---

## Latar Belakang

Lebih dari **64 juta UMKM** beroperasi di Indonesia, sebagian besar di sektor jasa seperti salon, laundry, dan bengkel. Tantangan terbesar yang mereka hadapi bukan sekadar mendapatkan pelanggan baru — melainkan **mempertahankan pelanggan yang sudah ada**.

Masalahnya: pelanggan yang berhenti datang jarang berpamitan. Mereka hanya berhenti — tanpa komplain, tanpa notifikasi. Fenomena ini dikenal sebagai **silent churn**, dan owner UMKM biasanya baru sadar setelah beberapa bulan kehilangan pendapatan.

Tanpa data dan alat yang tepat, owner tidak tahu:
- Siapa pelanggan yang mulai menjauh?
- Kapan waktu terbaik untuk melakukan re-engagement?
- Pesan seperti apa yang relevan untuk dikirim?

---

## Solusi

**Ancora** adalah platform retensi pelanggan berbasis AI yang dirancang khusus untuk UMKM jasa Indonesia. Nama *Ancora* berasal dari bahasa Italia yang berarti *jangkar* — sebuah metafora untuk menahan pelanggan agar tidak hanyut pergi.

Ancora bekerja dalam tiga lapisan:

```
Transaksi masuk  →  Heartbeat Score dihitung  →  Pelanggan berisiko terdeteksi
                                                          ↓
                                          Pesan WhatsApp personal digenerate
                                                          ↓
                                          Owner kirim, pelanggan kembali
```

**Heartbeat Score** adalah skor 0–100 yang merepresentasikan probabilitas seorang pelanggan untuk kembali dalam 30 hari ke depan. Skor ini dihitung otomatis setiap kali ada transaksi baru, menggunakan model XGBoost yang dilatih pada pola kunjungan historis.

---

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Heartbeat Score** | Skor churn real-time berbasis XGBoost untuk setiap pelanggan |
| **Weekly Churn Alert** | Dashboard mingguan yang menampilkan pelanggan berisiko tinggi |
| **AI Message Generator** | Pesan WhatsApp personal via Gemini 1.5 Flash, disesuaikan per vertikal bisnis |
| **Sentiment Analysis** | Analisis feedback pelanggan pasca-layanan secara otomatis |
| **PWA Booking** | Halaman booking pelanggan tanpa instalasi aplikasi |
| **Saved Revenue Dashboard** | Estimasi pendapatan yang berhasil dipertahankan dari program retensi |

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    FLUTTER APP (Owner)                  │
│   Dashboard · Weekly Alert · Message Generator · Booking│
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / REST
┌──────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                        │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  /sentiment │  │  /messages   │  │ /churn-detect │  │
│  │  (Gemini)   │  │  (Gemini)    │  │  (XGBoost)    │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                  │           │
│  ┌──────▼────────────────▼──────────────────▼───────┐  │
│  │              Google AI (Gemini 1.5 Flash)         │  │
│  │              XGBoost Model (.pkl)                 │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                FIRESTORE (Database)                     │
│   customers · transactions · heartbeat_scores · bookings│
└─────────────────────────────────────────────────────────┘
```

Cloud Scheduler (Google Cloud) mentrigger job churn detection setiap Senin pukul 06.00 WIB untuk memperbarui Heartbeat Score seluruh pelanggan secara batch.

---

## Tech Stack

### Machine Learning
| Komponen | Teknologi | Keterangan |
|---|---|---|
| Data Pipeline | Python + scikit-learn | Synthetic data generation, RFM+ feature engineering |
| Resampling | imbalanced-learn (SMOTE) | Menangani imbalanced class (85% aktif / 15% churn) |
| Model | XGBoost (XGBClassifier) | Churn prediction + Heartbeat Score |
| Experiment | Google Colab | Training & hyperparameter tuning |

### AI Layer
| Komponen | Teknologi | Keterangan |
|---|---|---|
| Sentiment Analysis | Gemini 1.5 Flash | Analisis feedback pelanggan → JSON output |
| Message Generation | Gemini 1.5 Flash | WhatsApp retention message per vertikal bisnis |

### Backend
| Komponen | Teknologi | Keterangan |
|---|---|---|
| Framework | FastAPI | REST API + async support |
| Hosting | Railway / Render | Free tier deployment |
| Scheduler | Google Cloud Scheduler | Weekly churn detection trigger |

### Frontend & Database
| Komponen | Teknologi | Keterangan |
|---|---|---|
| Mobile App | Flutter | Cross-platform (Android + iOS) |
| Database | Firestore | Real-time NoSQL |
| Booking Page | PWA | Tanpa instalasi aplikasi |

---

## Struktur Repository

```
Ancora-ChurnDetectionAI/
│
├── README.md
├── .env.example                        # Template environment variables
├── .gitignore
│
├── ml/                                 # Machine Learning pipeline
│   ├── scripts/
│   │   ├── ancora_modul1_data_pipeline.py
│   │   └── ancora_modul2_xgboost_training.py
│   ├── models/
│   │   └── .gitkeep                    # File .pkl tidak di-commit (lihat .gitignore)
│   └── data/
│       └── ancora_dataset_raw.csv      # Synthetic dataset (500 pelanggan)
│
├── backend/                            # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   ├── services/
│   │   └── gemini_service.py           # Gemini client + prompt functions
│   ├── routers/
│   │   ├── sentiment.py                # POST /api/v1/sentiment
│   │   └── messages.py                 # POST /api/v1/messages/generate
│   └── schemas/
│       └── ai_schemas.py               # Pydantic request/response models
│
├── frontend/                           # Flutter app
│   └── ...
│
└── docs/
    ├── architecture.md
    ├── api-reference.md
    └── images/
        ├── ancora-banner.png
        ├── system-design.png
        └── user-journey.png
```

---

## Cara Setup

### Prasyarat
- Python 3.10+
- Flutter 3.x
- Node.js 18+ (opsional, untuk tooling)
- Akun Google AI Studio (untuk Gemini API key — gratis)
- Akun Firebase (untuk Firestore)

### 1. Clone Repository

```bash
git clone https://github.com/Fathi-Farhan/Ancora-ChurnDetectionAI.git
cd Ancora-ChurnDetectionAI
```

### 2. Jalankan ML Pipeline (Google Colab)

Upload dan jalankan secara berurutan:

```bash
# Di Google Colab:
!pip install xgboost imbalanced-learn -q
!python ml/scripts/ancora_modul1_data_pipeline.py
!python ml/scripts/ancora_modul2_xgboost_training.py
```

Unduh `ancora_xgb_model.pkl` dan `scaler.pkl`, lalu taruh di `backend/models/`.

### 3. Setup Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Salin template environment dan isi dengan nilai yang sesuai:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_google_ai_studio_api_key
FIREBASE_CREDENTIALS_PATH=path/to/serviceAccountKey.json
MODEL_PATH=models/ancora_xgb_model.pkl
SCALER_PATH=models/scaler.pkl
CHURN_THRESHOLD=0.45
```

Jalankan server:

```bash
uvicorn main:app --reload --port 8000
```

API docs tersedia di: `http://localhost:8000/docs`

### 4. Setup Flutter

```bash
cd frontend
flutter pub get
flutter run
```

---

## API Reference

### `POST /api/v1/sentiment`
Menganalisis sentimen teks feedback pelanggan.

**Request:**
```json
{
  "text": "Pelayanannya lambat banget, udah gitu hasilnya kurang bersih",
  "customer_id": "CUST_0042"
}
```

**Response:**
```json
{
  "label": "NEGATIVE",
  "score": 0.87,
  "reason": "komplain pelayanan lambat dan kualitas tidak memuaskan",
  "customer_id": "CUST_0042"
}
```

---

### `POST /api/v1/messages/generate`
Menghasilkan 3 variasi pesan WhatsApp retensi berbasis profil pelanggan.

**Request:**
```json
{
  "customer_name": "Kak Sari",
  "business_type": "Salon",
  "business_name": "Salon Cantik Bu Yani",
  "last_service": "Blow dry + Creambath",
  "recency_days": 38,
  "churn_score": 0.72
}
```

**Response:**
```json
{
  "variants": [
    {
      "type": "gentle_reminder",
      "message": "Halo Kak Sari! Sudah sebulan lebih nih sejak terakhir blow dry di sini. Rambut kamu masih oke kan? Kita tunggu kamu balik ya Kak 🌸"
    },
    {
      "type": "promo_offer",
      "message": "Kak Sari, spesial buat pelanggan setia — minggu ini ada diskon 20% untuk creambath. Mau booking slot-nya? Langsung balas aja ya!"
    },
    {
      "type": "personal_touch",
      "message": "Kak Sari, Bu Yani nanya nih — rambutmu gimana kabarnya? Udah lama banget ga keliatan. Kangen deh. Mampir yuk kapan-kapan!"
    }
  ]
}
```

---

### `POST /api/v1/jobs/churn-detection` *(Internal — Cloud Scheduler)*
Memperbarui Heartbeat Score seluruh pelanggan secara batch. Endpoint ini hanya bisa dipanggil oleh Cloud Scheduler dengan header otorisasi internal.

---

## Tim

**MakeNoMistake** — Gunadarma Code Week 2.0

| Nama | Peran | Tanggung Jawab |
|---|---|---|
| [Fathi Farhan](https://github.com/Fathi-Farhan) | Hacker | ML Pipeline, FastAPI Backend, System Architecture |
| Nanda Manarfa | Hipster | UI/UX Design, Flutter Frontend, User Research |
| *(Hustler)* | Hustler | Business Strategy, BMC, Go-To-Market |

---

## Kontribusi SDGs

Ancora mendukung dua tujuan pembangunan berkelanjutan:

- **SDG 8** — *Decent Work and Economic Growth*: Membantu jutaan pelaku UMKM jasa mempertahankan pendapatan dan menstabilkan bisnis mereka.
- **SDG 9** — *Industry, Innovation and Infrastructure*: Membawa teknologi AI yang selama ini hanya bisa diakses enterprise ke tangan pelaku usaha kecil.

---

## Lisensi

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

<div align="center">

Dibuat dengan ❤️ oleh Tim MakeNoMistake · Universitas Gunadarma · 2025

</div>
