# Ancora Churn Detection AI — API Reference

This document outlines the API specifications for the FastAPI backend.

## Base URL
The backend is hosted locally by default at:
`http://localhost:8000`

---

## 1. Sentiment Analysis

### `POST /api/v1/sentiment`
Analyzes the sentiment of a customer feedback text.

#### Request Headers
- `Content-Type: application/json`

#### Request Body
```json
{
  "text": "Pelayanannya lambat banget, udah gitu hasilnya kurang bersih",
  "customer_id": "CUST_0042"
}
```

#### Response Body
```json
{
  "label": "NEGATIVE",
  "score": 0.87,
  "reason": "komplain pelayanan lambat dan kualitas tidak memuaskan",
  "customer_id": "CUST_0042"
}
```

---

## 2. AI Message Generator

### `POST /api/v1/messages/generate`
Generates 3 variations of customized WhatsApp retention messages based on the customer profile.

#### Request Headers
- `Content-Type: application/json`

#### Request Body
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

#### Response Body
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

## 3. Batch Churn Detection

### `POST /api/v1/jobs/churn-detection`
*(Internal Endpoint — Called by Cloud Scheduler)*

Runs a batch prediction job to update the Heartbeat Scores for all customers in the database.

#### Request Headers
- `Authorization: Bearer <token>`
