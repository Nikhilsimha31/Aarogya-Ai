# 🏥 MediScan AI — Disease Risk Predictor Web App

A beautiful Flask web application that connects **7 trained ML models** to a
professional symptom-based disease risk prediction interface.

---

## 📁 Project Structure

```
disease_predictor/
├── app.py                  ← Flask backend (main server)
├── templates/
│   └── index.html          ← Frontend UI (all-in-one HTML)
├── models/
│   ├── breast_model.pkl
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── liver_model.pkl
│   ├── pcos_model.pkl
│   ├── prostate_model.pkl
│   └── thyroid_model.pkl
└── README.md
```

---

## ⚡ Step-by-Step Setup

### Step 1 — Install Python & Flask

Make sure you have Python 3.8+ installed. Then:

```bash
pip install flask numpy scikit-learn
```

### Step 2 — Run the server

```bash
cd disease_predictor
python app.py
```

You will see:
```
🏥 Disease Risk Predictor starting...
   Models loaded: ['pcos', 'thyroid', 'liver', 'breast', 'heart', 'prostate', 'diabetes']
 * Running on http://127.0.0.1:5000
```

### Step 3 — Open in browser

Go to: **http://localhost:5000**

---

## 🔁 How It Works (Code Flow)

```
Browser (HTML form)
        │
        │  POST /predict  (form data)
        ▼
   app.py — Flask route
        │
        ├── symptoms_to_features()
        │     Converts ~40 symptom answers → 150+ feature values
        │     that each model expects
        │
        ├── For each model (breast, diabetes, heart, liver, pcos, prostate, thyroid):
        │     Load .pkl  →  build feature array  →  predict_proba()
        │     Apply gender weight (e.g. prostate=0 for females)
        │
        └── Return JSON with per-disease risk % + overall score
        
Browser renders animated result cards
```

---

## 🌐 API Endpoint

**POST `/predict`** — Form data fields:

| Field | Type | Description |
|-------|------|-------------|
| name | string | Patient name |
| gender | `female` / `male` | Biological sex |
| age | int | 18–100 |
| height | int | cm |
| weight | float | kg |
| waist | int | inches |
| smoking | 0/1 | Smoker? |
| alcohol | 0–3 | Intake level |
| activity | 0–3 | Exercise level |
| bp_high | 0–3 | Blood pressure |
| chest_pain | 0–3 | Chest pain severity |
| fatigue | 0–3 | Fatigue level |
| ... | | (and 30+ more symptom fields) |

**Response JSON:**
```json
{
  "success": true,
  "patient": "Priya Sharma",
  "gender": "female",
  "overall": 43.7,
  "o_level": "MODERATE",
  "results": [
    { "name": "PCOS", "risk": 95.9, "level": "CRITICAL", "color": "#7f1d1d" },
    { "name": "Heart Disease", "risk": 74.6, "level": "CRITICAL" },
    ...
  ]
}
```

---

## 🧬 Gender-Aware Risk Scoring

| Disease | Female Weight | Male Weight |
|---------|:---:|:---:|
| Breast Cancer | 1.0 | 0.0 |
| PCOS | 0.9 | 0.0 |
| Thyroid | 0.9 | 0.5 |
| Heart | 0.7 | 0.9 |
| Prostate | 0.0 | 1.0 |
| Diabetes | 0.6 | 0.7 |
| Liver | 0.6 | 0.8 |

---

## ⚠️ Disclaimer

This tool is for educational purposes only. Not a substitute for professional medical advice.
