# 🏥 MediScan AI — AI Disease Risk Prediction System

🔗 **Live Application:**  
https://aarogya-ai-09ha.onrender.com/

MediScan AI is an AI-powered healthcare web application designed to estimate potential disease risks based on a person’s health data, lifestyle habits, symptoms, and medical history.

I developed this project to explore how Artificial Intelligence can assist in early disease risk estimation using structured health data. The platform integrates multiple trained machine learning models and computer vision into a single intelligent diagnostic system.

The system combines multiple machine learning models and image classification to provide a comprehensive health risk overview.

---

# 🚀 Project Overview

The goal of this project was to build a unified disease prediction platform rather than creating separate tools for each disease.

The system collects health information from users through an interactive medical questionnaire and processes this data using multiple trained machine learning models.

Each model focuses on predicting the risk of a specific disease based on the provided inputs.

After analyzing the data, the system generates:

• Individual disease risk percentages  
• Severity levels  
• An overall health risk score  

The results are displayed in an interactive health dashboard.

---

# 🧠 How the System Works

The platform collects structured medical information from users through a guided multi-section interface and analyzes it using trained machine learning models.

The analysis includes:

• Symptom-based disease prediction  
• Lifestyle risk analysis  
• Gender-aware risk scoring  
• Lung disease detection using X-ray image classification  

The system then generates risk percentages and severity levels for multiple diseases.

The prediction pipeline works as follows:

User enters health data  
↓  
Flask backend receives form data  
↓  
Symptoms are converted into feature vectors  
↓  
Each ML model calculates probability using `predict_proba()`  
↓  
Gender-based weights adjust predictions  
↓  
System calculates overall health risk score  
↓  
Frontend displays animated risk cards

---

# 🧠 Diseases Predicted

The system currently predicts risk for the following diseases:

• PCOS  
• Thyroid Disorders  
• Liver Disease  
• Breast Cancer  
• Heart Disease  
• Prostate Disease  
• Diabetes  

Additionally, the platform includes:

🫁 **Lung Disease Detection using X-ray Image Classification**

---

# 1️⃣ Personal Identity

This section collects basic demographic and biological information that significantly affects disease risk.

### Information Collected

• Full Name  
• Age  
• Biological Sex  

### Why This Matters

Age and biological sex influence the probability of many diseases.

Examples:

Female-specific risks: PCOS, Breast Cancer, Hormonal disorders  
Male-specific risks: Prostate disease  
Age-related risks: Heart disease, diabetes, thyroid disorders  

This information helps the AI system apply **gender-aware weighting when calculating risks**.

---

# 2️⃣ Body Measurements

This section gathers physical health indicators used by many medical prediction models.

### Physical Metrics

• Height  
• Weight  
• Waist circumference  

From these values the system calculates **Body Mass Index (BMI)**.

BMI helps estimate risks for:

• Diabetes  
• Heart disease  
• Hormonal imbalance  
• Liver disease  

### Known Medical Conditions

Users can also indicate existing diagnoses such as:

• Hypertension  
• Diabetes  

These conditions significantly influence disease prediction models.

---

# 3️⃣ Symptoms & Medical History

This is the core diagnostic section of the system.

Users report symptoms across several medical categories.

### Lifestyle

• Smoking habits  
• Alcohol consumption  
• Physical activity  
• Diet quality  
• Stress levels  
• Sleep duration  

Lifestyle factors strongly influence risks for **heart disease, liver disease, and diabetes**.

---

### Heart & Cardiovascular

• Chest pain  
• Chest pressure  
• Heart palpitations  
• Irregular heartbeat  
• Shortness of breath  
• Leg swelling  

These indicators help estimate the probability of **cardiovascular disease**.

---

### Diabetes Indicators

• Fatigue  
• Weight fluctuations  
• Lifestyle factors  
• Known diabetes diagnosis  

Used by the **diabetes prediction model**.

---

### Liver Health

• Alcohol intake  
• Abdominal discomfort  
• Lifestyle risk factors  

These inputs are analyzed by the **liver disease prediction model**.

---

### Kidney Health

• Reduced urine output  
• Blood in urine  
• Foamy urine  
• Lower back pain  
• Swelling around eyes or legs  

These symptoms help identify **possible kidney function issues**.

---

### Hormonal Health (PCOS)

• Menstrual irregularities  
• Weight gain  
• Hormonal imbalance indicators  

These are used by the **PCOS prediction model**.

---

### Breast Health

• Breast discomfort  
• Family history indicators  

These factors contribute to **breast cancer risk prediction**.

---

### Thyroid Indicators

• Fatigue  
• Metabolism-related symptoms  
• Hormonal signals  

These inputs help evaluate **thyroid disorder risk**.

---

# 🫁 Lung Disease Detection (Image Classification)

In addition to symptom-based prediction, MediScan AI also supports **lung disease detection from chest X-ray images**.

Users can upload an X-ray image, and the system analyzes it using deep learning based image classification.

The model detects patterns related to:

• Pneumonia  
• Normal lungs  
• Other lung abnormalities depending on the dataset.

This adds **computer vision capability** to the medical prediction system.

---

# 📡 Example API Response

After processing the inputs, the backend returns prediction results in JSON format.

```json
{
  "success": true,
  "patient": "Priya Sharma",
  "gender": "female",
  "overall": 43.7,
  "o_level": "MODERATE",
  "results": [
    {
      "name": "PCOS",
      "risk": 95.9,
      "level": "CRITICAL"
    },
    {
      "name": "Heart Disease",
      "risk": 74.6,
      "level": "CRITICAL"
    },
    {
      "name": "Diabetes",
      "risk": 21.4,
      "level": "LOW"
    }
  ]
}
```

The frontend converts this JSON response into visual risk cards.

---

# 🧬 Gender-Aware Risk Scoring

Certain diseases affect genders differently. To make predictions more realistic, the system applies gender-based weighting.

| Disease | Female Weight | Male Weight |
|--------|--------|--------|
| Breast Cancer | 1.0 | 0.0 |
| PCOS | 0.9 | 0.0 |
| Thyroid | 0.9 | 0.5 |
| Heart Disease | 0.7 | 0.9 |
| Prostate Disease | 0.0 | 1.0 |
| Diabetes | 0.6 | 0.7 |
| Liver Disease | 0.6 | 0.8 |

These weights adjust prediction probabilities to reflect realistic medical patterns.

---

# 🏗 Project Structure

```
disease_predictor/
│
├── app.py
│   Flask backend server handling prediction logic
│
├── templates/
│   └── index.html
│   Frontend user interface
│
├── models/
│   ├── breast_model.pkl
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── liver_model.pkl
│   ├── pcos_model.pkl
│   ├── prostate_model.pkl
│   └── thyroid_model.pkl
│
└── README.md
```

Each `.pkl` file represents a trained machine learning model used for disease prediction.

---

# 💻 Technology Stack

Backend  
Python  
Flask  
NumPy  
Scikit-learn  

Machine Learning  
Multiple classification models  
Probability-based predictions (`predict_proba`)  
Feature engineering for symptom mapping  

Computer Vision  
Deep learning based lung X-ray image classification  

Frontend  
HTML  
CSS  
JavaScript  

Deployment  
Render Cloud Platform

---

# 🎯 Key Features

• Predicts multiple diseases simultaneously  
• Machine learning powered health predictions  
• Symptom-based medical risk analysis  
• Gender-aware risk scoring system  
• Lung disease detection from X-ray images  
• Interactive health dashboard  
• Cloud deployed AI healthcare application  

---

# ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only.

It should not be used as a medical diagnosis tool. Always consult a qualified healthcare professional for medical advice.
