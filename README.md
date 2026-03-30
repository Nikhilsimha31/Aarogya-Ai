# 🏥 MediScan AI — AI Disease Risk Prediction System

🔗 **Live Application:**  
https://aarogya-ai-09ha.onrender.com/

MediScan AI is a machine learning powered healthcare web application that predicts the risk of multiple diseases using a person's symptoms, lifestyle factors, and health indicators.

I developed this project to explore how Artificial Intelligence can assist in early disease risk estimation using structured health data. The platform integrates multiple trained machine learning models into a single intelligent diagnostic system.

---

# 🚀 Project Overview

The goal of this project was to build a unified disease prediction platform rather than creating separate tools for each disease.

The system collects health information from users through an interactive medical questionnaire and processes this data using seven different machine learning models.

Each model focuses on predicting the risk of a specific disease based on the provided inputs.

After analyzing the data, the system generates:

• Individual disease risk percentages  
• Severity levels  
• An overall health risk score  

The results are displayed in an interactive health dashboard.

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

Each disease uses its own trained machine learning model, allowing more specialized predictions.

---

# ⚙️ How the System Works

The prediction pipeline works as follows:

User enters health data  
↓  
Flask backend receives the form data  
↓  
Symptoms are converted into feature vectors  
↓  
Each ML model runs prediction using predict_proba()  
↓  
Gender-based weights adjust prediction probabilities  
↓  
System calculates overall health risk score  
↓  
Frontend displays results using visual risk cards

This architecture allows the system to analyze multiple diseases simultaneously from a single form.

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

The frontend converts this JSON response into visual disease risk cards.

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

These weights adjust the prediction probability to reflect realistic medical patterns.

---

# 🫁 Lung Disease Detection

In addition to symptom-based prediction, the platform also includes lung disease detection using chest X-ray image classification.

Users can upload an X-ray image, and the model analyzes it using deep learning techniques.

The model detects patterns related to:

• Pneumonia  
• Normal lungs  
• Other lung abnormalities depending on the dataset used for training.

This feature integrates computer vision with the machine learning prediction system.

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

Each `.pkl` file represents a trained machine learning model used for predicting disease risk.

---

# 💻 Technology Stack

Backend  
Python  
Flask  
NumPy  
Scikit-learn  

Machine Learning  
Multiple classification models  
Probability-based predictions (predict_proba)  
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

• Predicts risk for multiple diseases simultaneously  
• Uses multiple machine learning models  
• Gender-aware risk scoring system  
• Symptom-based medical analysis  
• Lung disease detection from X-ray images  
• Interactive health dashboard  
• Cloud deployed AI healthcare application  

---

# ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only.

It should not be used as a medical diagnosis tool. Always consult a qualified healthcare professional for medical advice.
