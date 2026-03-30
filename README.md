**🏥 MediScan AI — Intelligent Disease Risk Predictor**

🔗 Live Application:
https://aarogya-ai-09ha.onrender.com/

MediScan AI is an AI-powered healthcare web application designed to estimate potential disease risks based on a person’s health data, lifestyle habits, symptoms, and medical history.

The system combines multiple machine learning models and image classification to provide a comprehensive health risk overview.

🧠 How the System Works

The platform collects structured medical information from users through a guided multi-section interface and analyzes it using trained machine learning models.

The analysis includes:

• Symptom-based disease prediction
• Lifestyle risk analysis
• Gender-aware risk scoring
• Lung disease detection using X-ray image classification

The system then generates risk percentages and severity levels for multiple diseases.

1️⃣ Personal Identity

This section collects basic demographic and biological information that significantly affects disease risk.

Information Collected

• Full Name
• Age
• Biological Sex

Why This Matters

Age and biological sex influence the probability of many diseases.

Examples:

Female-specific risks: PCOS, Breast Cancer, Hormonal disorders
Male-specific risks: Prostate disease
Age-related risks: Heart disease, diabetes, thyroid disorders

This information helps the AI system apply gender-aware weighting when calculating risks.

2️⃣ Body Measurements

This section gathers physical health indicators used by many medical prediction models.

Physical Metrics

• Height
• Weight
• Waist circumference

From these values the system calculates Body Mass Index (BMI).

BMI helps estimate risks for:

Diabetes
Heart disease
Hormonal imbalance
Liver disease
Known Medical Conditions

Users can also indicate existing diagnoses such as:

• Hypertension
• Diabetes

These conditions significantly influence disease prediction models.

3️⃣ Symptoms & Medical History

This is the core diagnostic section of the system.

Users report symptoms across several medical categories.

Lifestyle

• Smoking habits
• Alcohol consumption
• Physical activity
• Diet quality
• Stress levels
• Sleep duration

Lifestyle factors strongly influence risks for heart disease, liver disease, and diabetes.

Heart & Cardiovascular

• Chest pain
• Chest pressure
• Heart palpitations
• Irregular heartbeat
• Shortness of breath
• Leg swelling

These indicators help estimate the probability of cardiovascular disease.

Diabetes Indicators

• Fatigue
• Weight fluctuations
• Lifestyle factors
• Known diabetes diagnosis

Used by the diabetes prediction model.

Liver Health

• Alcohol intake
• Abdominal discomfort
• Lifestyle risk factors

These inputs are analyzed by the liver disease prediction model.

Kidney Health

• Reduced urine output
• Blood in urine
• Foamy urine
• Lower back pain
• Swelling around eyes or legs

These symptoms help identify possible kidney function issues.

Hormonal Health (PCOS)

• Menstrual irregularities
• Weight gain
• Hormonal imbalance indicators

These are used by the PCOS prediction model.

Breast Health

• Breast discomfort
• Family history indicators

These factors contribute to breast cancer risk prediction.

Thyroid Indicators

• Fatigue
• Metabolism-related symptoms
• Hormonal signals

These inputs help evaluate thyroid disorder risk.

🫁 Lung Disease Detection (Image Classification)

In addition to symptom-based prediction, MediScan AI also includes lung disease detection using chest X-ray images.

Users can upload a medical image, and the system analyzes it using deep learning image classification.

The model can detect patterns related to:

• Pneumonia
• Normal lungs
• Other lung abnormalities depending on training data.

This feature integrates computer vision with medical prediction models.

🧬 Diseases Predicted

The platform currently evaluates risk for:

• PCOS
• Thyroid Disorders
• Liver Disease
• Breast Cancer
• Heart Disease
• Prostate Disease
• Diabetes

Plus:

🫁 Lung disease detection via X-ray image analysis

⚙️ Technology Stack
Backend

Python
Flask
NumPy
Scikit-learn

Machine Learning

Multiple trained classification models
Probability-based predictions

Computer Vision

Deep learning image classification for lung scans

Frontend

HTML
CSS
JavaScript

Deployment

Render Cloud Platform**
