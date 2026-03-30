🏥 MediScan AI — Disease Risk Predictor

🔗 Live App: https://aarogya-ai-09ha.onrender.com/

MediScan AI is a machine learning powered healthcare web application that predicts potential disease risks based on a person's symptoms, lifestyle, and health indicators.

I built this project to explore real-world applications of machine learning in healthcare, combining Flask, Python, and multiple trained ML models into a single intelligent system.

🚀 How I Developed This Project

The goal of this project was to create a single platform capable of analyzing multiple diseases simultaneously instead of building separate predictors.

To achieve this, I designed a Flask-based backend that connects seven different machine learning models trained for different diseases.

These models analyze user inputs such as:

Age
Gender
Height & Weight
Lifestyle habits (smoking, alcohol, activity level)
Symptoms (fatigue, chest pain, blood pressure, etc.)

The system converts these inputs into feature vectors that match the format expected by each trained model.

Each model then calculates a probability score using predict_proba(), and the backend aggregates these results into a disease risk dashboard.

🧠 Diseases Predicted

The application currently predicts risk for 7 diseases:

• PCOS
• Thyroid Disorders
• Liver Disease
• Breast Cancer
• Heart Disease
• Prostate Cancer
• Diabetes

Each disease is predicted using its own trained ML model, allowing more specialized predictions.

⚙️ Technology Stack
Backend
Python
Flask
NumPy
Scikit-learn
Machine Learning
Multiple trained classification models
Probability-based prediction (predict_proba)
Feature engineering for symptom mapping
Frontend
HTML
CSS
JavaScript
Interactive animated result cards
Deployment
Render Cloud Platform
🧬 Smart Gender-Aware Risk Scoring

Some diseases affect genders differently.
To make predictions more realistic, I implemented gender-based weighting.

For example:

Disease	Female	Male
Breast Cancer	High relevance	Ignored
PCOS	High relevance	Ignored
Thyroid	Higher	Moderate
Heart	Moderate	Higher
Prostate	Ignored	High

This ensures the overall risk score reflects realistic medical probabilities.

📊 How the Prediction Pipeline Works

User enters health data →

Flask backend processes the form →

Symptoms are converted into model features →

Each ML model generates a probability score →

Scores are weighted based on gender →

Results are returned as risk percentages and severity levels

The frontend then displays the results using visual risk cards and an overall health risk score.

🎯 Key Features

✔ Predicts 7 diseases simultaneously
✔ Gender-aware risk calculation
✔ Interactive UI with animated results
✔ Works directly from symptom inputs
✔ Cloud deployed and publicly accessible

⚠️ Disclaimer

This project is designed for educational and demonstration purposes only.

It should not be used as a medical diagnosis tool.
Always consult qualified healthcare professionals for medical advice.
