# Fraud Risk Analyzer 🛡️

A data-driven solution to analyze and predict business fraud risks using transaction data. This project combines statistical analysis, visualizations, and a machine learning model deployed via a Flask app.

## 🔍 Overview

This project uses a synthetic dataset (`fraud_summary.csv`) that aggregates fraud-related features by company. The goal is to classify companies into **low**, **medium**, or **high** fraud risk using:

- Exploratory Data Analysis (EDA)
- Rule-based flagging
- Machine Learning (Random Forest)
- Flask web application for interactive predictions

## 📁 Project Structure

fraud-risk-analyzer/
│
├── fraud_app/ # Flask application
│ ├── app.py # Main Flask backend
│ └── templates/ # HTML templates for frontend
│
├── plots/ # Visualizations (confusion matrix, boxplots, etc.)
│
├── fraud_summary.csv # Input data
├── company_features.csv # Features used for prediction
├── fraud_model.pkl # Trained model
├── main.py # Model training + EDA script
└── README.md # You're here!


## 📊 Features

- Classification model using `RandomForestClassifier`
- Confusion matrix and performance metrics
- Visualizations of feature distributions and fraud patterns
- Web interface to predict risk level of a company by `company_id`

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Pratheekns/fraud-risk-analyzer.git
cd fraud-risk-analyzer
