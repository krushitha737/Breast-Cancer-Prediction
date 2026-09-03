# 🎗️ Breast Cancer Prediction

A Machine Learning based breast cancer classification project using
Logistic Regression and an interactive Streamlit dashboard.

## 📌 Project Overview

This project uses the Breast Cancer Wisconsin Diagnostic dataset
to classify tumor measurements into two classes:

Malignant
Benign

Several Machine Learning algorithms were evaluated, and Logistic
Regression was selected as the final model based on its performance
and simplicity.

## 🎯 Objectives

Perform exploratory data analysis on breast cancer data
Preprocess and scale numerical features
Train multiple Machine Learning models
Compare model performance
Select the best-performing model
Build an interactive prediction dashboard using Streamlit

## 📊 Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset
available through Scikit-learn.

Dataset Characteristics

Total Samples: 569
Total Features: 30
Target Classes: Malignant and Benign
Features: Numerical tumor measurements

## 🛠️ Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit
Joblib
Jupyter Notebook

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Model Evaluation
   ↓
Final Model Selection
   ↓
Streamlit Dashboard

## 🤖 Models Compared

The following models were evaluated:
| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   98.25% |    98.61% | 98.61% |   98.61% |  99.54% |
| SVM                 |   98.25% |    98.61% | 98.61% |   98.61% |  99.50% |
| KNN                 |   95.61% |    95.89% | 97.22% |   96.55% |  97.88% |
| Random Forest       |   95.61% |    95.89% | 97.22% |   96.55% |  99.37% |
| Decision Tree       |   91.23% |    95.59% | 90.28% |   92.86% |  91.57% |

## 🏆 Final Model

Logistic Regression was selected as the final model.

Performance
Accuracy: 98.25%
Precision: 98.61%
Recall: 98.61%
F1 Score: 98.61%
ROC-AUC: 99.54%
📈 Dashboard Features

### The Streamlit dashboard provides:

Model performance metrics
Dataset statistics
Diagnosis distribution
Feature correlation analysis
Model comparison
Accuracy comparison chart
Confusion matrix
ROC curve
Interactive prediction
Benign and malignant demo samples
Prediction probabilities

## ▶️ How to Run
1. Clone the Repository
git clone <your-repository-url>
cd Breast-Cancer-Prediction
2. Install Dependencies
pip install -r requirements.txt
3. Run the Streamlit Dashboard
streamlit run src/app.py

The application will open in your browser at:

http://localhost:8501
📁 Project Structure
Breast-Cancer-Prediction/
│
├── data/
│
├── models/
│   ├── breast_cancer_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── breast_cancer_prediction.ipynb
│
├── src/
│   └── app.py
│
├── README.md
└── requirements.txt

📌 Key Highlights
Implemented complete Machine Learning workflow from data exploration to deployment
Compared five different classification algorithms
Achieved 98.25% test accuracy using Logistic Regression
Evaluated the model using Precision, Recall, F1 Score, and ROC-AUC
Developed an interactive Streamlit dashboard for predictions and visualization
Saved the trained model and scaler using Joblib

⚠️ Disclaimer

This project is developed for educational and demonstration purposes
only. It is not a clinical diagnostic system and should not be used
as a substitute for professional medical advice or diagnosis.
