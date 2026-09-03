import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "breast_cancer_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

data = load_breast_cancer(as_frame=True)
df = data.frame


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎗️ Breast Cancer Prediction Dashboard")

st.write(
    "Machine Learning based breast cancer classification "
    "using Logistic Regression."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    This project uses Machine Learning to classify
    breast cancer cases as Malignant or Benign.

    Model:
    Logistic Regression

    Dataset:
    Breast Cancer Wisconsin Diagnostic Dataset
    """
)

st.sidebar.info(
    "⚠️ This application is intended for educational "
    "and demonstration purposes only. It is not a "
    "clinical diagnostic tool."
)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "98.25%")
col2.metric("Precision", "98.61%")
col3.metric("Recall", "98.61%")
col4.metric("F1 Score", "98.61%")


# ==========================================
# DATASET ANALYTICS
# ==========================================

st.header("📊 Dataset Analytics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Samples", df.shape[0])
col2.metric("Total Features", df.shape[1] - 1)

col3.metric(
    "Malignant Cases",
    int((df["target"] == 0).sum())
)

col4.metric(
    "Benign Cases",
    int((df["target"] == 1).sum())
)


# ==========================================
# DIAGNOSIS DISTRIBUTION
# ==========================================

st.subheader("Diagnosis Distribution")

diagnosis_counts = df["target"].value_counts()

diagnosis_df = pd.DataFrame({
    "Diagnosis": ["Malignant", "Benign"],
    "Cases": [
        diagnosis_counts.get(0, 0),
        diagnosis_counts.get(1, 0)
    ]
})

st.bar_chart(
    diagnosis_df.set_index("Diagnosis")
)


# ==========================================
# FEATURE CORRELATION
# ==========================================

st.subheader("🔗 Feature Correlation")

correlation = df.drop("target", axis=1).corr()

st.dataframe(
    correlation,
    use_container_width=True
)


# ==========================================
# MODEL COMPARISON
# ==========================================

st.header("🤖 Model Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Random Forest",
        "Decision Tree"
    ],

    "Accuracy": [
        0.982456,
        0.982456,
        0.956140,
        0.956140,
        0.912281
    ],

    "Precision": [
        0.986111,
        0.986111,
        0.958904,
        0.958904,
        0.955882
    ],

    "Recall": [
        0.986111,
        0.986111,
        0.972222,
        0.972222,
        0.902778
    ],

    "F1 Score": [
        0.986111,
        0.986111,
        0.965517,
        0.965517,
        0.928571
    ],

    "ROC-AUC": [
        0.995370,
        0.995040,
        0.978836,
        0.993717,
        0.915675
    ]
})


# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

st.subheader("Model Performance Comparison")

st.dataframe(
    comparison_data,
    use_container_width=True
)


# ==========================================
# MODEL ACCURACY CHART
# ==========================================

st.subheader("Model Accuracy Comparison")

accuracy_chart = comparison_data.set_index("Model")[["Accuracy"]]

st.bar_chart(
    accuracy_chart
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

st.header("📌 Confusion Matrix")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Malignant", "Actual Benign"],
    columns=["Predicted Malignant", "Predicted Benign"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ==========================================
# ROC CURVE
# ==========================================


st.header("📈 ROC Curve")

# Get prediction probabilities
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probability)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Create ROC plot
fig, ax = plt.subplots(figsize=(8, 6))

# ROC curve
ax.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

# Random classifier line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.50)"
)

# Labels
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")

# Axis limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Grid and legend
ax.grid(True)
ax.legend(loc="lower right")

# Display
st.pyplot(fig)

# Display AUC
st.metric(
    "ROC-AUC Score",
    f"{roc_auc:.4f}"
)

# ==========================================
# PREDICTION SECTION
# ==========================================

st.header("🔬 Breast Cancer Prediction")

st.write(
    "Enter the tumor measurements below or use one of "
    "the sample cases for demonstration."
)


# ==========================================
# SAMPLE CASES
# ==========================================

st.subheader("🧪 Demo Samples")

sample1, sample2 = st.columns(2)

if sample1.button("🟢 Load Benign Sample"):

    benign_sample = data.data[data.target == 1].iloc[0]

    st.session_state["sample_values"] = benign_sample.tolist()


if sample2.button("🔴 Load Malignant Sample"):

    malignant_sample = data.data[data.target == 0].iloc[0]

    st.session_state["sample_values"] = malignant_sample.tolist()


# ==========================================
# DEFAULT SAMPLE
# ==========================================

if "sample_values" not in st.session_state:

    st.session_state["sample_values"] = data.data.iloc[0].tolist()


# ==========================================
# USER INPUTS
# ==========================================

feature_names = data.feature_names

input_values = []

cols = st.columns(3)

for i, feature in enumerate(feature_names):

    with cols[i % 3]:

        value = st.number_input(
            feature,
            value=float(
                st.session_state["sample_values"][i]
            ),
            format="%.4f",
            key=f"feature_{i}"
        )

        input_values.append(value)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Diagnosis"):

    input_data = np.array(input_values).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]


    # ==========================================
    # RESULT
    # ==========================================

    st.subheader("📋 Prediction Result")

    if prediction == 0:

        st.error(
            "⚠️ Prediction: Malignant"
        )

    else:

        st.success(
            "✅ Prediction: Benign"
        )


    # ==========================================
    # PROBABILITY METRICS
    # ==========================================

    col1, col2 = st.columns(2)

    col1.metric(
        "Malignant Probability",
        f"{probability[0] * 100:.2f}%"
    )

    col2.metric(
        "Benign Probability",
        f"{probability[1] * 100:.2f}%"
    )


    # ==========================================
    # PROBABILITY CHART
    # ==========================================

    probability_df = pd.DataFrame(
        {
            "Probability": [
                probability[0],
                probability[1]
            ]
        },
        index=[
            "Malignant",
            "Benign"
        ]
    )

    st.subheader("📊 Prediction Probability")

    st.bar_chart(
        probability_df
    )


# ==========================================
# DISCLAIMER
# ==========================================

st.divider()

st.caption(
    "⚠️ Disclaimer: This project is developed for "
    "educational and demonstration purposes only. "
    "Predictions should not be used as a substitute "
    "for professional medical advice or diagnosis."
)