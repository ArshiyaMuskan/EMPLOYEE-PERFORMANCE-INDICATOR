import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# PATH HANDLING
# ------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide"
)

# ------------------------------
# LOAD MODEL & DATA
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

# ------------------------------
# TITLE
# ------------------------------
st.title("📊 Employee Performance Predictor")
st.markdown("AI-powered HR Analytics Dashboard")

# ------------------------------
# SIDEBAR INPUT
# ------------------------------
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 22, 60, 30)
experience = st.sidebar.slider("Experience (Years)", 1, 20, 5)
salary = st.sidebar.slider("Salary", 20000, 150000, 50000)
training = st.sidebar.slider("Training Hours", 10, 100, 40)
projects = st.sidebar.slider("Projects Completed", 1, 20, 8)

# ------------------------------
# INPUT DATAFRAME
# ------------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Experience': [experience],
    'Salary': [salary],
    'TrainingHours': [training],
    'ProjectsCompleted': [projects]
})

# ------------------------------
# PREDICTION
# ------------------------------
prediction = model.predict(input_data)[0]

labels = {0: "Low", 1: "Medium", 2: "High"}
result = labels[prediction]

# ------------------------------
# MAIN DISPLAY
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Employee Input")
    st.write(input_data)

with col2:
    st.subheader("🎯 Prediction Result")

    if result == "High":
        st.success(f"Performance: {result} 🚀")
    elif result == "Medium":
        st.warning(f"Performance: {result} ⚖️")
    else:
        st.error(f"Performance: {result} ⚠️")

# ------------------------------
# TABS
# ------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Data Insights",
    "📈 Feature Importance",
    "🔍 Compare Employee"
])

# ------------------------------
# TAB 1: DATA INSIGHTS
# ------------------------------
with tab1:
    st.subheader("Performance Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x='Performance', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

# ------------------------------
# TAB 2: FEATURE IMPORTANCE
# ------------------------------
with tab2:
    st.subheader("Model Feature Importance")

    importance = model.feature_importances_
    features = ['Age', 'Experience', 'Salary', 'TrainingHours', 'ProjectsCompleted']

    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    fig3, ax3 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax3)
    st.pyplot(fig3)

# ------------------------------
# TAB 3: COMPARE EMPLOYEE (FIXED)
# ------------------------------
with tab3:
    st.subheader("Compare with Average Employee")

    features = ['Age', 'Experience', 'Salary', 'TrainingHours', 'ProjectsCompleted']

    avg_values = data[features].mean()

    comparison = pd.DataFrame({
        'Feature': features,
        'Employee': input_data.iloc[0].values,
        'Average': avg_values.values
    })

    st.write(comparison)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Built with ❤️ for Data Science Portfolio")