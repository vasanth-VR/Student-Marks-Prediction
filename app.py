import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/student_scores.csv")

# -----------------------------
# Load Model and Metrics
# -----------------------------
model = joblib.load("models/model.pkl")
metrics = joblib.load("models/metrics.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Marks Prediction")
st.write(
    "Predict student exam marks based on study hours using a Linear Regression model."
)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
    )

    if st.button("Predict Marks"):
        input_df = pd.DataFrame({"Hours": [hours]})
        prediction = model.predict(input_df)

        st.success(f"Predicted Marks: {prediction[0]:.2f}")

with col2:
    st.metric("R² Score", f"{metrics['r2']:.2f}")
    st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")

# -----------------------------
# Graph
# -----------------------------
st.subheader("Dataset Visualization")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(data["Hours"], data["Marks"], label="Actual Data")

predicted_line = model.predict(data[["Hours"]])

ax.plot(
    data["Hours"],
    predicted_line,
    label="Regression Line"
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset")

st.dataframe(data)