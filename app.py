import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL AND SCALER
# =========================

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# =========================
# TITLE
# =========================

st.title("Medical Insurance Cost Prediction")

st.write("Predict insurance charges using KNN Regression")

# =========================
# USER INPUTS
# =========================

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Select Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "Enter BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# =========================
# ENCODING
# =========================

# Male = 1, Female = 0

if sex == "Male":
    sex = 1
else:
    sex = 0

# Yes = 1, No = 0

if smoker == "Yes":
    smoker = 1
else:
    smoker = 0

# Region Encoding

region_dict = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region = region_dict[region]

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Insurance Cost"):

    # Input data

    input_data = np.array([
        [age, sex, bmi, children, smoker, region]
    ])

    # Scaling

    input_data = scaler.transform(input_data)

    # Prediction

    prediction = model.predict(input_data)

    # Display Result

    st.success(
        f"Predicted Insurance Cost: ${prediction[0]:.2f}"
    )
