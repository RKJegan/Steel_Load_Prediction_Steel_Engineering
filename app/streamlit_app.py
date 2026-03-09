import os
import streamlit as st
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "best_model.pkl"))
ENCODER_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "label_encoder.pkl"))

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title("Steel Structural Strength Predictor")

steel_grade_name = st.selectbox("Steel Grade", list(label_encoder.classes_))
yield_strength = st.number_input("Yield Strength (MPa)", min_value=0.0, value=350.0)
area = st.number_input("Cross Section Area (mm²)", min_value=0.0, value=300.0)
load = st.number_input("Load Applied (N)", min_value=0.0, value=50000.0)
length = st.number_input("Length (m)", min_value=0.0, value=3.0)
temp = st.number_input("Temperature (°C)", value=25.0)

if st.button("Predict Failure Load"):
    steel_grade = int(label_encoder.transform([steel_grade_name])[0])

    stress = load / area if area != 0 else 0
    slender = length / area if area != 0 else 0
    load_factor = load / yield_strength if yield_strength != 0 else 0
    safety = yield_strength / stress if stress != 0 else 0

    features = np.array([[
        steel_grade,
        yield_strength,
        area,
        load,
        length,
        temp,
        stress,
        slender,
        load_factor,
        safety
    ]])

    predicted_failure_load = model.predict(features)[0]

    st.success(f"Estimated Maximum Load Capacity: {predicted_failure_load:.2f} N")
    st.write(f"Selected steel grade: {steel_grade_name}")
    st.write(f"Applied Load: {load:.2f} N")

    margin = predicted_failure_load - load

    if margin > 0:
        st.info(f"Remaining Load Capacity: {margin:.2f} N")
        st.success("Status: Safe zone according to the model")
    elif abs(margin) <= 1000:
        st.warning("Status: Near failure zone according to the model")
    else:
        st.error(f"Overload by: {abs(margin):.2f} N")
        st.error("Status: Unsafe / possible failure risk according to the model")
