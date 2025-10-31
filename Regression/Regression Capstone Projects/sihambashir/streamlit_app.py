import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ----- 1. Load saved artifacts -----
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')  # dict: {column_name: encoder object}

# ----- 2. App title -----
st.set_page_config(page_title="Health Score Predictor", layout="centered")
st.title('🏥 Health Score Predictor')
st.write("ادخل بيانات المستخدم أدناه للتنبؤ بالـ Health Score.")

# ----- 3. Collect user input dynamically -----
user_input = {}

# Categorical features from encoders
for col, enc in encoders.items():
    options = enc.classes_.tolist()
    user_input[col] = st.selectbox(f"{col}", options)

# Numeric features from scaler
numeric_cols = scaler.feature_names_in_
for col in numeric_cols:
    if col not in user_input:
        user_input[col] = st.number_input(f"{col}", value=0.0, step=0.1)

# ----- 4. Prepare DataFrame -----
input_df = pd.DataFrame([user_input])

# Apply encoders
for col, enc in encoders.items():
    input_df[col] = enc.transform(input_df[col])

# Apply scaler
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ----- 5. Make prediction -----
if st.button("Predict Health Score"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Health Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----- 6. Optional: Show processed input for verification -----
with st.expander("Show processed input"):
    st.dataframe(input_df)
