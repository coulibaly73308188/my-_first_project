import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.sav", "rb"))

# Title and description
st.title("House Price Prediction App")
st.write("""
This app predicts the **house price of unit area** based on several features:
- Transaction date
- House age
- Distance to the nearest MRT station
- Number of convenience stores
- Latitude
- Longitude
""")

# Input features
X1 = st.number_input("Transaction Date", value=2013.5)
X2 = st.number_input("House Age", value=20.0)
X3 = st.number_input("Distance to MRT Station (meters)", value=300.0)
X4 = st.number_input("Number of Convenience Stores", value=5)
X5 = st.number_input("Latitude", value=24.97)
X6 = st.number_input("Longitude", value=121.54)

# Make prediction
if st.button("Predict"):
    scaled_features = np.array([[X1, X2, X3, X4, X5, X6]])
    prediction = model.predict(scaled_features)
    st.success(f"The predicted house price is: {prediction[0]:.2f}")

