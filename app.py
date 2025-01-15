import streamlit as st
import pickle
import numpy as np

# Load the saved scaler and model
scaler = pickle.load(open("scaler.sav", "rb"))
model = pickle.load(open("trained_scaled_model.sav", "rb"))

# Title and Description
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

# User Inputs for Prediction
transaction_date = st.number_input("Transaction Date (e.g., 2013.5)", value=2013.5)
house_age = st.number_input("House Age (years)", value=20.0)
distance_to_mrt = st.number_input("Distance to MRT Station (meters)", value=500.0)
number_of_stores = st.number_input("Number of Convenience Stores", value=5)
latitude = st.number_input("Latitude", value=24.97)
longitude = st.number_input("Longitude", value=121.54)

# Make Prediction
if st.button("Predict"):
    # Prepare the input as a 2D array
    new_data = [[transaction_date, house_age, distance_to_mrt, number_of_stores, latitude, longitude]]
    
    # Scale the input data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    st.success(f"The predicted house price is: {prediction[0]:.2f}")
