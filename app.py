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

# Adding visualization to my app
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load saved model and scaler
scaler = pickle.load(open("scaler.sav", "rb"))
model_scaled = pickle.load(open("trained_scaled_model.sav", "rb"))

# File uploader
st.title("House Price Prediction App")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    # Visualizations
    st.subheader("Scatter Plot: Features vs Target")
    feature_columns = ["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
                       "X4 number of convenience stores", "X5 latitude", "X6 longitude"]
    target_column = "Y house price of unit area"

    if all(col in data.columns for col in feature_columns + [target_column]):
        selected_feature = st.selectbox("Select a feature to plot against the target:", feature_columns)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=selected_feature, y=target_column, ax=ax, alpha=0.7, color="blue")
        ax.set_title(f"{selected_feature} vs {target_column}")
        st.pyplot(fig)

        # Histogram
        st.subheader("Histogram of Target Variable")
        fig, ax = plt.subplots()
        ax.hist(data[target_column], bins=20, color="green", edgecolor="black", alpha=0.7)
        ax.set_title("Distribution of House Prices")
        st.pyplot(fig)
    else:
        st.error("Dataset must contain all the required columns for visualization.")
else:
    st.info("Please upload a dataset to view visualizations.")

