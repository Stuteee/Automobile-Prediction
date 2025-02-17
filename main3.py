import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the saved model and PCA
try:
    model = pickle.load(open(r'E:\second year\compute\Automobile\trained_model.pkl', 'rb'))
    pca = joblib.load(r'E:\second year\compute\Automobile\pca_model.pkl')
    st.write("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or PCA: {str(e)}")

# Load the final data
try:
    df_combined = pd.read_csv('final_data.csv')
    st.write("Dataset loaded successfully!")
    st.write("Columns in dataset:", df_combined.columns.tolist())  # Debugging
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    df_combined = pd.DataFrame()  # Create empty DataFrame to avoid crashes

# Streamlit app
def main():
    st.title("Automobile Price Prediction App")

    # Sidebar options
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Select an option", ["View Data", "Make a Prediction"])

    # View Data option
    if option == "View Data":
        st.header("Data")
        if df_combined.empty:
            st.error("Dataset is empty. Check if the file is missing or corrupted.")
        else:
            st.write(df_combined.head(10))

    # Make a Prediction option
    elif option == "Make a Prediction":
        st.header("Make a Prediction")

        if df_combined.empty:
            st.error("Dataset is not loaded properly. Unable to make predictions.")
            return

        # Ensure column names are clean (strip whitespace & lower case)
        df_combined.columns = df_combined.columns.str.strip()

        # Ensure 'make' exists
        if 'make' in df_combined.columns:
            make_options = df_combined['make'].dropna().unique()
        else:
            st.error("Column 'make' not found in dataset.")
            make_options = []

        # User inputs
        num_doors = st.number_input("Number of Doors", min_value=1, max_value=5)
        bore = st.number_input("Bore", min_value=0.0, max_value=5.0)
        stroke = st.number_input("Stroke", min_value=0.0, max_value=5.0)
        horsepower = st.number_input("Horsepower", min_value=0, max_value=500)
        peak_rpm = st.number_input("Peak RPM", min_value=0, max_value=10000)
        city_mpg = st.number_input("City MPG", min_value=1, max_value=100)
        highway_mpg = st.number_input("Highway MPG", min_value=1, max_value=100)
        engine_size = st.number_input("Engine Size", min_value=1, max_value=10000)
        curb_weight = st.number_input("Curb Weight", min_value=500, max_value=10000)

        # Categorical inputs
        make = st.selectbox("Make", make_options)
        fuel_type = st.selectbox("Fuel Type", df_combined['fuel-type'].unique() if 'fuel-type' in df_combined.columns else [])
        aspiration = st.selectbox("Aspiration", df_combined['aspiration'].unique() if 'aspiration' in df_combined.columns else [])
        body_style = st.selectbox("Body Style", df_combined['body-style'].unique() if 'body-style' in df_combined.columns else [])
        drive_wheels = st.selectbox("Drive Wheels", df_combined['drive-wheels'].unique() if 'drive-wheels' in df_combined.columns else [])
        engine_location = st.selectbox("Engine Location", df_combined['engine-location'].unique() if 'engine-location' in df_combined.columns else [])
        engine_type = st.selectbox("Engine Type", df_combined['engine-type'].unique() if 'engine-type' in df_combined.columns else [])
        fuel_system = st.selectbox("Fuel System", df_combined['fuel-system'].unique() if 'fuel-system' in df_combined.columns else [])

        # Input dictionary
        input_values = {
            'num-of-doors': num_doors,
            'bore': bore,
            'stroke': stroke,
            'horsepower': horsepower,
            'peak-rpm': peak_rpm,
            'city-mpg': city_mpg,
            'highway-mpg': highway_mpg,
            'engine-size': engine_size,
            'curb-weight': curb_weight,
            'make': make,
            'fuel-type': fuel_type,
            'aspiration': aspiration,
            'body-style': body_style,
            'drive-wheels': drive_wheels,
            'engine-location': engine_location,
            'engine-type': engine_type,
            'fuel-system': fuel_system
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_values])

        # One-hot encode categorical features
        categorical_cols = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # Align with training data
        missing_cols = set(df_combined.columns) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0  # Add missing columns

        input_df_encoded = input_df_encoded[df_combined.columns]  # Reorder columns

        # Standardize input features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_combined.drop(columns=['price'], errors='ignore'))  # Fit on full dataset
        input_scaled = scaler.transform(input_df_encoded.drop(columns=['price'], errors='ignore'))  # Transform input

        # Apply PCA
        input_pca = pca.transform(input_scaled)

        # Predict price
        prediction = model.predict(input_pca)

        # Display prediction
        st.success(f"Predicted Price: **${prediction[0]:,.2f}**")

if __name__ == '__main__':
    main()
