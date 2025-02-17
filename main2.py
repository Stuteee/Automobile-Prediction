import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and PCA
try:
    model = pickle.load(open(r'E:\second year\compute\Automobile\trained_model.pkl', 'rb'))
    pca = joblib.load("E:\second year\compute\Automobile\trained_model.pkl")  # Load the trained PCA object used during training
    st.write("Models loaded successfully!")
except Exception as e:
    st.write(f"Error loading model or PCA: {e}")

# Load the final data (for reference and encoding)
df_combined = pd.read_csv('final_data.csv')

# Streamlit app
def main():
    st.title("Automobile Price Prediction App")

    # Add a sidebar with options
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Select an option", ["View Data", "Make a Prediction"])

    # View Data option
    if option == "View Data":
        st.header("Data")
        st.write(df_combined.head(10))

    # Make a Prediction option
    elif option == "Make a Prediction":
        st.header("Make a Prediction")
        
        # Create input fields for the user to enter values
        num_doors = st.number_input("Number of Doors")
        bore = st.number_input("Bore")
        stroke = st.number_input("Stroke")
        horsepower = st.number_input("Horsepower")
        peak_rpm = st.number_input("Peak RPM")
        city_mpg = st.number_input("City MPG")
        highway_mpg = st.number_input("Highway MPG")
        engine_size = st.number_input("Engine Size")
        curb_weight = st.number_input("Curb Weight")
        make = st.selectbox("Make", df_combined['make'].unique())
        fuel_type = st.selectbox("Fuel Type", df_combined['fuel-type'].unique())
        aspiration = st.selectbox("Aspiration", df_combined['aspiration'].unique())
        body_style = st.selectbox("Body Style", df_combined['body-style'].unique())
        drive_wheels = st.selectbox("Drive Wheels", df_combined['drive-wheels'].unique())
        engine_location = st.selectbox("Engine Location", df_combined['engine-location'].unique())
        engine_type = st.selectbox("Engine Type", df_combined['engine-type'].unique())
        fuel_system = st.selectbox("Fuel System", df_combined['fuel-system'].unique())

        # Create a dictionary to store the input values
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

        # Create a DataFrame from the input values
        input_df = pd.DataFrame([input_values])

        # Encode the categorical values
        nonnumeric = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system']
        input_df_encoded = pd.get_dummies(input_df, columns=nonnumeric)

        # Align the encoded data with the training data (this ensures no missing columns)
        input_df_encoded = input_df_encoded.reindex(columns=df_combined.columns.drop('price'), fill_value=0)

        # Scale the input values
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df_encoded)

        # Apply PCA transformation
        input_pca = pca.transform(input_scaled)

        # Make a prediction using the model
        prediction = model.predict(input_pca)

        # Display the prediction
        st.write("Predicted Price: ${:.2f}".format(prediction[0]))

if __name__ == '__main__':
    main()
