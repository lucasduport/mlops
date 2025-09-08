import streamlit as st
import joblib

model = joblib.load('regression.joblib')

st.title('House Price Prediction')

size = st.number_input('Size (in sq ft)', min_value=0.0, step=1.0)
bedrooms = st.number_input('Number of bedrooms', min_value=0, step=1)
garden = st.number_input('Has garden? (1 for yes, 0 for no)', min_value=0, max_value=1, step=1)

if st.button('Predict Price'):
    input_data = [[size, bedrooms, garden]]
    prediction = model.predict(input_data)
    st.write(f'Predicted house price: ${prediction[0]:.2f}')
