import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = torch.load("full_model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# User input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine one-hot encoded geography
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale input data and convert to tensor
input_data_scaled = scaler.transform(input_data)
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Ensure correct input shape
if len(input_tensor.shape) == 1:
    input_tensor = input_tensor.unsqueeze(0)

# Predict churn
with torch.no_grad():
    prediction = model(input_tensor)

prediction_proba = prediction.item()

st.write(f'Churn Probability: {prediction_proba:.2f}')
st.write('The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.')
