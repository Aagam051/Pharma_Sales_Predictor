import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Define the predict_sales function
def predict_sales(start_date, end_date, drug, loaded_model):
    # Generate a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create the DataFrame with dates as the index
    df_test = pd.DataFrame(index=dates)
    df_test['Year'] = df_test.index.year
    df_test['Month'] = df_test.index.month
    df_test['Weekday Name'] = df_test.index.weekday
    df_test['day'] = df_test.index.day
    df_test['Drug'] = drug
    
    # Encoding weekday names using LabelEncoder
    le = LabelEncoder()
    df_test['Weekday Name'] = le.fit_transform(df_test['Weekday Name'])

    # Assuming you have loaded_model already defined
    df_test['predicted_quantity'] = loaded_model.predict(df_test)
    
    return df_test

# Streamlit UI
st.title('Sales Prediction App')

# Date selection widgets
start_date = st.date_input('Select start date')
end_date = st.date_input('Select end date')

# Drug category selection
drug = st.slider('Select drug category', min_value=0, max_value=7)

# Load the model from pickle file
with open('pharma_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Button to trigger prediction
if st.button('Predict'):
    # Call predict_sales function with input values
    sales_prediction = predict_sales(start_date, end_date, drug, loaded_model)
    # Display prediction result
    st.write(sales_prediction)
