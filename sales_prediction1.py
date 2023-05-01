import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load the dataset
df = pd.read_csv("C:/Users/NAMAN/Desktop/Rajkot_SML.csv")
df.dtypes
df.isnull().count()
df = df.dropna()
df = df.replace(',','',regex=True)
# Rename the column
#df = df.rename(columns={' GENERAL PRODUCT LIST': 'GENERAL PRODUCT LIST'})



# Rename the column
df = df.rename(columns={' GENERAL PRODUCT LIST': 'GENERAL PRODUCT LIST'})

# Get the list of all product names
product_names = df['GENERAL PRODUCT LIST'].unique().tolist()

# Function to train the model and make predictions
def train_and_predict(product_name, target_month, sales_data):
    # Filter the dataset for the selected product
    product_data = df[df['GENERAL PRODUCT LIST'] == product_name]
    
    # Get the list of all months
    months = ['April', 'May', 'June', 'July', 'August', 'September', 'October','November','December']
    
    # Prepare the training data
    X_train = []
    y_train = []
    for i, month in enumerate(months):
        if month != target_month:
            X_train.append([i])
            y_train.append(product_data[f'TOTAL({month})'].values[0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make the prediction
    target_month_index = months.index(target_month)
    X_test = np.array([[target_month_index]])
    y_pred = model.predict(X_test)[0]
    
    # Print the result
    result = f"Predicted sales for {product_name} in {target_month} is {int(y_pred)} units."
    st.success(result)

# Create the Streamlit app
st.title('Sales Prediction')
product_name = st.selectbox('Select a product', product_names)
target_month = st.selectbox('Select a target month', ['October', 'November', 'December'])
sales_data = []
for month in ['April', 'May', 'June', 'July', 'August', 'September']:
    value = st.number_input(f"Enter sales data for {month} (in units)", min_value=0, step=1)
    sales_data.append(value)
if st.button('Predict'):
    train_and_predict(product_name, target_month, sales_data)
