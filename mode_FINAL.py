# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Construct the file path
file_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'house_price.csv')

# Load dataset
df = pd.read_csv("C:\\Users\\Admin\\Downloads\\house_price.csv")

# Define features and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'furnished']]
y = df['price']

# Adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Calculate evaluation metrics without printing
mse, r2 = evaluate_model(y_test, y_pred)

# Streamlit app code
# st.title("Multi-Linear Regression Predictor")
# st.write("Enter the features to get a prediction:")

# Sidebar navigation
st.sidebar.subheader("Menu")
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "About"])

if page == "Home":
    st.title("Welcome to House Price Prediction")
    st.subheader('Accurate Predictions for Your Real Estate Needs')
    st.write("Discover the future value of your home with our advanced House Price Prediction tool. Whether you're a homeowner, buyer, or real estate professional, our platform offers insights into property values based on a variety of factors.")
    st.image("C:\\Users\\Admin\\Downloads\\prediction-transformed.jpeg", width=700)
    st.subheader('Features and Benefits:')
    st.markdown('''
                    -Highlight the accuracy of your model\n
                    -User-Friendly Interface\n
                    -Emphasize the ease of use\n
                    -Data Security \n
                    -Assure users that their data is secure.
             ''')

elif page == "Prediction":
    st.title(" Welcome to Prediction Page")
    st.image("C:\\Users\\Admin\\OneDrive\\Desktop\\house.jpg", width=500)
    st.write('Here you can predict the house prices of houses on the basis of multiple factors.')

    st.subheader('Enter the Required details here')
    area = st.number_input("Area in Square Feet", min_value=500, max_value=10000, value=1000)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
    furnished = st.selectbox('Furnished?', ['Yes', 'No'])

    if st.button('Predict'):
        # Convert user input to a feature array
        furnished_value = 1 if furnished == 'Yes' else 0
        features = np.array([[area, bedrooms, bathrooms, furnished_value]])
        
        # Apply polynomial transformation
        features_poly = poly.transform(features)
        
        # Predict the price
        predicted_price = model.predict(features_poly)
        
        # Display prediction
        predicted_price_inr = predicted_price[0] * 83  # Convert to INR (assuming 1 USD = 83 INR)
        st.write(f"Predicted price for the house: ₹{predicted_price_inr:,.2f}")

elif page == "About":
    st.title("About")
    st.write("This tool uses advanced algorithms to predict house prices.")
    st.write('Multiple Linear Regression (MLR) is a powerful statistical technique used to predict the value of a dependent variable based on the values of two or more independent variables. Here’s a detailed description tailored for your house price prediction model')
    st.write('''What is Multiple Linear Regression?
Multiple Linear Regression (MLR) is an extension of simple linear regression that allows for the inclusion of multiple independent variables to predict a single dependent variable. In the context of predicting house prices, these independent variables could include factors like the size of the house, the number of bedrooms, the location, the age of the property, and more.
How Does It Work?
The goal of MLR is to model the linear relationship between the dependent variable (house price) and the independent variables (features of the house). The relationship is expressed with the following equation:
Y=β0​+β1​X1​+β2​X2​+…+βn​Xn​+ϵ
''')
