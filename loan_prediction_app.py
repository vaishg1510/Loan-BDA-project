import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName('loan_prediction').getOrCreate()

# Load the trained model
model_path = 'loan_prediction_model'
data_model_path = 'data_model_pipeline'
model = PipelineModel.load(model_path)
data_model = PipelineModel.load(data_model_path)

# Define the layout of the Streamlit app
st.title("Loan Prediction App")
st.write("Enter the details to predict loan approval status.")

# Input fields for user data
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Convert to Spark DataFrame
input_spark_df = spark.createDataFrame(input_data)

# Transform the input data using the data pipeline
transformed_input = data_model.transform(input_spark_df)

# Get the prediction
prediction = model.transform(transformed_input).select("prediction").collect()[0][0]

# Display the prediction result
if st.button("Predict"):
    if prediction == 1:
        st.success("The loan is likely to be approved!")
    else:
        st.error("The loan is likely to be denied.")
