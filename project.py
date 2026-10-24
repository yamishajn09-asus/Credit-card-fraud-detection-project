import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.header("Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
df = pd.get_dummies(df,drop_first = True)
    

st.subheader("Data Preview")
st.dataframe(df.head())

st.write("Dataset shape:", df.shape)
st.write("Missing values:", int(df.isnull().sum().sum()))
st.write("Transaction classes:")
st.write(df["Class"].value_counts())

if df.isnull().sum().sum() > 0:
    st.error("Please clean missing values before training")
    st.stop()

x=df.drop(columns=["Class"])
y=df["Class"]  

if y.nunique() !=2 or y.value_counts().min() < 2:
    st.error("Dataset must contain both classes with at least 2 samples each for training.")
    st.stop()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler=StandardScaler()
x_train =x_train.copy()
x_test=x_test.copy()
LogisticRegression = LogisticRegression(max_iter = 200)
columns_to_scale=[
    c for c in ["Time", "Amount"] if c in x.columns
]

x_train[columns_to_scale]=scaler.fit_transform(
    x_train[columns_to_scale]
)
x_test[columns_to_scale]=scaler.transform(  
     x_test[columns_to_scale]
)
LogisticRegression.fit(x_train, y_train)

predictions = LogisticRegression.predict(x_test)

st.subheader("Model Performance")
st.text(classification_report(y_test, predictions,zero_division=0
))

st.write("Confusion Matrix")
st.dataframe(pd.DataFrame(
    confusion_matrix(y_test,predictions),
    index=["Actual Legitimate","Actual Fraud"],
    columns=["Predicted legitimate","Predicted Fraud"]

))
st.success("Model Trained Successfully")

st.subheader("Test a transaction")
index=st.number_input(
    "Test row number",
    min_value=0,
    max_value=len(x_test) - 1,
    value=0,
    step=1
)
if st.button("Predict Transaction"):
    sample=x_test.iloc[[int(index)]]
    prediction=LogisticRegression.predict(sample)[0]
    probability = LogisticRegression.predict_proba(sample)[0][1]

    st.metric("Estimated Fraud Probability",
              f"{probability:2%}")

    if prediction==1:
        st.error("Potential Fraudulent Transaction!")
    else:
        st.success("Classified as Legitimate")

else:
    st.info("Upload your creditcard.csv dataset to begin")                     