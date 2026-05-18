import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# ---------------- TITLE ----------------

st.title("SVR Regression Prediction App")

st.write("Support Vector Regression using SVR")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("dataset.csv")

# ---------------- SHOW DATA ----------------

st.subheader("Dataset Preview")

st.write(df.head())

# ---------------- INPUT AND OUTPUT ----------------

X = df.drop("target", axis=1)

y = df["target"]

# ---------------- SPLIT DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------- CREATE MODEL ----------------

svr_model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)

# ---------------- TRAIN MODEL ----------------

svr_model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

y_pred = svr_model.predict(X_test)

# ---------------- EVALUATION ----------------

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

st.subheader("Model Performance")

st.success(f"R² Score: {r2:.2f}")

st.success(f"MSE Value: {mse:.2f}")

# ---------------- USER INPUT ----------------

st.subheader("Enter Input Values")

input_data = []

for column in X.columns:

    value = st.number_input(
        f"Enter {column}",
        min_value=0,
        max_value=500,
        value=0,
        step=1
    )

    input_data.append(value)

# ---------------- PREDICT BUTTON ----------------

if st.button("Predict"):

    input_array = np.array([input_data])

    # SCALE INPUT DATA

    input_array = scaler.transform(input_array)

    prediction = svr_model.predict(input_array)

    st.subheader("Prediction Result")

    final_prediction = round(prediction[0])

    st.success(f"Predicted Value: {final_prediction}")