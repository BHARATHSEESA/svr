# SVR Regression Prediction App

This project is a Machine Learning web application developed using Streamlit and Support Vector Regression (SVR).

The application predicts target values based on user input features.
streamlit link: https://hp2cmbqotbg482kqyrfaaq.streamlit.app/
---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## Machine Learning Algorithm

This project uses Support Vector Regression (SVR).

Kernel used:
- RBF Kernel

Parameters used:
- C = 100
- gamma = 0.1
- epsilon = 0.1

---

## Dataset Used

Custom CSV dataset.

The dataset contains:
- Input features
- Target column

The target column is used for prediction.

---

## Features of the App

- User-friendly interface
- Accepts user input values
- Predicts target values
- Displays R² Score
- Displays Mean Squared Error (MSE)
- Uses Feature Scaling

---

## Project Structure

```text
project-folder/
│
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
