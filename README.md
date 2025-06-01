# Breast Cancer Detection

 This project is a machine learning model built using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`. It uses **Logistic Regression** to classify tumors as **Malignant (Cancerous)** or **Benign (Non-cancerous)** based on input features.

## 📌 Features Used

The dataset contains 30 features per sample, such as:

- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- ... and more.

Each feature represents characteristics of the cell nuclei present in a digitized image of a fine needle aspirate (FNA) of a breast mass.

## 🚀 How It Works

    1. Load the dataset from `sklearn.datasets`.
    2. Split the data into training and testing sets (90% train, 10% test).
    3. Train a Logistic Regression model on the training data.
    4. Evaluate the model on both training and test data using accuracy score.
    5. Allow the user to input 30 feature values to predict whether a tumor is benign or malignant.

## 🧠 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`

## 📊 Accuracy

- **Training Accuracy**: ~95%
- **Testing Accuracy**: ~93%

## 📁 Dataset

The dataset is built into `scikit-learn` and does not require external downloads.
