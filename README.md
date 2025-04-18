# Predict-loan-.Default
# 🏦 Loan Default Prediction using Random Forest

This project aims to predict whether a borrower will default on a loan using a machine learning model. The dataset is processed and a Random Forest Classifier is trained to make predictions.

## 📁 Dataset

- **File:** `1. Predict Loan Default.csv`
- **Description:** Contains borrower details and whether they defaulted on a loan.
- **Target Column:** `Default` (1 = Defaulted, 0 = Not Defaulted)

## ⚙️ Libraries Used

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🧠 Steps Involved

### 1. Data Preprocessing
- Removed `LoanID` (non-informative).
- Encoded categorical features using Label Encoding.
- Sampled 20,000 records for faster training.

### 2. Model Training
- Used `RandomForestClassifier` from Scikit-learn.
- Trained on 80% of the data, tested on 20%.

### 3. Evaluation Metrics
- Accuracy
- Precision
- Recall
- Confusion Matrix (visualized using Seaborn heatmap)

## 🧪 Model Performance

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.81  |
| Precision  | 0.76  |
| Recall     | 0.74  |

## 📊 Confusion Matrix

![Confusion Matrix Heatmap](confusion_matrix.png) <!-- Replace this with actual path if image is saved -->

## 🚀 How to Run

1. Clone the repo or download the script.
2. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
