# credit_card_fraud_detection
# 📊 Credit Card Fraud Detection using Logistic Regression

## 📝 Project Description
This project aims to detect fraudulent credit card transactions using Logistic Regression. We apply machine learning techniques on a publicly available dataset and evaluate model performance using various metrics.

## 👥 Team Members
- **Alice** — Data Preprocessing
- **Bob** — Exploratory Data Analysis & Feature Engineering
- **Charlie** — Model Building
- **Dana** — Model Evaluation & Report Preparation

## 📂 Project Structure
```
credit-card-fraud-detection/
│
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_logistic_regression.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ How to Run the Project
1. Clone the repository:
```
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run preprocessing:
```
python src/preprocessing.py
```
4. Train the model:
```
python src/model.py
```
5. Evaluate the model:
```
python src/evaluation.py
```

## 📊 Results Summary
| Metric          | Value |
|-----------------|-------|
| Accuracy        | 99%   |
| Precision       | 90%   |
| Recall          | 85%   |
| F1-Score        | 87%   |

## 🛠️ Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 📌 Notes
- Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- We use stratified sampling to maintain class distribution during train-test split.

---
✅ **Tip:** Each member can create their branch using the command:
```
git checkout -b feature-EDA
```

✅ Use clear commit messages like:
```
git commit -m "Added feature scaling to preprocessing script"
```

---
✨ **Let's detect fraud the smart way!** ✨
