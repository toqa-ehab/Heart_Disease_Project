# 🫀 Heart Disease Prediction Project

## 📌 Project Overview
This project implements a **full machine learning pipeline** on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).  
The goal is to analyze, predict, and visualize heart disease risks using both **supervised** and **unsupervised** learning techniques.

---

## 🎯 Objectives
- Perform **data preprocessing & cleaning** (missing values, encoding, scaling).
- Apply **dimensionality reduction (PCA)** to retain essential features.
- Use **feature selection** methods (RFE, Random Forest importance, Chi-Square).
- Train **supervised models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Apply **unsupervised learning**:
  - K-Means clustering
  - Hierarchical clustering
- Optimize models using **hyperparameter tuning** (GridSearchCV & RandomizedSearchCV).
- Save the **final model** for reproducibility.
  
---

## 🛠️ Tools & Libraries
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Feature Selection & Dimensionality Reduction:** PCA, RFE, Chi-Square  
- **Supervised Learning:** Logistic Regression, Decision Tree, Random Forest, SVM  
- **Unsupervised Learning:** K-Means, Hierarchical Clustering  
- **Optimization:** GridSearchCV, RandomizedSearchCV  

---

## 📂 Project Structure
- `data/` → contains the dataset (`heart_disease.csv`)  
- `notebooks/` → step-by-step analysis (preprocessing, PCA, feature selection, models)  
- `models/` → saved trained models (`final_model.pkl`)  
- `results/` → evaluation metrics  

---

## 📌 Dataset
UCI Heart Disease Dataset:  
👉 [https://archive.ics.uci.edu/ml/datasets/heart+Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

