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

---

## 🚀 How to Run the Project
Follow these steps to set up the environment and run the project on your local machine.


### 1. Clone the Repository
First, get a copy of the project code on your computer:

git clone https://github.com/toqa-ehab/Heart_Disease_Project.git
cd Heart_Disease_Project

---

### 2. Install Dependencies
Install all required Python libraries using pip:


pip install -r requirements.txt

---

### 3. Run the Jupyter Notebooks
Open Jupyter Notebook and run the notebooks in sequential order:

-Data Preprocessing: Clean and prepare the data
01_data_preprocessing.ipynb

-Dimensionality Reduction: PCA analysis
02_pca_analysis.ipynb

-Feature Selection: Identify most important features
03_feature_selection.ipynb

-Supervised Learning: Train classification models
04_supervised_learning.ipynb

-Unsupervised Learning: Clustering analysis
05_unsupervised_learning.ipynb

-Hyperparameter Tuning: Optimize the best model
06_hyperparameter_tuning.ipynb

---

### 4. Access the Trained Model
After running all notebooks, the final trained model will be saved at:
models/final_model.pkl

