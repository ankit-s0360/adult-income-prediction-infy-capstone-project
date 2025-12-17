# 🧠 Adult Income Prediction using Machine Learning

##  Project Overview:

This project is an end-to-end machine learning solution built on the UCI Adult Census Income Dataset.
The objective is to predict whether an individual earns more than $50,000 per year based on demographic, educational, and employment-related attributes.

The project follows industry-standard data science practices, including:

### Data cleaning & preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Machine learning pipelines

Model comparison & evaluation

Interpretability & professional documentation


Given census data containing attributes such as age, education, occupation, work hours, capital gains/losses, etc.,
build a machine learning model that accurately predicts:

Does this person earn more than $50K per year?

This is a binary classification problem.

### Dataset Information

Dataset Name: UCI Adult Census Income Dataset

Source: UCI 'Adult Dataset' Infy Capstone

Rows: ~32,561

Columns: ~15

Target Variable: income

 '<=50K'

 '>50K'

### Feature Types

Numeric Features:

age,
fnlwgt,
education_num,
capital_gain,
capital_loss,
hours_per_week

Categorical Features:

workclass,
education,
marital_status,
occupation,
relationship,
race,
sex,
native_country,

### Data Cleaning & Preprocessing

The dataset required multiple preprocessing steps before modeling:

#### -> Missing Values

Missing values represented by "?"

Converted to NaN

Rows with missing values were removed for data integrity

#### -> Outlier Handling

Outliers removed using the IQR method

Applied selectively to:

age

capital_gain

capital_loss

hours_per_week


#### -> Encoding & Scaling

One-Hot Encoding for categorical variables

Standard Scaling for numeric features

Implemented using ColumnTransformer


#### Machine Learning Pipeline

A scikit-learn Pipeline was used to combine preprocessing and modeling:
```text
Raw Data
   ↓
ColumnTransformer
   ├── Numeric → StandardScaler
   └── Categorical → OneHotEncoder
   ↓
Machine Learning Model
   ↓
Predictions
```
#### Why Pipeline?

Prevents data leakage

Ensures consistent preprocessing

Cleaner and reusable code

Production-ready design

#### Models Implemented

The following models were trained and compared:

Model	Purpose

Logistic Regression	Interpretable baseline classifier

Random Forest	Non-linear ensemble model

Gradient Boosting	Best-performing model

Support Vector Machine (SVM)	High-dimensional classifier

#### Model Evaluation

Models were evaluated using:

Accuracy,
Precision,
Recall,
F1-Score,
Confusion Matrix,
ROC Curve & AUC Score

#### Typical Performance (Adult Dataset)
Model	Accuracy
```text
Model comparison:
Model                Accuracy
Gradient Boosting  	 0.841603
Random Forest	     0.798917
Naive Bayes	         0.797400
SVM	                 0.788299
KNN	                 0.769881
LogisticRegression	 0.757963
```

Gradient Boosting performed best overall.

#### Feature Insights

###### Important predictors of high income:

Education level (education_num)

Capital gain

Working hours per week

Professional occupations

Negative indicators:

Low education

Certain low-paying workclasses

Unmarried status

#### Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

#### 🚀 How to Run This Project
# Clone repository
git clone https://github.com/ankit-s0360/adult-income-prediction-infy-capstone-project.git

# Navigate to project
cd adult-income-prediction-infy-capstone-project.git

# Open notebook
jupyter notebook

#### 📌 Key Learnings

Importance of preprocessing and encoding

Why pipelines prevent data leakage

How different models behave on tabular data

Interpreting ML results using metrics & plots

Writing production-ready ML code

#### Conclusion

This project demonstrates a complete machine learning workflow using real-world census data.
It emphasizes clean preprocessing, reproducibility, interpretability, and professional presentation — all critical skills for a data scientist.

#### Acknowledgements

UCI Machine Learning Repository
Scikit-learn Documentation

#### Contact

If you have suggestions or feedback, feel free to connect!
