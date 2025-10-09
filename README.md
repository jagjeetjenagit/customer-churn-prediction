# Customer Churn Prediction Project

A comprehensive machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline for binary classification, specifically designed for data science beginners. We predict whether a customer will churn (leave the service) based on their demographic and service usage patterns.

## 📊 Dataset

- **Source**: Telco Customer Churn Dataset (via Kaggle Hub)
- **Features**: 21 features including demographics, services, and account information
- **Target**: Churn (Yes/No)
- **Size**: ~7,000 customer records

## 🔧 Technical Stack

- **Python 3.13.7**
- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: LogisticRegression, RandomForest, XGBoost
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Deployment**: Streamlit web app
- **Model Persistence**: joblib

## 🚀 Key Features

### Data Analysis
- Comprehensive EDA with distribution analysis
- Data quality checks (missing values, data types)
- Feature correlation analysis

### Machine Learning Pipeline
- ColumnTransformer for preprocessing
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Class imbalance handling with SMOTE
- Hyperparameter tuning with RandomizedSearchCV
- Feature importance analysis

### Model Deployment
- Saved model artifacts for production use
- Interactive Streamlit web application
- Real-time prediction with probability scores

## 📁 Project Structure

```
Customer curn data/
├── customer_churn_prediction.ipynb  # Main analysis notebook
├── churn_app.py                     # Streamlit web application
├── models/                          # Saved model artifacts
│   ├── preprocessor.pkl
│   ├── best_churn_model.pkl
│   └── model_info.pkl
└── README.md
```

## 🎯 Model Performance

- **Best Model**: XGBoost Classifier
- **Accuracy**: ~80% on test set
- **ROC-AUC**: Strong performance on imbalanced dataset
- **Feature Importance**: Monthly charges and tenure are key predictors

## 🌐 Web Application

### 🚀 One-Click Deployment

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/new?repository=jagjeetjenagit/customer-churn-prediction&branch=main&mainModule=churn_app.py)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jagjeetjenagit/customer-churn-prediction?quickstart=1)

### 🚀 Live Demo
**[Try the live app on Streamlit Cloud!](https://customer-churn-prediction-jagjeetjenagit.streamlit.app)**

### Local Setup
Run the Streamlit app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run churn_app.py
```

The app provides:
- Interactive input form with 18 customer features
- Real-time churn prediction
- Probability scores for decision confidence
- Professional UI for business users

## 🔍 Key Insights

1. **Monthly charges** and **tenure** are the strongest predictors of churn
2. **Contract type** significantly impacts customer retention
3. **Class imbalance** (SMOTE) improved minority class detection

## 🚧 Next Steps

1. Feature engineering with tenure buckets and charge ratios
2. Advanced ensemble methods (Voting, Stacking)
3. Model interpretability with SHAP values

## 👨‍💻 For Beginners

This project includes:
- Detailed code comments explaining each step
- Business context for technical decisions
- Error handling and data validation
- Production-ready model deployment

## 📝 License

This project is for educational purposes and portfolio demonstration.