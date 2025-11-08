
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# DEMO ONLY - Minimal Streamlit app for customer churn prediction
# To run: pip install streamlit && streamlit run churn_app.py

st.title("🔮 Customer Churn Prediction")

# Add navigation to dashboard
st.info("💡 **New!** Check out the [Analysis Dashboard](dashboard.py) for comprehensive insights and visualizations!")
st.write("**Demo App** - Predict if a customer will churn")

# Load saved artifacts
@st.cache_resource
def load_models():
    """Load preprocessor and model - cached for performance"""
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/best_churn_model.pkl')
        model_info = joblib.load('models/model_info.pkl')
        return preprocessor, model, model_info
    except FileNotFoundError:
        st.error("Model files not found! Run the notebook first to train and save models.")
        return None, None, None

preprocessor, model, model_info = load_models()

if model is not None:
    st.success(f"✅ Loaded: {model_info['model_name']} (F1: {model_info['f1_score']:.3f})")

    # Simple input form (demo - you'd want a proper UI)
    st.subheader("📝 Customer Information")

    # Example inputs - in real app, you'd have proper forms for all features
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        payment_method = st.selectbox("Payment Method", 
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Additional customer details
    st.subheader("👤 Personal & Service Details")
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        partner = st.selectbox("Partner", ['No', 'Yes'])
        dependents = st.selectbox("Dependents", ['No', 'Yes'])
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

    with col4:
        online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    # Note: Now using most actual features
    st.info("✅ Enhanced Demo: Now using 18 out of 19 features for more accurate predictions!")

    if st.button("🔮 Predict Churn"):
        # Create a sample input (demo purposes - missing many features)
        # In production, you'd collect all required features
        sample_input = {
            'SeniorCitizen': senior_citizen,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': monthly_charges * tenure,  # Rough estimate
            'Contract': contract,
            'InternetService': internet_service,
            'PaymentMethod': payment_method,
            # Now using actual user inputs instead of defaults
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'PaperlessBilling': paperless_billing
        }

        # Convert to DataFrame and make prediction
        input_df = pd.DataFrame([sample_input])

        try:
            # Preprocess and predict
            processed_input = preprocessor.transform(input_df)
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0]

            # Display results
            if prediction == 1:
                st.error(f"🚨 HIGH CHURN RISK - Probability: {probability[1]:.2%}")
                st.write("Consider retention strategies!")
            else:
                st.success(f"✅ LOW CHURN RISK - Probability: {probability[0]:.2%}")
                st.write("Customer likely to stay!")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.write("---")
st.write("**Note**: This is a demo app. Production version would:")
st.write("- Collect all required features properly")
st.write("- Have better error handling")  
st.write("- Include data validation")
st.write("- Have a professional UI design")
