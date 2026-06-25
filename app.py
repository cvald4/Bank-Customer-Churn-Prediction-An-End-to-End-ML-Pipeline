import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter a customer's profile below to predict their likelihood of leaving the bank.")

# Cache Data & Model Training
# @st.cache_resource ensures the model only trains once when the app starts, making it fast.
@st.cache_resource
def load_and_train_model():
    # Load Data
    df = pd.read_csv('Churn_Modelling.csv')
    df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    
    # Feature Engineering
    df_clean['BalanceSalaryRatio'] = df_clean['Balance'] / df_clean['EstimatedSalary']
    df_clean['ProductsPerTenure'] = df_clean['NumOfProducts'] / (df_clean['Tenure'] + 1)
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    
    # Encoding
    df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender', 'AgeGroup'], drop_first=True)
    
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train LightGBM
    model = lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

model, scaler, feature_columns = load_and_train_model()

# User Input Sidebar
st.sidebar.header("📝 Customer Profile")

age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
salary = st.sidebar.number_input("Estimated Salary ($)", 1000.0, 200000.0, 60000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
has_crcard = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Process Inputs on Button Click
if st.button("Predict Churn Risk"):
    # Translate UI inputs into the exact format the model expects
    input_data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_crcard == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': salary,
        'BalanceSalaryRatio': balance / salary if salary > 0 else 0,
        'ProductsPerTenure': num_products / (tenure + 1),
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'AgeGroup_31-40': 1 if 31 <= age <= 40 else 0,
        'AgeGroup_41-50': 1 if 41 <= age <= 50 else 0,
        'AgeGroup_51-60': 1 if 51 <= age <= 60 else 0,
        'AgeGroup_60+': 1 if age > 60 else 0
    }

    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    prob_percentage = round(prob * 100, 2)

    # Display Results
    st.divider()
    st.subheader("📊 Prediction Results")
    
    # Logic to color-code the risk level
    if prob > 0.70:
        st.error(f"**High Flight Risk:** This customer has a **{prob_percentage}%** probability of churning. Immediate retention action is recommended. 🚨")
    elif prob > 0.40:
        st.warning(f"**Moderate Risk:** This customer has a **{prob_percentage}%** probability of churning. Monitor activity closely. ⚠️")
    else:
        st.success(f"**Low Risk:** This customer is stable with only a **{prob_percentage}%** probability of churning. ✅")
        
    st.progress(float(prob))