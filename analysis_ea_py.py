import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve, classification_report, mean_squared_error, r2_score)
from sklearn.utils import class_weight
import io

# Set Streamlit Page Config
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", page_icon="📊")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING & PREPROCESSING ---

@st.cache_data
def load_data():
    try:
        # Attempt to load the dataset provided by the user
        df = pd.read_csv("Employee-Attrition - Employee-Attrition.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'Employee-Attrition - Employee-Attrition.csv' not found. Please upload it or ensure it is in the directory.")
        return None

def preprocess_data(df):
    """
    Cleans and preprocesses the dataframe.
    """
    df_clean = df.copy()
    
    # 1. Drop irrelevant columns (Constant values or IDs)
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    # 2. Encode Binary Categorical Variables
    # Attrition: Yes=1, No=0
    if 'Attrition' in df_clean.columns:
        df_clean['Attrition'] = df_clean['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    if 'OverTime' in df_clean.columns:
        df_clean['OverTime'] = df_clean['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
        
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # 3. Feature Engineering
    # Tenure Ratios
    if 'TotalWorkingYears' in df_clean.columns and 'Age' in df_clean.columns:
        df_clean['CareerMaturity'] = df_clean['TotalWorkingYears'] / df_clean['Age']
    
    return df_clean

def get_feature_matrix(df, target_col):
    """
    Prepares X and y for ML models. Handles OneHotEncoding for remaining categorical vars.
    """
    df_processed = df.copy()
    
    # Drop target from X
    if target_col in df_processed.columns:
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
    else:
        # Handle case where target col might not be in input (for inference sometimes)
        X = df_processed
        y = None
    
    # Identify categorical columns that still need encoding (excluding already encoded binary ones)
    # We kept Gender/OverTime as numeric (1/0), so we look for object types
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y

# --- 2. MODEL TRAINING FUNCTIONS ---

def train_attrition_model(X_train, y_train, model_type="Random Forest"):
    if model_type == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    return model

def train_performance_model(X_train, y_train):
    # PerformanceRating is typically 3 or 4 in this dataset. Classification is appropriate.
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_promotion_model(X_train, y_train):
    # Target: YearsSinceLastPromotion (Regression)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- MAIN APP STRUCTURE ---

def main():
    st.markdown("<h1 class='main-header'>🏢 HR Analytics: Attrition, Performance & Promotion </h1>", unsafe_allow_html=True)
    
    # Load Data
    raw_df = load_data()
    if raw_df is None:
        return

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Project Report", "📊 Exploratory Data Analysis", "🤖 Predictions (ML)", "🔮 Interactive Tool"])

    # --- TAB 1: PROJECT REPORT ---
    with tab1:
        st.header("Project Documentation & Deliverables")
        
        st.subheader("1. Problem Statement")
        st.write("""
        Employee turnover poses a significant challenge for organizations. This project analyzes employee data 
        to identify key drivers of attrition and builds predictive models to support proactive retention strategies.
        """)
        
        st.subheader("2. Approach")
        st.markdown("""
        * **Data Collection**: Used the Employee Attrition dataset.
        * **Preprocessing**: Removed constant columns (`EmployeeCount`, `Over18`), encoded binary variables (`Attrition`, `OverTime`), and applied One-Hot Encoding for categorical features.
        * **EDA**: Analyzed correlations and distributions to find high-risk groups.
        * **Modeling**: 
            * *Task 1*: **Attrition Prediction** (Binary Classification) using Random Forest/Logistic Regression.
            * *Task 2*: **Performance Rating Prediction** (Classification) using Random Forest.
            * *Task 3*: **Promotion Likelihood** (Regression) predicting Years Since Last Promotion.
        * **Evaluation**: Used Accuracy, Precision, Recall, F1-Score, AUC-ROC (Classification) and MSE/R2 (Regression).
        """)
        
        st.subheader("3. Key Deliverables")
        st.success("✅ Source Code (This Streamlit App)")
        st.success("✅ Cleaned Data (Processed in real-time)")
        st.success("✅ Trained Models (Random Forest, Logistic Regression)")
        st.success("✅ Visualizations & Dashboard")

    # --- TAB 2: EDA ---
    with tab2:
        st.header("Exploratory Data Analysis")
        
        df_display = raw_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(df_display.head())
            st.write(f"**Shape:** {df_display.shape[0]} rows, {df_display.shape[1]} columns")
            
        with col2:
            st.subheader("Attrition Distribution")
            if 'Attrition' in df_display.columns:
                fig, ax = plt.subplots()
                sns.countplot(x='Attrition', data=df_display, palette='viridis', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Attrition column not found.")

        st.divider()
        
        # Correlation Heatmap
        st.subheader("Correlation Matrix (Numerical Features)")
        # Select only numeric for heatmap
        numeric_df = preprocess_data(raw_df).select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        
        st.divider()
        
        # Bivariate Analysis
        st.subheader("Key Factors Analysis")
        factor = st.selectbox("Select Factor to Analyze against Attrition:", 
                              ["OverTime", "JobRole", "MaritalStatus", "Department", "Gender"])
        
        if factor in df_display.columns and 'Attrition' in df_display.columns:
            fig_bi, ax_bi = plt.subplots(figsize=(10, 5))
            if df_display[factor].dtype == 'O' or len(df_display[factor].unique()) < 10:
                # Categorical plot
                sns.countplot(x=factor, hue='Attrition', data=df_display, palette='Set2', ax=ax_bi)
                plt.xticks(rotation=45)
            else:
                # Numerical plot (Boxplot)
                sns.boxplot(x='Attrition', y=factor, data=df_display, palette='Set2', ax=ax_bi)
            st.pyplot(fig_bi)

    # --- TAB 3: MODEL TRAINING & RESULTS ---
    with tab3:
        st.header("Machine Learning Model Results")
        
        # Process Data
        df_clean = preprocess_data(raw_df)
        
        # --- TASK 1: ATTRITION PREDICTION ---
        st.subheader("Task 1: Predicting Employee Attrition (Classification)")
        
        model_choice = st.radio("Choose Algorithm for Attrition:", ["Random Forest", "Logistic Regression"], horizontal=True)
        
        # Prepare Data
        X_att, y_att = get_feature_matrix(df_clean, 'Attrition')
        X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_att, y_att, test_size=0.2, random_state=42, stratify=y_att)
        
        # Train
        model_att = train_attrition_model(X_train_att, y_train_att, model_choice)
        y_pred_att = model_att.predict(X_test_att)
        y_prob_att = model_att.predict_proba(X_test_att)[:, 1]
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{accuracy_score(y_test_att, y_pred_att):.2%}")
        col_m2.metric("Precision", f"{precision_score(y_test_att, y_pred_att):.2%}")
        col_m3.metric("Recall", f"{recall_score(y_test_att, y_pred_att):.2%}")
        col_m4.metric("AUC-ROC", f"{roc_auc_score(y_test_att, y_prob_att):.2f}")
        
        # Plots
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test_att, y_pred_att)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            st.pyplot(fig_cm)
            
        with col_p2:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test_att, y_prob_att)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test_att, y_prob_att):.2f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.legend()
            st.pyplot(fig_roc)
            
        st.divider()
        
        # --- TASK 2: PERFORMANCE PREDICTION ---
        st.subheader("Task 2: Predicting Performance Rating (Classification)")
        st.write("Target Variable: `PerformanceRating` (3 = High, 4 = Very High)")
        
        # Remove Attrition from features
        df_perf = df_clean.drop(columns=['Attrition']) if 'Attrition' in df_clean.columns else df_clean
        X_perf, y_perf = get_feature_matrix(df_perf, 'PerformanceRating')
        
        X_train_perf, X_test_perf, y_train_perf, y_test_perf = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42, stratify=y_perf)
        
        # Train
        model_perf = train_performance_model(X_train_perf, y_train_perf)
        y_pred_perf = model_perf.predict(X_test_perf)
        
        col_pm1, col_pm2 = st.columns(2)
        col_pm1.metric("Accuracy", f"{accuracy_score(y_test_perf, y_pred_perf):.2%}")
        col_pm2.metric("F1 Score (Weighted)", f"{f1_score(y_test_perf, y_pred_perf, average='weighted'):.2%}")
        
        st.divider()

        # --- TASK 3: PROMOTION LIKELIHOOD ---
        st.subheader("Task 3: Promotion Likelihood (Regression)")
        st.write("Target Variable: `YearsSinceLastPromotion` (Predicting expected years since last promotion based on profile)")
        
        # Remove Attrition from features, using clean data
        df_promo = df_clean.drop(columns=['Attrition']) if 'Attrition' in df_clean.columns else df_clean
        
        # Target is YearsSinceLastPromotion
        X_promo, y_promo = get_feature_matrix(df_promo, 'YearsSinceLastPromotion')
        
        X_train_promo, X_test_promo, y_train_promo, y_test_promo = train_test_split(X_promo, y_promo, test_size=0.2, random_state=42)
        
        # Train
        model_promo = train_promotion_model(X_train_promo, y_train_promo)
        y_pred_promo = model_promo.predict(X_test_promo)
        
        # Metrics for Regression
        col_pr1, col_pr2 = st.columns(2)
        rmse = np.sqrt(mean_squared_error(y_test_promo, y_pred_promo))
        r2 = r2_score(y_test_promo, y_pred_promo)
        
        col_pr1.metric("RMSE (Years)", f"{rmse:.2f}")
        col_pr2.metric("R2 Score", f"{r2:.2f}")
        
        st.info("A higher R2 score indicates the model explains the variance in promotion timelines well.")


    # --- TAB 4: INTERACTIVE PREDICTION ---
    with tab4:
        st.header("Interactive Prediction Tool")
        st.write("Adjust the parameters on the sidebar to simulate an employee profile.")
        
        # Sidebar Inputs
        st.sidebar.header("Employee Profile Input")
        
        def user_input_features():
            age = st.sidebar.slider("Age", 18, 60, 30)
            daily_rate = st.sidebar.slider("DailyRate", 100, 1500, 800)
            distance = st.sidebar.slider("DistanceFromHome", 1, 30, 5)
            education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)
            env_sat = st.sidebar.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
            job_sat = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
            job_involve = st.sidebar.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
            job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
            over_time = st.sidebar.selectbox("OverTime", ["Yes", "No"])
            income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
            total_years = st.sidebar.slider("Total Working Years", 0, 40, 8)
            years_at_company = st.sidebar.slider("Years At Company", 0, 40, 5)
            years_in_role = st.sidebar.slider("Years In Current Role", 0, 20, 3)
            # This is input for Attrition, but Target for Promotion.
            years_since_promo_input = st.sidebar.slider("Years Since Last Promotion (Current)", 0, 15, 1)
            
            data = {
                'Age': age,
                'DailyRate': daily_rate,
                'DistanceFromHome': distance,
                'Education': education,
                'EnvironmentSatisfaction': env_sat,
                'JobSatisfaction': job_sat,
                'JobInvolvement': job_involve,
                'JobLevel': job_level,
                'OverTime': over_time,
                'MonthlyIncome': income,
                'TotalWorkingYears': total_years,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_role,
                'YearsSinceLastPromotion': years_since_promo_input,
                # Default dummy values
                'Gender': 'Male', 
                'MaritalStatus': 'Single', 
                'Department': 'Sales', 
                'JobRole': 'Sales Executive', 
                'BusinessTravel': 'Travel_Rarely', 
                'EducationField': 'Life Sciences', 
                'NumCompaniesWorked': 1,
                'PercentSalaryHike': 15,
                'RelationshipSatisfaction': 3,
                'StockOptionLevel': 0,
                'TrainingTimesLastYear': 2,
                'WorkLifeBalance': 3,
                'YearsWithCurrManager': 2,
                'EmployeeCount': 1, 'StandardHours': 80, 'Over18': 'Y', 'EmployeeNumber': 9999
            }
            return pd.DataFrame(data, index=[0])

        input_df = user_input_features()
        
        st.subheader("Employee Profile Summary")
        st.write(input_df[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'JobSatisfaction']])
        
        # PREDICTION BUTTON
        if st.button("Predict Outcomes"):
            # 1. Preprocess input
            input_clean = preprocess_data(input_df)
            
            # --- PREDICT ATTRITION ---
            input_encoded_att = pd.get_dummies(input_clean)
            train_cols_att = X_train_att.columns
            for col in train_cols_att:
                if col not in input_encoded_att.columns:
                    input_encoded_att[col] = 0
            input_encoded_att = input_encoded_att[train_cols_att]
            
            att_prediction = model_att.predict(input_encoded_att)[0]
            att_probability = model_att.predict_proba(input_encoded_att)[0][1]
            
            # --- PREDICT PERFORMANCE ---
            input_encoded_perf = pd.get_dummies(input_clean)
            train_cols_perf = X_train_perf.columns
            for col in train_cols_perf:
                if col not in input_encoded_perf.columns:
                    input_encoded_perf[col] = 0
            input_encoded_perf = input_encoded_perf[train_cols_perf]
            
            perf_prediction = model_perf.predict(input_encoded_perf)[0]

            # --- PREDICT PROMOTION ---
            # IMPORTANT: Remove 'YearsSinceLastPromotion' from input features because it is the target here
            input_clean_promo = input_clean.drop(columns=['YearsSinceLastPromotion'], errors='ignore')
            input_encoded_promo = pd.get_dummies(input_clean_promo)
            
            train_cols_promo = X_train_promo.columns
            for col in train_cols_promo:
                if col not in input_encoded_promo.columns:
                    input_encoded_promo[col] = 0
            input_encoded_promo = input_encoded_promo[train_cols_promo]
            
            promo_prediction_years = model_promo.predict(input_encoded_promo)[0]
            
            # Display Results
            st.divider()
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("1. Attrition Risk")
                if att_prediction == 1:
                    st.error(f"High Risk ({att_probability:.1%})")
                    st.write("👉 Recommendation: Review salary & workload.")
                else:
                    st.success(f"Low Risk ({att_probability:.1%})")
                    st.write("👉 Employee seems stable.")
            
            with c2:
                st.subheader("2. Performance")
                st.info(f"Predicted Rating: {perf_prediction}")
                if perf_prediction == 4:
                    st.write("🌟 Top Performer")
                else:
                    st.write("👍 Meets Expectations")
                    
            with c3:
                st.subheader("3. Promotion")
                st.warning(f"Expected Time Since Last Promo: {promo_prediction_years:.1f} Years")
                
                # Logic to interpret the result
                current_val = input_df['YearsSinceLastPromotion'].iloc[0]
                diff = current_val - promo_prediction_years
                
                if diff > 2:
                    st.write("⚠️ **Overdue**: This employee has waited longer than typical for their profile.")
                elif diff < -2:
                    st.write("✅ **Fast Track**: This employee was promoted more recently than typical.")
                else:
                    st.write("⚖️ **On Track**: Tenure aligns with expected timeline.")

if __name__ == "__main__":
    main()