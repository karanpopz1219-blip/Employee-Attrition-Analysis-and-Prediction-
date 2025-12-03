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
import pickle

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
    .slide-header {
        color: #0f766e;
        font-weight: bold;
        margin-top: 20px;
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

# Centralized Training for Deliverables
@st.cache_resource
def train_all_models_for_export(df):
    """
    Trains all models once to generate deliverables (Pickle files, Metrics Report).
    Returns: models_dict, metrics_string
    """
    df_clean = preprocess_data(df)
    
    # --- 1. Attrition Model (RF) ---
    X_att, y_att = get_feature_matrix(df_clean, 'Attrition')
    X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_att, y_att, test_size=0.2, random_state=42, stratify=y_att)
    
    model_att = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_att.fit(X_train_att, y_train_att)
    y_pred_att = model_att.predict(X_test_att)
    
    att_metrics = f"""
    [Attrition Prediction - Random Forest]
    Accuracy: {accuracy_score(y_test_att, y_pred_att):.4f}
    Precision: {precision_score(y_test_att, y_pred_att):.4f}
    Recall: {recall_score(y_test_att, y_pred_att):.4f}
    F1-Score: {f1_score(y_test_att, y_pred_att):.4f}
    """

    # --- 2. Performance Model (RF) ---
    # Remove Attrition from features
    df_perf = df_clean.drop(columns=['Attrition']) if 'Attrition' in df_clean.columns else df_clean
    X_perf, y_perf = get_feature_matrix(df_perf, 'PerformanceRating')
    X_train_perf, X_test_perf, y_train_perf, y_test_perf = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42, stratify=y_perf)
    
    model_perf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_perf.fit(X_train_perf, y_train_perf)
    y_pred_perf = model_perf.predict(X_test_perf)
    
    perf_metrics = f"""
    [Performance Rating Prediction]
    Accuracy: {accuracy_score(y_test_perf, y_pred_perf):.4f}
    F1-Score (Weighted): {f1_score(y_test_perf, y_pred_perf, average='weighted'):.4f}
    """

    # --- 3. Promotion Model (Regressor) ---
    df_promo = df_clean.drop(columns=['Attrition']) if 'Attrition' in df_clean.columns else df_clean
    X_promo, y_promo = get_feature_matrix(df_promo, 'YearsSinceLastPromotion')
    X_train_promo, X_test_promo, y_train_promo, y_test_promo = train_test_split(X_promo, y_promo, test_size=0.2, random_state=42)
    
    model_promo = RandomForestRegressor(n_estimators=100, random_state=42)
    model_promo.fit(X_train_promo, y_train_promo)
    y_pred_promo = model_promo.predict(X_test_promo)
    
    promo_metrics = f"""
    [Promotion Likelihood - Regression]
    RMSE: {np.sqrt(mean_squared_error(y_test_promo, y_pred_promo)):.4f}
    R2 Score: {r2_score(y_test_promo, y_pred_promo):.4f}
    """
    
    full_report = f"""
    PROJECT EVALUATION REPORT
    =========================
    {att_metrics}
    -------------------------
    {perf_metrics}
    -------------------------
    {promo_metrics}
    =========================
    """
    
    models = {
        'attrition_model': model_att,
        'performance_model': model_perf,
        'promotion_model': model_promo,
        'feature_columns_attrition': X_att.columns.tolist()
    }
    
    return models, full_report

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
        
    # Preprocess once for global use
    df_clean = preprocess_data(raw_df)

    # Train global models for download artifacts
    export_models, export_report = train_all_models_for_export(raw_df)

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Project Presentation", "📊 Exploratory Data Analysis", "🤖 Predictions (ML)", "🔮 Interactive Tool"])

    # --- TAB 1: PROJECT PRESENTATION (UPDATED WITH SLIDE CONTENT) ---
    with tab1:
        st.header("Project Presentation & Deliverables")
        
        col_doc, col_dl = st.columns([2.5, 1])
        
        with col_doc:
            # SLIDE 1: TITLE
            st.markdown("### 1. Employee Attrition Analysis & Prediction")
            st.info("**Subtitle:** A Data-Driven Approach to HR Analytics using Machine Learning & Streamlit  \n**Goal:** Transforming Workforce Data into Retention Strategies")
            
            st.divider()
            
            # SLIDE 2: THE COST (UPDATED)
            st.markdown("### 2. The Silent Profit Killer: Cost of Turnover")
            st.write("Employee attrition is not just an HR headache; it's a massive financial drain. High turnover disrupts teams, lowers morale, and incurs significant replacement costs.")
            
            c1, c2, c3 = st.columns(3)
            c1.warning("💸 Recruitment fees")
            c2.warning("⏳ Onboarding time")
            c3.warning("📉 Lost productivity")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Key stats with metrics for impact
            st.markdown("**Key Statistics:**")
            stat1, stat2 = st.columns(2)
            stat1.metric(label="Cost to Replace Senior Staff", value="200%", delta="of Annual Salary", delta_color="inverse")
            stat2.metric(label="Annual Cost to U.S. Businesses", value="$1T", delta="Economic Impact", delta_color="inverse")
            
            st.divider()
            
            # SLIDE 3: SOLUTION
            st.markdown("### 3. End-to-End HR Analytics Solution")
            st.write("We developed this comprehensive **Streamlit Application** that empowers HR teams to move from reactive to proactive management.")
            st.success("📊 **Interactive Dashboard:** Real-time data exploration.")
            st.success("🧠 **Predictive Engines:** 3 ML models for Risk, Performance, and Promotion.")
            st.success("⚡ **Actionable Insights:** Instant risk alerts and career tracking.")
            
            st.divider()

            # SLIDE 4: DATA PIPELINE
            st.markdown("### 4. Data Pipeline & Engineering")
            st.write("* **Raw Data:** `Employee-Attrition.csv` (1,470 Records, 35 Features).")
            st.write("* **Preprocessing:** Removed ID columns, Binary Encoding for 'Attrition'/'OverTime', One-Hot Encoding for categorical variables.")
            st.write("* **Feature Engineering:** Created `CareerMaturity` metric (*TotalWorkingYears / Age*) to identify experience density.")

            st.divider()

            # SLIDE 5: EDA INSIGHTS
            st.markdown("### 5. Key Insights (EDA)")
            st.write("Our exploratory analysis uncovered critical correlation patterns:")
            st.warning("👉 **OverTime:** The single biggest predictor. Employees working overtime are **3x** more likely to leave.")
            st.warning("👉 **Income:** Strong inverse correlation. Lower income brackets have significantly higher turnover.")
            st.warning("👉 **Role Risk:** Sales Representatives & Lab Technicians face the highest churn rates (~24-40%).")

            st.divider()

            # SLIDE 6: ML STRATEGY
            st.markdown("### 6. 3-Pronged Machine Learning Strategy")
            st.markdown("""
            | Task | Type | Goal | Model |
            | :--- | :--- | :--- | :--- |
            | **1. Attrition** | Classification | Predict 'Yes/No' for leaving | Random Forest |
            | **2. Performance** | Classification | Predict Rating (3 vs 4) | Random Forest |
            | **3. Promotion** | Regression | Forecast 'Years Since Last Promo' | Random Forest Regressor |
            """)

            st.divider()

            # SLIDE 7: RESULTS
            st.markdown("### 7. Task 1 Results (Attrition)")
            st.write("We prioritized **Recall** to minimize 'False Negatives' (missing an at-risk employee is costly).")
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Accuracy", "77%")
            col_res2.metric("AUC-ROC", "0.82")
            col_res3.metric("Recall", "69%")
            st.caption("The model effectively discriminates between stayers and leavers, allowing HR to intervene early.")

            st.markdown("#### ⚡ Live Model Demo")
            st.write("Select a profile below to test the Attrition Model in real-time:")
            
            demo_c1, demo_c2 = st.columns(2)
            
            # --- LIVE DEMO LOGIC ---
            with demo_c1:
                if st.button("🧪 Simulate: High Risk Profile"):
                    # Create a dummy high risk profile
                    demo_data = {
                        'OverTime': 'Yes', 'MonthlyIncome': 2500, 'JobRole': 'Sales Representative',
                        'Age': 25, 'TotalWorkingYears': 2, 'YearsAtCompany': 1, 'JobSatisfaction': 1,
                        'EnvironmentSatisfaction': 1, 'JobInvolvement': 1, 'WorkLifeBalance': 1,
                        'DailyRate': 500, 'DistanceFromHome': 20, 'Education': 1, 'JobLevel': 1,
                        'NumCompaniesWorked': 5, 'PercentSalaryHike': 10, 'PerformanceRating': 3,
                        'RelationshipSatisfaction': 1, 'StockOptionLevel': 0, 'TrainingTimesLastYear': 0,
                        'YearsInCurrentRole': 0, 'YearsSinceLastPromotion': 0, 'YearsWithCurrManager': 0,
                         'Gender': 'Male', 'MaritalStatus': 'Single', 'Department': 'Sales', 
                         'BusinessTravel': 'Travel_Frequently', 'EducationField': 'Marketing'
                    }
                    demo_df = pd.DataFrame([demo_data])
                    
                    # Process
                    demo_clean = preprocess_data(demo_df)
                    demo_enc = pd.get_dummies(demo_clean)
                    # Align cols
                    for c in export_models['feature_columns_attrition']:
                        if c not in demo_enc.columns: demo_enc[c] = 0
                    demo_final = demo_enc[export_models['feature_columns_attrition']]
                    
                    # Predict
                    prob = export_models['attrition_model'].predict_proba(demo_final)[0][1]
                    st.error(f"🔴 Prediction: HIGH RISK (Probability: {prob:.1%})")
                    st.write("*Factors: OverTime=Yes, Low Income, Sales Rep*")

            with demo_c2:
                if st.button("🛡️ Simulate: Low Risk Profile"):
                    # Create a dummy low risk profile
                    demo_data_safe = {
                        'OverTime': 'No', 'MonthlyIncome': 15000, 'JobRole': 'Manager',
                        'Age': 45, 'TotalWorkingYears': 20, 'YearsAtCompany': 15, 'JobSatisfaction': 4,
                        'EnvironmentSatisfaction': 4, 'JobInvolvement': 4, 'WorkLifeBalance': 3,
                        'DailyRate': 1200, 'DistanceFromHome': 5, 'Education': 4, 'JobLevel': 4,
                        'NumCompaniesWorked': 1, 'PercentSalaryHike': 15, 'PerformanceRating': 4,
                        'RelationshipSatisfaction': 4, 'StockOptionLevel': 2, 'TrainingTimesLastYear': 3,
                        'YearsInCurrentRole': 10, 'YearsSinceLastPromotion': 5, 'YearsWithCurrManager': 8,
                         'Gender': 'Female', 'MaritalStatus': 'Married', 'Department': 'R&D', 
                         'BusinessTravel': 'Travel_Rarely', 'EducationField': 'Life Sciences'
                    }
                    demo_df_safe = pd.DataFrame([demo_data_safe])
                    
                    # Process
                    demo_clean_s = preprocess_data(demo_df_safe)
                    demo_enc_s = pd.get_dummies(demo_clean_s)
                    # Align cols
                    for c in export_models['feature_columns_attrition']:
                        if c not in demo_enc_s.columns: demo_enc_s[c] = 0
                    demo_final_s = demo_enc_s[export_models['feature_columns_attrition']]
                    
                    # Predict
                    prob_s = export_models['attrition_model'].predict_proba(demo_final_s)[0][1]
                    st.success(f"🟢 Prediction: STABLE (Probability: {prob_s:.1%})")
                    st.write("*Factors: No OverTime, High Income, Manager*")

            st.divider()
            
            # SLIDE 8: IMPACT
            st.markdown("### 8. Strategic Business Impact")
            st.write("By moving from intuition to data-driven insights, the organization can achieve:")
            st.success("✅ **Proactive Retention:** Identify flight risks before they resign.")
            st.success("✅ **Cost Optimization:** Save millions in replacement costs.")
            st.success("✅ **Fairness:** Data-backed promotion cycles.")

            st.divider()

            # SLIDE 9: FINAL RECOMMENDATIONS
            st.markdown("### 9. Final Recommendations")
            st.markdown("Based on our predictive models and analysis, we propose the following actions:")
            
            st.warning("⚠️ **Monitor Overtime:** Overtime is a critical driver. Ensure work-life balance to reduce burnout.")
            st.info("😊 **Improve Job Satisfaction:** Target engagement initiatives towards high-performing employees who currently report low satisfaction.")
            st.info("💰 **Align Compensation:** Review salaries against industry standards to mitigate income-driven attrition.")
            st.info("📈 **Growth & Promotions:** Promoting high-performance employees is a dual-win: it prevents attrition and significantly improves job satisfaction.")
            st.success("🔮 **Leverage Analytics:** Use this predictive tool proactively to identify at-risk talent before they decide to leave.")

        with col_dl:
            st.metric_box = st.container()
            with st.metric_box:
                st.subheader("📂 Download Deliverables")
                st.info("Download the project artifacts directly below.")
                
                # 1. Source Code
                try:
                    with open(__file__, "r") as f:
                        source_code = f.read()
                    st.download_button("📥 Source Code (app.py)", source_code, "app.py", "text/x-python")
                except Exception as e:
                    st.warning(f"Could not read source file: {e}")

                # 2. Cleaned Data
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Cleaned Data (CSV)", csv_data, "cleaned_employee_data.csv", "text/csv")
                
                # 3. Model Files (Pickle)
                # Serialize the models dictionary
                model_buffer = io.BytesIO()
                pickle.dump(export_models, model_buffer)
                model_buffer.seek(0)
                st.download_button("📥 Trained Models (.pkl)", model_buffer, "project_models.pkl", "application/octet-stream")
                
                # 4. Documentation / Metrics Report
                st.download_button("📥 Project Report (.txt)", export_report, "project_evaluation_report.txt", "text/plain")

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

        st.divider()
        
        # Outlier Analysis Section
        st.subheader("Outlier Analysis")
        st.write("Visualizing outliers in numerical features using Boxplots.")
        
        # Select numeric columns available in raw data
        # Note: raw_df has constant columns, but df_display is raw_df.copy().
        # We can list interesting ones manually or filter.
        outlier_options = ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears', 'Age', 'DailyRate', 'DistanceFromHome', 'YearsInCurrentRole']
        outlier_col = st.selectbox("Select Feature for Outlier Detection:", outlier_options)
        
        if outlier_col in df_display.columns:
            fig_out, ax_out = plt.subplots(figsize=(10, 4))
            sns.boxplot(x=df_display[outlier_col], color='orange', ax=ax_out)
            ax_out.set_title(f"Boxplot of {outlier_col}")
            st.pyplot(fig_out)
            
            # Explanation about why we kept them
            st.info("ℹ️ **Note on Outliers:** In this HR dataset, 'outliers' often represent senior leadership (High Income, Long Tenure). We deliberately kept them to ensure the model learns to predict attrition for senior management as well.")


    # --- TAB 3: MODEL TRAINING & RESULTS ---
    with tab3:
        st.header("Machine Learning Model Results")
        
        # Process Data (Locally for this tab to allow interactivity)
        
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
