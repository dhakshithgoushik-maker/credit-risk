# credit-risk
# Credit Risk Modeling with XGBoost, SHAP, and LIME
# Project Overview
This project demonstrates how to build and interpret a sophisticated credit risk prediction model using XGBoost, alongside SHAP and LIME for model interpretability. You’ll learn not just to optimize for accuracy, but to deliver **transparent and actionable explanations** for a business audience in the financial sector.

# Dataset
Columns expected in the dataset:
- customer_id
- customer_age
- customer_income
- home_ownership
- employment_duration
- loan_intent
- loan_grade
- loan_amnt
- loan_int_rate
- term_years
- historical_default
- cred_hist_length
- Current_loan_status

# Steps
1. **Data Preprocessing**: Clean and encode categorical data, split into features and target.
2. **Model Training**: Train and tune an XGBoost model to optimize AUC/balanced accuracy.
3. **Global Interpretability (SHAP)**: Generate SHAP summary plots to document top 10 feature importances driving default risk.
4. **Local Interpretability (SHAP & LIME)**: For three loan applicants (high, borderline, low risk), use SHAP force plots and LIME HTML reports to explain individual predictions.
5. **SHAP vs LIME Comparison**: Analyze and contrast local explanations to uncover complementarities and divergences.
6. **Policy Recommendations**: Formulate three actionable lending strategies based on model findings.

# Deliverables
- **Python code** (`credit_risk_interpretability.py`): End-to-end workflow, model training, SHAP/LIME interpretation, textual outputs.
- **Performance Report**: Key metrics (AUC, balanced accuracy, classification report) and summary of global SHAP rankings.
- **Analysis Section**: Comparison of SHAP vs LIME explanations for three selected cases.
- **Strategic Summary**: Three robust lending policy recommendations leveraging interpretable ML insights.

# How to Run
1. Install dependencies:
   pip install xgboost shap lime scikit-learn matplotlib pandas
2. Place your dataset as `credit_risk_data.csv` in the project directory.
3. Run the code:
   python credit_risk_interpretability.py
4. Examine outputs:
    - Performance metrics and print statements in terminal
    - SHAP summary and force plots (saved as PNG)
    - LIME local explanations (saved as HTML)
    - Policy recommendations printout


- Ensure your data has no missing values and correct formatting for categorical variables.
- For regulated industries, use both SHAP and LIME for transparency in model-based decisions.
- The analysis can be enhanced further by visualizing more local/global explanations or by examining feature importance stability over time.
