# Bank-Customer-Churn-Prediction-An-End-to-End-ML-Pipeline

## 📌 Project Overview
Customer acquisition is significantly more expensive than customer retention. This project builds an end-to-end Machine Learning pipeline to predict which bank customers are most likely to churn (leave the bank). By identifying at-risk customers, the bank can proactively deploy targeted retention campaigns, ultimately saving revenue.

This project covers the entire data science lifecycle: Data Preprocessing, Exploratory Data Analysis (EDA), Feature Engineering, predictive modeling with ensemble methods (XGBoost, LightGBM), Model Explainability (SHAP), and deployment via an interactive Streamlit web application.

## 📊 Dataset & Tech Stack
* **Dataset:** Bank Customer Churn Dataset (10,000 records, 14 attributes).
* **Target Variable:** `Exited` (1 = Churned, 0 = Stayed).
* **Tech Stack:** Python, Pandas, Matplotlib/Seaborn, Scikit-Learn, XGBoost, LightGBM, SHAP, Streamlit, Power BI.

---

## 🚀 Project Workflow & Key Findings

### 1. Data Cleaning & Preparation
* **Clean Dataset:** The dataset started with 10,000 records and 14 columns. I successfully identified that `RowNumber`, `CustomerId`, and `Surname` carry no predictive weight. 
* **Feature Refinement:** By dropping those, the dataset was streamlined to 11 highly predictive features. This reduced data noise and prevented the algorithms from learning false, overfit patterns based on unique customer IDs.

### 2. Exploratory Data Analysis (EDA)
Through visual analysis using Seaborn and Matplotlib, several critical behavioral trends were uncovered:
* **Demographics (Age):** Age is a massive factor. The age distribution for churners peaks significantly higher than non-churners, indicating that older customers are leaving the bank at a higher rate.
* **Geography:** Customers in Germany have a disproportionately higher churn rate compared to those in France and Spain.
* **Product Saturation:** Almost every single customer who holds 3 or 4 products with the bank churns. This suggests the bank's multi-product bundles are either poorly managed, too expensive, or frustrating to use.
* **Engagement:** Customers who are not "Active Members" are significantly more likely to leave.

### 3. Feature Engineering
Created custom business-logic features to improve model performance and capture human behavior:
* **Balance-to-Salary Ratio:** `Balance / EstimatedSalary`
* **Products-per-Tenure:** `NumOfProducts / (Tenure + 1)`
* **Age Grouping:** Binned continuous age data into distinct life-stage categories.

### 4. Machine Learning & Modeling
* **Dealing with Imbalance:** I correctly identified that *accuracy* is a misleading metric for this problem because only ~20% of customers churn. By utilizing class balancing techniques (like SMOTE and algorithm class weights) and focusing on **Recall** and **ROC-AUC**, the models were optimized to catch the actual churners rather than safely guessing "No Churn" every time.
* **Algorithm Performance:** Tree-based models (LightGBM, XGBoost, and Random Forest) significantly outperformed the Logistic Regression baseline. This proves that customer churn behavior is non-linear (e.g., age doesn't just increase churn evenly; it spikes at specific life stages).

### 5. Model Explainability (The SHAP Analysis)
Advanced models like LightGBM act as "black boxes." To make the model actionable for the business, I utilized SHAP (SHapley Additive exPlanations). The SHAP summary plot acts as the ultimate "Business Insights" generator, proving to stakeholders exactly *why* the AI makes its predictions:
* **The Top Drivers:** The top 3 drivers of churn across the entire bank are Age, Number of Products, and IsActiveMember.
* **Directional Impact:** The SHAP analysis explicitly proves that **high age** pushes the model to predict churn, **low activity** pushes toward churn, and a **high number of products** pushes toward churn.

---

## 📂 Deliverables

1. **`CustomerChurnEndToEnd.ipynb`**: Fully documented Jupyter Notebook containing the data cleaning, feature engineering, ML pipeline, and SHAP analysis.
2. **Streamlit Web App (`app.py`)**: An interactive tool allowing stakeholders to input a customer's profile and receive a real-time Churn Probability and Risk Level.
3. **Power BI Dashboard**: An executive summary dashboard visualizing total churn rates, high-risk segments, and geographic revenue at risk.
   https://app.powerbi.com/links/FUARUkjg6F?ctid=e202cd47-7a56-4baa-99e3-e3b71a7c77dd&pbi_source=linkShare

*(Optional: Insert a GIF or Screenshot of your Power BI dashboard here)*

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/bank-customer-churn.git](https://github.com/yourusername/bank-customer-churn.git)
