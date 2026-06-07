# Health Insurance Premium Cost Predictor 🏥💰

An interactive machine learning web application that predicts personalized health insurance premiums by leveraging user demographics, BMI, smoking status, and medical history.Built using **XGBoost** and deployed via **Streamlit**, this project features age-based model segmentation to deliver highly accurate, real-time premium predictions.

## 🔗 Live Links & Demos
* **Live Web App:** [Healthcare Premium Prediction on Streamlit](https://premium-insurance-prediction-harini.streamlit.app/)
* **Repository:** `https://github.com/your-username/healthcare-premium-prediction`

---

## 🛠️ Core Capabilities
This project implements a production-grade, end-to-end machine learning pipeline engineered for personalized risk-pricing in the insurance domain:
* **Dynamic Web Dashboard:** A responsive Streamlit front-end designed to capture demographic, financial, and multi-modal health telemetry.
* **Algorithmic Segmentation:** Dual-routed XGBoost architectures optimized independently for distinct age cohorts to capture non-linear risk behaviors.
* **Production-Ready Preprocessing:** Robust Scikit-learn pipelines that handle feature encoding and scaling without data leakage.
* **Low-Latency Inference:** Serialized model and scaler binaries optimized via `joblib` for instantaneous, real-time premium calculation.

---

## 🔄 Execution Pipeline
1. **Feature Ingestion:** The client inputs personal health attributes (Age, BMI, Smoking Frequency, and Regional Data) via the interactive interface.
2. **Payload Processing:** The front-end packages the data into a structured payload and forwards it to the `prediction_helper.py` orchestration engine.
3. **Deterministic Routing:** The execution pipeline applies a hard decision boundary based on the applicant's age to prevent structural underfitting:
   * **If Age $\le$ 25** $\rightarrow$ Routes to `model_young.joblib` & `scaler_young.joblib` *(optimized for early-adulthood risk profiles)*.
   * **If Age $>$ 25** $\rightarrow$ Routes to `model_rest.joblib` & `scaler_rest.joblib` *(optimized for general adult demographics)*.
4. **Inference Delivery:** The targeted XGBoost model computes the personalized premium and outputs the actuarial estimate to the UI in real time.

---

## 🎯 Value Proposition
As the insurance industry shifts away from rigid lookup tables toward data-driven, individualized pricing, this framework highlights crucial MLOps principles:
* **Stratified Risk Modeling:** Demonstrates how population segmentation resolves underfitting caused by structural breaks in data distributions.
* **Seamless Deployment:** Transitions smoothly from exploratory Jupyter Notebooks to a live, cloud-hosted web environment.
* **Clean Architecture:** Balances predictive complexity (XGBoost) with decoupled, highly maintainable engineering patterns.
---

## 📊 Visual Insights

Below are key insights and visualizations derived during model development. These plots support feature selection, model explainability, and validation across age segments.

| 📸 Screenshot | 🔍 Description |
| :--- | :--- |
| <img src="desktop/img/1.png" width="500" alt="Age & Dependents"> | Feature distributions for age and dependents show high concentration in young cohorts (18–25), discrete peaking at 0, and clear skewness. |
| <img src="desktop/img/2.png" width="500" alt="Income & Premium"> | Feature distributions for income and premium show bimodal skew, an abrupt drop past 40 Lakhs, and clear transformation needs for the target metric. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.29.21 PM.jpg" width="500" alt="Correlation Heatmap"> | Correlation heatmap matrix isolating linear dependencies, highlighting a critical 0.91 redundancy between income scales and plan premium drivers. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.38.39 PM.jpg" width="500" alt="Scatter Plots"> | Bivariate scatter plots mapping feature anomalies against premium tiers, confirming non-linear step-functions across age and income cuts. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.45.53 PM.png" width="500" alt="Demographic Percentages"> | Categorical percentage charts validating uniform sample balance across gender, unmarried statuses, and heavy regional clusters in the South. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.46.17 PM.png" width="500" alt="Lifestyle Percentages"> | Lifestyle distribution shares highlighting concentration spreads for normal BMI segments, non-smokers, and salaried professional types. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.46.45 PM.png" width="500" alt="Linear Regression Importance"> | Linear regression feature coefficients prioritizing insurance plan and age metrics as primary positive cost weight factors. |
| <img src="desktop/img/Screenshot 2026-06-07 at 7.46.54 PM.png" width="500" alt="XGBoost Feature Importance"> | XGBoost feature significance profiles isolating structural variance, focusing heavily on explicit policy choices and age cohort breaks. |
| <img src="desktop/img/Screenshot 2026-06-07 at 8.42.03 PM.png" width="500" alt="Distribution of Residuals"> | Residual distribution plot visualizes percentage errors (Diff PCT)—confirming a well-calibrated, zero-centered error model with minimal variance. |

## 🛠️ Technologies Used

* **Python (v3.9+):** Core programming environment for data orchestration and modeling.
* **Streamlit:** Interactive web application framework powering the user interface.
* **Scikit-learn:** Data preprocessing, standard scaling, and pipeline evaluation.
* **XGBoost:** Gradient-boosted decision tree architecture optimized for premium forecasting.
* **Pandas & NumPy:** High-performance matrices and vector alignment for engineering inputs.
* **Joblib:** Lightweight serialization for loading low-latency production model binaries.
