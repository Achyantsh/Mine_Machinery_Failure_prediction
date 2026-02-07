# Predictive Maintenance for Mining Equipment
### *Leveraging Machine Learning to Anticipate Failures Before They Happen*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/View-Documentation-b31b1b?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./Machinery_failure_prediction_Docmentation.pdf)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

---

## Overview

In the mining industry, equipment failure is not just an inconvenience—it's a massive financial and safety risk. This project implements a **Machine Learning-based Predictive Maintenance System** designed to forecast machinery breakdowns using sensor data.

By analyzing operational metrics like torque, temperature, and rotational speed, our model predicts potential failures, enabling engineers to perform maintenance proactively rather than reactively.

> **Detailed Analysis:** For a comprehensive breakdown of the theoretical framework, mathematical models, and exploratory data analysis, please refer to the [Project Documentation](https://drive.google.com/file/d/1GeIWA-GeZizUaME6hEJM_gGgcA1KGhCm/view?usp=sharing).
> ###  **Try the Live Demo:** [Mining Failure Predictor · Streamlit](https://mine-machinery-failure-prediction.streamlit.app/)

---

## The Dataset

We utilized the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository. It reflects real-world industrial data with **10,000 data points**.

| Feature | Unit | Description |
| :--- | :--- | :--- |
| **Air Temperature** | [K] | Ambient environmental temperature |
| **Process Temperature** | [K] | Internal machine temperature |
| **Rotational Speed** | [rpm] | Speed of moving parts |
| **Torque** | [Nm] | Mechanical force on the shaft |
| **Tool Wear** | [min] | Cumulative usage time of the tool |
| **Type** | L/M/H | Quality variant of the machine |

**Target Variable:** `Machine failure` (Binary: 0 = No Failure, 1 = Failure)

---

## Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, RandomForest
* **Deployment:** Streamlit
* **Model Interpretation:** Feature Importance & Confusion Matrices

---

## Methodology

1.  **Preprocessing:**
    * Dropped irrelevant identifiers (`UDI`, `Product ID`).
    * Encoded categorical variables (`Type`) using Label Encoding.
    * Scaled continuous features using `StandardScaler`.
2.  **Handling Imbalance:** Stratified train-test splits were used to handle the 96:4 class imbalance.
3.  **Model Training:**
    * **XGBoost Classifier** (Gradient Boosting)
    * **Random Forest Classifier** (Ensemble Bagging)
4.  **Evaluation:** Models were benchmarked using Accuracy, F1-Score, and ROC-AUC.

---

## Model Performance

We achieved state-of-the-art results, with **XGBoost** slightly outperforming Random Forest in terms of stability and precision.

| Metric | XGBoost | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | **98.9%** | 98.4% |
| **ROC-AUC** | **0.96** | 0.97 |
| **Precision** | 0.91 | 0.93 |
| **Recall** | 0.75 | 0.57 |
| **F1-Score** | **0.82** | 0.71 |

### Detailed Evaluation
While the table above highlights the key performance indicators, the nuances of the model's performance—including confusion matrices, error analysis on false negatives, and feature importance rank plotting—are critical for understanding the model's reliability in a mining environment.

 --> **[Click here to view the full Evaluation & Error Analysis in the Documentation](https://drive.google.com/file/d/1GeIWA-GeZizUaME6hEJM_gGgcA1KGhCm/view?usp=sharing)**

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/mining-predictive-maintenance.git](https://github.com/yourusername/mining-predictive-maintenance.git)
cd mining-predictive-maintenance
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the training notebook

### 4. Launch the web app
```bash
streamlit run app.py
```
---

## Visuals

### Feature Importance
*Torque and Rotational Speed were identified as the most critical indicators of failure.*

![Feature Importance Plot](images/feature_importance.png)
### Confusion Matrix
*The model successfully identified the majority of failure cases with minimal false positives.*

![Confusion Matrix](images/confusion_matrix.png)
### Streamlit Dashboard
*A user-friendly web interface for real-time predictions.*

![Streamlit App Interface](images/streamlit_app_screenshot.png)

---

## Future Scope
* **Real-time IoT Integration:** Connect model to live sensor feeds.
* **Deep Learning:** Experiment with LSTMs for time-series forecasting.
* **Mobile App:** Develop a mobile alert system for field engineers.

---

## Author

**Achyant Shrivastava**
*Department of Mining Engineering, IIT (BHU), Varanasi*

---

*“Predicting the future is hard, but preventing a failure is smart.”*
