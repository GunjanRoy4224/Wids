# Equity Classification using Machine Learning (Winter in Data Science (WiDS))

This project contains an end-to-end machine learning pipeline aimed at solving a fundamental data science problem: **Equity Classification using ML**. 

The goal is to build a systematic, fundamentally-driven stock selection model that predicts whether a stock will outperform a market benchmark (Investable) or underperform (Non-Investable), prioritizing the minimization of False Positives to avoid direct financial loss.

## Repository Structure

The repository is structured into progressive assignments that build up to the final optimized model:

- **Assignment 1: Fundamentals**
  - Explores basic financial metrics and identifies key indicators.
  - Contains: `Fundamentals.ipynb` and financial metric datasets (`.csv`).

- **Assignment 2: Problem Formulation**
  - Outlines the business problem, prediction target, and evaluation criteria.
  - Contains: `Problem formulation.pdf`.

- **Assignment 3: EDA & Feature Engineering**
  - Focuses on exploratory data analysis, handling extreme financial outliers, and engineering 13 core financial ratios (e.g., ROE, EV-to-Sales) to mitigate size-based multicollinearity.
  - Contains: `WIDS_Equity_Classification.ipynb`, `raw_dataset.csv`, `engineered_features.csv`.

- **Assignment 4: Model Building & Optimization**
  - Implements the complete ML pipeline including target encoding, log transformations, and feature selection. Evaluates multiple algorithms (Logistic Regression, Random Forest, SVM, Decision Tree, XGBoost) and performs hyperparameter tuning on the best model.
  - Contains: `Assignment_4_Model_Building.ipynb`.

- **Final Report**
  - Contains: `Final_Project_Report.docx` - A professionally formatted executive summary covering the entire project workflow, business insights, and final model evaluation.

## Key Findings & Results
- **Best Model:** XGBoost (Tuned) achieved 91% Accuracy, 82% Precision, 84% Recall, and an F1 Score of 83% on unseen data.
- **Top Predictors:** Return on Equity (ROE), Price-to-Earnings (P/E) Ratio, and Industry Group emerged as the strongest indicators of stock outperformance.

## Getting Started

To run the notebooks in this repository, you will need Python installed along with Jupyter and the necessary data science libraries.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd WIDS
   ```

2. **Install dependencies:**
   It is recommended to create a virtual environment first.
   ```bash
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn xgboost category_encoders
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Navigate to the respective assignment folder and open the `.ipynb` files to view or run the code.
