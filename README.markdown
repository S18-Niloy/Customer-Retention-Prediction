# Customer Retention Prediction

## Project Overview
This project aims to predict customer churn (whether a customer will stop doing business with a company) using machine learning and deep learning techniques. The analysis involves exploratory data analysis (EDA), feature engineering, and training multiple predictive models, including Random Forest, XGBoost, LightGBM, and an Artificial Neural Network (ANN). The goal is to identify the best-performing model based on metrics like accuracy, precision, recall, F1-score, and AUC (Area Under the ROC Curve).

## Dataset Description
The dataset (`dataset.csv`) contains 1,000 customer records with 15 features:
- **Columns**: 
  - `Customer_ID`: Unique identifier for each customer.
  - `Age`: Customer's age.
  - `Gender`: Customer's gender (e.g., Male, Other).
  - `Annual_Income`: Customer's annual income.
  - `Total_Spend`: Total amount spent by the customer.
  - `Years_as_Customer`: Duration of customer relationship.
  - `Num_of_Purchases`: Number of purchases made.
  - `Average_Transaction_Amount`: Average transaction value.
  - `Num_of_Returns`: Number of returns made.
  - `Num_of_Support_Contacts`: Number of times customer contacted support.
  - `Satisfaction_Score`: Customer satisfaction score.
  - `Last_Purchase_Days_Ago`: Days since last purchase.
  - `Email_Opt_In`: Whether customer opted into email marketing (True/False).
  - `Promotion_Response`: Customer's response to promotions (e.g., Responded, Ignored, Unsubscribed).
  - `Target_Churn`: Target variable (True/False) indicating if the customer churned.
- **Shape**: (1000, 15).
- **Missing Values**: None observed.
- **Duplicates**: None observed.
- **Target Distribution**: Visualized using a countplot to show the balance of churned vs. non-churned customers.
- **Correlations**: Explored via a heatmap for numeric features.

## Setup Instructions
To run the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**:
   Ensure Python 3 is installed. Install required libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm tensorflow
   ```

3. **Prepare the Dataset**:
   Place the `dataset.csv` file in the project directory or update the file path in the script (`data = pd.read_csv("/content/dataset.csv")`) to point to the correct location.

4. **Run the Script**:
   Execute the Python script:
   ```bash
   python customer_retention_prediction.py
   ```
   Alternatively, run it in a Jupyter Notebook environment (e.g., Google Colab, JupyterLab) by converting the `.py` file to `.ipynb` or running it cell-by-cell.

5. **Environment**:
   - Recommended: Use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
   - The script was originally run in Google Colab, but it is compatible with local Python environments.

## Key Results
- **Data Preprocessing**:
  - Categorical features (e.g., `Gender`, `Promotion_Response`) were encoded using `LabelEncoder`.
  - Numeric features were capped for outliers (1st and 99th percentiles).
  - New features were engineered: ratios, sums, differences, and binary flags based on numeric columns.
  - Highly correlated features (>0.9) were dropped to reduce multicollinearity.
  - Top 20 features were selected based on combined Mutual Information and Random Forest importance scores.
  - Data was scaled using `StandardScaler` and split into train (70%), validation (15%), and test (15%) sets.
  - Class imbalance was addressed using SMOTE for oversampling the training set.

- **Models Trained**:
  - Random Forest (`n_estimators=200`)
  - XGBoost (`eval_metric='logloss'`)
  - LightGBM
  - ANN (128 neurons → Dropout → 64 neurons → Dropout → 1 neuron with sigmoid activation)

- **Evaluation Metrics**:
  - Models were evaluated on accuracy, precision, recall, F1-score, and AUC.
  - Visualizations included confusion matrices and ROC curves for each model.
  - A bar plot compared model performance across Accuracy, F1, and AUC.

- **Best Model**:
  - **ANN** achieved the highest AUC of **0.495**.
  - **Note**: An AUC of ~0.5 indicates near-random performance, suggesting potential issues like insufficient data, poor feature quality, or suboptimal model tuning. Further investigation (e.g., hyperparameter tuning, feature engineering, or additional data) is recommended to improve performance.