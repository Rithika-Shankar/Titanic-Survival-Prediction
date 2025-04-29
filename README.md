# Titanic Survivability Prediction

This project explores multiple classification models to predict passenger survival from the Titanic disaster using the cleaned and engineered dataset (`tested.csv`). The models include Random Forest, Gradient Boosting, XGBoost, Logistic Regression, and Support Vector Machine.

---

## Task Objectives

- Preprocess and engineer new features from the Titanic dataset.
- Create a reusable data transformation pipeline.
- Train and evaluate multiple machine learning models.
- Compare models based on various metrics (accuracy, precision, recall, F1 score, ROC AUC).
- Use feature importance to interpret top-performing models.

---

## Steps to Run the Project

### 1. Clone the repository or open in Colab

You can copy the code or run directly using Google Colab.

### 2. Install required packages

Ensure the following libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 3. Prepare dataset

Place the `tested.csv` file (Titanic dataset with test data) in the same directory as the script.

### 4. Run the script

You can run the full notebook or script to execute the pipeline:
- Feature Engineering
- Preprocessing
- Model Training
- Evaluation
- Comparison of models

---

## Code Structure

- **Data Loading**: Reads the Titanic dataset from `tested.csv`.
- **Feature Engineering**: Adds features like `Title`, `IsAlone`, `AgeBin`, `FareBin`, etc.
- **Preprocessing Pipeline**: Uses `ColumnTransformer` with numerical and categorical pipelines.
- **Model Training**: Trains five classifiers using `Pipeline` and `SelectKBest` for feature selection.
- **Evaluation**: Calculates accuracy, precision, recall, F1 score, ROC AUC, and cross-validation scores.
- **Feature Importance**: Displays feature importance for tree-based models.

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Cross-Validation Accuracy

A final summary DataFrame ranks models by F1 Score and other metrics.

---

## Clean Code Practices Followed

- Functions are modular (`process_features`, `build_preprocessor`, `evaluate`, etc.).
- Code follows a pipeline architecture (`Pipeline`, `ColumnTransformer`).
- Avoids hardcoding by using variables and programmatic feature selection.
- Clear separation of data preparation, model building, and evaluation stages.

---

## Example Output

After execution, a summary table is printed:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC | CV Mean Acc | CV Std |
|---------------------|----------|-----------|--------|----------|---------|--------------|--------|
| Random Forest        | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    | 1.00         | 0.00   |
| Gradient Boosting    | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    | 1.00         | 0.00   |
| XGBoost              | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    | 1.00         | 0.00   |
| Logistic Regression  | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    | 1.00         | 0.00   |
| SVM                  | 0.96     | 0.97      | 0.94   | 0.95     | 0.99    | 0.96         | 0.03   |

---

## Note

- Ensure you provide the correct path to the `tested.csv` dataset.
- Some classifiers (e.g., SVM) do not support `.feature_importances_`, so feature importance may not be displayed for them.
