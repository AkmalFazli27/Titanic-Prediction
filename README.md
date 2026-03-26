# Titanic Survival Prediction

End-to-end machine learning project to predict passenger survival on the Titanic using the Kaggle Titanic dataset.

## Project Structure

```text
Titanic-Prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── gender_submission.csv
│   └── submission.csv
├── model/
└── notebook/
   └── titanic.ipynb
```

- [notebook/titanic.ipynb](notebook/titanic.ipynb) — full EDA, preprocessing, modeling, CV, tuning, submission, and model export
- [data/train.csv](data/train.csv) — training data
- [data/test.csv](data/test.csv) — test data
- [data/gender_submission.csv](data/gender_submission.csv) — sample submission
- [data/submission.csv](data/submission.csv) — generated predictions
- [model/](model/) — exported model artifacts
- [requirements.txt](requirements.txt) — Python dependencies

## Workflow Summary

1. **EDA**
   - Target distribution (`Survived`) and feature visualizations.
2. **Preprocessing**
   - Missing value handling:
     - `Age`: median by `Sex` and `Pclass`
     - `Cabin`: dropped (high missingness)
     - `Embarked`: drop missing rows in train
     - `Fare`: median fill in test
3. **Feature Engineering**
   - `FamilySize`, `IsAlone`
   - `Title` extracted from `Name`
   - One-hot encoding for `Sex`, `Embarked`, and `Title`
   - Drop `Name` and `Ticket`
4. **Modeling**
   - Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, SVM, KNN, Naive Bayes
   - Pipelines with scaling where required
5. **Validation**
   - Train/validation split + metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Stratified 5-fold cross-validation
6. **Hyperparameter Tuning**
   - `GridSearchCV` on Gradient Boosting pipeline
7. **Export**
   - Submission CSV written to [data/submission.csv](data/submission.csv)
   - Tuned model exported to [model/](model/)

## Key Results

- Best validation Accuracy in model comparison: **SVM (~0.837)**
- Best CV mean Accuracy: **Gradient Boosting (~0.838)**
- Tuned Gradient Boosting best CV Accuracy: **~0.8437**

## How to Run

1. Install dependencies from [requirements.txt](requirements.txt).
2. Open [notebook/titanic.ipynb](notebook/titanic.ipynb) in VS Code/Jupyter.
3. Run cells from top to bottom.
4. Outputs generated:
   - Predictions: [data/submission.csv](data/submission.csv)
   - Trained model (`.pkl`) in [model/](model/)

## Tech Stack

- Python (Jupyter Notebook)
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- joblib