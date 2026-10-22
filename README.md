# Machine Learning in Detecting Financial Fraud

## 1. Project Overview

This project develops a machine learning prototype to identify potentially fraudulent financial transactions using classification algorithms. It compares Logistic Regression, Decision Tree, and Random Forest models to evaluate their ability to detect suspicious transactions.

## 2. Problem Statement

Financial fraud can cause significant financial losses to individuals and organizations. Traditional detection methods may struggle to identify complex transaction patterns. Machine learning can help detect suspicious activities by learning patterns from transaction data and classifying transactions as legitimate or potentially fraudulent.

## 3. Objectives

- Prepare and preprocess financial transaction data.
- Explore the dataset using statistical analysis and visualization.
- Handle class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
- Train Logistic Regression, Decision Tree, and Random Forest classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Compare model performance and identify the best-performing model.
- Maintain project files, dependencies, and documentation for reproducibility.

## 4. Technologies Used

- **Programming language:** Python
- **Development environment:** Google Colab / Jupyter Notebook
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Imbalanced data handling:** Imbalanced-learn (SMOTE)
- **Version control:** Git and GitHub

## 5. Dataset Description

The project uses a synthetic, Kaggle-style financial transaction dataset containing 1,000 records. It is intended for educational experimentation and does not represent verified real-world financial transactions.

The dataset contains transaction-related features used for classification. The target variable, `FraudIndicator`, represents:

- `0` — Legitimate transaction
- `1` — Potentially fraudulent transaction

The dataset contains 955 legitimate transactions and 45 fraudulent transactions, making it an imbalanced classification problem.

## 6. Methodology

The project follows these steps:

1. Load and inspect the dataset.
2. Perform data cleaning and preprocessing.
3. Conduct exploratory data analysis.
4. Separate input features and target labels.
5. Split the data into training and testing sets using an 80:20 ratio.
6. Apply SMOTE to the training data to address class imbalance.
7. Train Logistic Regression, Decision Tree, and Random Forest models.
8. Evaluate the models on the test dataset.
9. Compare performance metrics and save the results.
10. Document the project and maintain version history using GitHub.

SMOTE is applied only to the training data to avoid resampling the test data.

## 7. Models Implemented

### Logistic Regression
A baseline classification algorithm used to estimate whether a transaction is fraudulent.

### Decision Tree
A tree-based classification algorithm that learns decision rules from transaction features.

### Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

## 8. Project Structure

```text
Fraud-Detection-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── fraud_detection_kaggle_style_1000_rows.csv
├── docs/
│   └── README.md
├── notebooks/
│   ├── Experiment_6_Financial_Fraud_Detection.ipynb
│   └── README.md
├── results/
│   ├── README.md
│   └── experiment6_model_comparison.csv
├── src/
│   ├── main.py
│   └── README.md
└── tests/
    └── README.md
```

## 9. Installation and Execution

### Prerequisites

- Python 3.9 or later
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/pratiksha-jadhav01/Fraud-Detection-ML.git
cd Fraud-Detection-ML
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Python prototype

```bash
python src/main.py
```

The program loads the dataset, preprocesses the data, applies SMOTE to the training data, trains the classification models, evaluates their performance, and saves the model comparison results in the `results/` directory.

### Step 4: Run the research notebook

Open `notebooks/Experiment_6_Financial_Fraud_Detection.ipynb` in Google Colab or Jupyter Notebook. Ensure that the dataset path is correct, install the required dependencies if necessary, and execute the notebook cells in order.

## 10. Experimental Results

The prototype was executed in Google Colab after cloning the GitHub repository and installing the required dependencies.

The following results were obtained during the recorded execution:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.50% | 11.29% | 77.78% | 19.72% | 0.7923 |
| Decision Tree | 90.00% | 26.09% | 66.67% | 37.50% | 0.7888 |
| Random Forest | 97.00% | 66.67% | 66.67% | 66.67% | 0.8287 |

Random Forest achieved the highest accuracy, F1-score, and ROC-AUC in this experiment. Logistic Regression achieved the highest recall but produced more false-positive predictions.

These results are specific to the current dataset and execution. The test set contains only nine fraudulent transactions, so the evaluation metrics should be interpreted cautiously.

## 11. Output

The prototype displays:

- Dataset dimensions
- Training dataset size before and after SMOTE
- Testing dataset size
- Classification performance metrics for each model
- The model with the highest F1-score
- The location where the model comparison results are saved

The model comparison output is saved as:

`results/experiment6_model_comparison.csv`

## 12. Version Control

Git and GitHub are used to track project development and changes.

Version control supports:

- Tracking modifications to source code and documentation
- Maintaining a history of project development
- Identifying changes between commits
- Sharing the project repository
- Marking a project milestone through the `v1.0` release

Repository: https://github.com/pratiksha-jadhav01/Fraud-Detection-ML

Releases: https://github.com/pratiksha-jadhav01/Fraud-Detection-ML/releases

## 13. Reproducibility

The project includes a `requirements.txt` file to document its Python dependencies. The repository can be cloned into a fresh environment, the dependencies installed, and the prototype executed using the commands provided above.

The project was tested in a fresh Google Colab session by cloning the repository, installing dependencies, and running the prototype successfully.

Exact results may vary with software versions, random seeds, and execution settings. The synthetic dataset is included in the repository for the documented workflow.

## 14. Limitations

- The dataset is synthetic and may not reflect real-world fraud patterns.
- The dataset contains relatively few fraudulent transactions.
- Evaluation metrics can be sensitive to the small number of fraud cases in the test set.
- A high accuracy score does not necessarily mean that every fraudulent transaction is detected.
- The prototype is intended for academic demonstration and is not ready for deployment in a real financial system.

## 15. Future Scope

- Evaluate the system using verified real-world financial transaction datasets.
- Test additional machine learning and anomaly detection algorithms.
- Improve model validation using cross-validation and repeated experiments.
- Tune classification thresholds to balance fraud detection and false positives.
- Develop a simple user interface for transaction risk prediction.
- Add model explainability and improved monitoring.

## 16. Conclusion

This project demonstrates the application of machine learning classification algorithms to financial fraud detection. Logistic Regression, Decision Tree, and Random Forest were trained and evaluated using a synthetic transaction dataset. Random Forest achieved the strongest overall results in the recorded experiment based on accuracy, F1-score, and ROC-AUC.

The project also demonstrates practical research workflow elements, including preprocessing, class imbalance handling, model evaluation, technical documentation, GitHub version control, and execution in a fresh environment.

## 17. References

1. Scikit-learn documentation: https://scikit-learn.org/
2. Imbalanced-learn documentation: https://imbalanced-learn.org/
3. Pandas documentation: https://pandas.pydata.org/docs/
4. NumPy documentation: https://numpy.org/doc/
5. Matplotlib documentation: https://matplotlib.org/stable/
6. Seaborn documentation: https://seaborn.pydata.org/

## Author

T.E. Computer Science and Engineering (Data Science) Student
