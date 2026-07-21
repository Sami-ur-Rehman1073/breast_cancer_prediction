# 🩺 Breast Cancer Prediction using Machine Learning



*A complete end-to-end Machine Learning pipeline for Breast Cancer Diagnosis developed as part of an ML Internship Assignment.*



# 📌 Project Overview

Early diagnosis of breast cancer plays a vital role in improving patient survival rates and reducing treatment costs. The objective of this project is to build a robust Machine Learning pipeline capable of accurately classifying breast tumors as **Malignant** or **Benign** based on various diagnostic measurements.

This project follows an industry-standard Machine Learning workflow, including:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Scaling
- Class Imbalance Handling
- Machine Learning Pipeline
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Final Model Testing

The implementation strictly avoids data leakage by performing preprocessing and resampling inside a Scikit-Learn Pipeline, ensuring reliable model evaluation.

---

# 📂 Project Structure

```
breast_cancer_prediction/
│
├── data/
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── outputs/
│   └── models/
│       ├── preprocessor.pkl
│       ├── best_model.pkl
│       └── tuned_pipeline.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset Description

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

### Dataset Characteristics

| Property | Value |
|----------|------:|
| Samples | 569 |
| Features | 30 |
| Target Classes | 2 |
| Missing Values | None |
| Duplicate Rows | None |

### Target Classes

| Label | Meaning |
|-------|----------|
| 0 | Malignant |
| 1 | Benign |

### Features

The dataset contains 30 numerical features extracted from digitized images of breast cell nuclei, including:

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness
- Mean Compactness
- Mean Concavity
- Mean Concave Points
- Mean Symmetry
- Mean Fractal Dimension

along with their corresponding **standard error** and **worst-case** measurements.

---

# 🔍 Exploratory Data Analysis (EDA)

Several analyses were performed to understand the dataset before model development.

### ✔ Data Quality Checks

- No missing values
- No duplicate records
- All predictor variables are numerical
- Target variable is categorical

### ✔ Distribution Analysis

The following visualizations were generated:

- Histograms
- Feature-wise Boxplots
- Correlation Heatmap
- Target Class Distribution

### Key Findings

- Dataset contains no missing values.
- No duplicate observations were found.
- Multiple features contain outliers.
- Features have significantly different scales.
- Strong correlations exist between several variables.
- Class imbalance is present.
- Feature scaling is required before model training.

---

# ⚙️ Data Preprocessing

To ensure a clean and reproducible workflow, preprocessing was separated into its own stage.

The preprocessing pipeline includes:

- Train-Test Split
- Standard Scaling using `ColumnTransformer`
- Saving train/test datasets
- Saving fitted preprocessor using Joblib

The fitted preprocessor is reused during model training to maintain consistency.

---

# 🤖 Machine Learning Workflow

The project follows the following pipeline:

```
Raw Dataset
      │
      ▼
Train-Test Split
      │
      ▼
ColumnTransformer
(StandardScaler)
      │
      ▼
SMOTE
(Class Imbalance Handling)
      │
      ▼
Machine Learning Model
      │
      ▼
Cross Validation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Final Model
      │
      ▼
Testing on Unseen Data
```

---

# 🧠 Models Evaluated

Multiple Machine Learning algorithms were trained and compared.

Examples include:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost 

Each model was evaluated using **5-Fold Stratified Cross Validation**.

---

# 🔧 Hyperparameter Tuning

The best-performing model was further optimized using **GridSearchCV**.

Hyperparameter tuning was performed to maximize the ROC-AUC score while improving the model's overall generalization performance.

---

# 📈 Evaluation Metrics

The following metrics were used during model evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

# 🏆 Final Model Performance

After hyperparameter tuning, the final model achieved:

| Metric | Score |
|---------|-------|
| **Accuracy** | **97.37%** |
| **Precision** | **98.57%** |
| **Recall** | **97.18%** |
| **F1 Score** | **97.87%** |
| **ROC-AUC** | **99.61%** |

These results indicate excellent classification performance while maintaining strong precision and recall.

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- Joblib
- XGBoost 

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/breast_cancer_prediction.git
```

Move into the project directory

```bash
cd breast_cancer_prediction
```

Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

## Step 1

Run the Exploratory Data Analysis notebook

```
01_EDA.ipynb
```

---

## Step 2

Run the preprocessing notebook

```
02_preprocessing.ipynb
```

This will:

- Split the dataset
- Fit the preprocessor
- Save train/test data
- Save the preprocessing object

---

## Step 3

Run the modeling notebook

```
03_modeling.ipynb
```

This notebook will:

- Load the preprocessed data
- Build ML pipelines
- Handle class imbalance
- Perform cross validation
- Tune hyperparameters
- Evaluate models
- Test the best model

---

# 📊 Results Summary

The developed Machine Learning pipeline successfully classifies breast tumors with high accuracy while following best practices to avoid data leakage.

### Highlights

- End-to-End ML Pipeline
- Separate preprocessing stage
- Proper Train/Test split
- StandardScaler using ColumnTransformer
- SMOTE integrated inside the Pipeline
- 5-Fold Stratified Cross Validation
- Hyperparameter Tuning using GridSearchCV
- Excellent ROC-AUC performance (99.61%)
- Fully reproducible workflow

---

# 🚀 Future Improvements

Possible extensions to this project include:

- Feature Selection techniques
- Model Explainability using SHAP
- Model Explainability using LIME
- Calibration Curve Analysis
- Precision-Recall Curve
- Threshold Optimization
- Experiment Tracking using MLflow
- Docker containerization
- CI/CD pipeline using GitHub Actions
- Cloud deployment (AWS, Azure, or GCP)

---

# 🎯 Learning Outcomes

This project demonstrates practical experience in:

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Handling Class Imbalance
- Scikit-Learn Pipelines
- Cross Validation
- Hyperparameter Optimization
- Model Evaluation
- Machine Learning Best Practices
- Reproducible ML Workflows
