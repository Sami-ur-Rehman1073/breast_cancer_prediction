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

# 📁 Project Structure

```
week1_breast_cancer_prediction/
│
├── data/
│   └── Contains the dataset used for training and evaluation.
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
│
├── src/
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── train.py           # Model training pipeline
│   └── evaluate.py        # Model evaluation functions
│
├── outputs/
│   ├── figures/           # Generated visualizations and plots
│   └── models/            # Saved preprocessing objects and trained models
│
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── .gitignore             # Files ignored by Git
```

### Directory Description

| Directory/File | Description |
|----------------|-------------|
| **data/** | Stores the Breast Cancer Wisconsin Diagnostic dataset used in this project. |
| **notebooks/** | Contains the complete machine learning workflow, including Exploratory Data Analysis, Data Preprocessing, and Model Development. |
| **src/** | Python scripts containing reusable preprocessing, training, and evaluation code. |
| **outputs/figures/** | Stores plots generated during EDA and model evaluation such as histograms, boxplots, heatmaps, ROC curves, and confusion matrices. |
| **outputs/models/** | Stores serialized preprocessing objects and trained machine learning models for future inference. |
| **requirements.txt** | Lists all required Python packages to reproduce the project. |
| **README.md** | Documentation explaining the project, setup instructions, workflow, and results. |
| **.gitignore** | Specifies files and directories excluded from version control. |

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


---

# 🏅 Model Selection

Several Machine Learning algorithms were trained and evaluated to identify the most suitable model for breast cancer classification. Each model was assessed using **5-Fold Stratified Cross Validation** to ensure reliable and unbiased performance estimation.

The following models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost

The evaluation was based on the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

After comparing the cross-validation results, **Logistic Regression** was selected as the final model for hyperparameter tuning.

## Why Logistic Regression?

Logistic Regression consistently demonstrated the best overall performance across all evaluation metrics while maintaining excellent generalization on unseen data.

The model was selected for the following reasons:

- In the medical field, recall is the most important metric to be considerd. Logistic Regression had the highest recall amoung the models.

- The training and testing metrics of each model were compared, and in case of logistic regression, the gap between the training and testing metrics was lowest. It indicates that there was no overfitting.

- The performance of logistic regression was satisfactory if we consider other metrics like accuarcy, precision, f1 score etc.

---



# 🔧 Hyperparameter Tuning

The best-performing model was further optimized using **GridSearchCV**.

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
|---------|-------:|
| **Accuracy** | **97.37%** |
| **Precision** | **98.57%** |
| **Recall** | **97.18%** |
| **F1-Score** | **97.87%** |
| **ROC-AUC Score** | **99.61%** |

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
