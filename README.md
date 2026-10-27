# Five Classifiers for Diabetes Risk Assessment in Young Adults

An end-to-end machine learning study and interactive cloud dashboard comparing five classification algorithms—**Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**—for prediabetes and diabetes risk screening in young adults aged 18–34.

Developed for **Cloud Computing Lab (Question 4)** by:
- **Sachin Kumar**
- **Shreyansh Jugran**
- **Suryansh Bhargava**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Technology Stack](#technology-stack)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
  - [Source and Scope](#source-and-scope)
  - [Severe Class Imbalance](#severe-class-imbalance)
  - [Preprocessing Pipeline and Leakage Prevention](#preprocessing-pipeline-and-leakage-prevention)
- [Models and Hyperparameter Tuning](#models-and-hyperparameter-tuning)
- [Empirical Results and Performance Analysis](#empirical-results-and-performance-analysis)
  - [Held-Out Test Set Performance](#held-out-test-set-performance)
  - [Cross-Validation Performance](#cross-validation-performance)
  - [Training and Inference Latency](#training-and-inference-latency)
  - [Why Recall and ROC-AUC Matter](#why-recall-and-roc-auc-matter)
  - [Feature Importance Highlights](#feature-importance-highlights)
- [Interactive Dashboard](#interactive-dashboard)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Environment Variables](#environment-variables)
- [Cloud Deployment](#cloud-deployment)
- [Testing and Troubleshooting](#testing-and-troubleshooting)
- [Limitations and Ethical Disclaimer](#limitations-and-ethical-disclaimer)
- [License and Citation](#license-and-citation)

---

## Overview

Diabetes prevalence is increasing globally, and early detection in young demographics offers the highest potential for preventative intervention. However, identifying at-risk individuals in younger populations is fundamentally challenging: in survey data, prediabetes/diabetes prevalence among young adults is low (~2.18%), creating an acute class imbalance where naive models reach 97.8% accuracy simply by predicting negative for every individual while missing 100% of true cases.

This project addresses that challenge by:
1. Filtering and cleaning CDC Behavioral Risk Factor Surveillance System (BRFSS) survey responses for adults aged 18–34.
2. Formulating strict leakage-free ML pipelines combining feature standardization and imbalance remedies (cost-sensitive balanced class weighting and SMOTENC oversampling).
3. Systematically tuning and evaluating five distinct classifier families using stratified 5-fold cross-validation and held-out test evaluation.
4. Deploying a multi-page interactive Streamlit dashboard providing comparative analytics, ROC/PR curves, dynamic confusion matrix thresholding, feature importances, and a live risk assessment simulator.

---

## Key Features

- **Five Distinct Classifier Families**:
  - *Linear*: Logistic Regression (L2 regularized, standard scaled).
  - *Instance/Distance-based*: K-Nearest Neighbors (SMOTENC oversampled, standard scaled, distance weighted).
  - *Tree*: Decision Tree (cost-sensitive, pre-pruned).
  - *Ensemble*: Random Forest (300 estimators, balanced subsampling).
  - *Kernel/Margin-based*: Support Vector Machine (RBF kernel, standard scaled).
- **Leakage-Free Imbalance Handling**:
  - Distance scaling (`StandardScaler`) and synthetic oversampling (`SMOTENC`) are embedded strictly inside training pipeline folds to prevent test set data contamination.
  - Class-weighted cost adjustment (`class_weight="balanced"`) penalizes minority misclassifications proportionally (~45x penalty per positive case).
- **Metric-Driven Model Selection**:
  - Models were selected via `GridSearchCV` optimized on ROC-AUC rather than misleading accuracy.
  - Comprehensive reporting across Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- **Interactive Multi-Page Streamlit Dashboard**:
  - **Overview**: Dataset KPIs, headline model rankings, and screening trade-off notes.
  - **Dataset Explorer**: Class balance metrics, feature distributions by diagnostic status, and Pearson correlation matrices.
  - **Model Comparison**: Metric bar charts, precision-recall trade-off plots, cross-validation variance bars ($\pm 1\text{ SD}$), and compute timing.
  - **ROC & Confusion Matrices**: Multi-model ROC curves and interactive decision-threshold slider.
  - **Feature Importance**: Impurity-based importances and standardized logistic regression odds coefficients.
  - **Risk Predictor**: Demonstration input form ranking a user-defined survey profile against the test population.

---

## System Architecture

```text
+-------------------------------------------------------------------------------+
|                        CDC BRFSS 2015 Survey Dataset                          |
|                  (253,680 rows, 21 health indicator features)                 |
+-------------------------------------------------------------------------------+
                                        |
                                        v filter.py
+-------------------------------------------------------------------------------+
|                     Young Adult Subset (Ages 18-34)                           |
|             Age Categories 1-3 | 24,421 rows | 532 Positives (2.18%)           |
+-------------------------------------------------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v eda.ipynb                             v models.ipynb
+---------------------------------------+   +-----------------------------------+
|      Exploratory Data Analysis        |   |   Stratified 80/20 Split (Seed 42)|
| - Feature distributions & outlier EDA |   |   - Train: 19,536 rows (426 pos)  |
| - Correlation matrix & target ranks   |   |   - Test:   4,885 rows (106 pos)  |
| - Pre-generated figure export         |   +-----------------------------------+
+---------------------------------------+                     |
                                                              v
                                            +-----------------------------------+
                                            | Scikit-Learn / ImbLearn Pipelines |
                                            | - StandardScaler (LR, KNN, SVM)   |
                                            | - SMOTENC (KNN training fold only)|
                                            | - Balanced class weights          |
                                            +-----------------------------------+
                                                              |
                                                              v
                                            +-----------------------------------+
                                            | Stratified CV Grid Search         |
                                            | - Hyperparameter optimization     |
                                            | - Out-of-fold scoring (ROC-AUC)   |
                                            +-----------------------------------+
                                                              |
                                                              v
                                            +-----------------------------------+
                                            | Evaluation & Joblib Artifacts     |
                                            | - best_models.joblib              |
                                            | - test_predictions.joblib         |
                                            | - results.csv, cv_results.csv     |
                                            +-----------------------------------+
                                                              |
                                                              v
+-------------------------------------------------------------------------------+
|                       Streamlit Web Dashboard (app.py)                        |
|   Interactive Analytics | Model Comparison | Confusion Matrices | Predictor   |
+-------------------------------------------------------------------------------+
```

---

## Directory Structure

```text
Lab_Project/
├── README.md                                          # Complete technical documentation
├── filter.py                                          # Data filtering script (extracts young adults 18-34)
├── eda.ipynb                                          # Exploratory Data Analysis notebook
├── models.ipynb                                       # ML pipeline, Grid Search, validation & export
├── diabetes_binary_health_indicators_BRFSS2015.csv   # Source CDC BRFSS dataset (253,680 rows)
├── diabetes_young_adults.csv                          # Filtered young adult dataset (24,421 rows)
├── dashboard/                                         # Streamlit application package
│   ├── app.py                                         # Multi-page interactive Streamlit dashboard
│   ├── requirements.txt                               # Exact Python dependencies for dashboard
│   ├── config.toml                                    # Dashboard theme configuration
│   ├── best_models.joblib                             # Serialized tuned pipelines for all 5 models
│   ├── test_predictions.joblib                        # Cached test predictions and probability scores
│   ├── train.csv                                      # Stratified training split (19,536 rows)
│   ├── test.csv                                       # Stratified held-out test split (4,885 rows)
│   ├── results.csv                                    # Test set evaluation metrics for all 5 models
│   ├── cv_results.csv                                 # 5-fold cross-validation scores (mean ± std)
│   ├── tuning_results.csv                             # Best hyperparameters and tuning durations
│   └── timing.csv                                     # Model training and prediction runtimes
└── figures/                                           # Exported EDA and dashboard evaluation plots
    ├── 01_class_distribution.png
    ├── 02_numeric_distributions.png
    ├── 03_ordinal_distributions.png
    ├── 04_binary_prevalence.png
    ├── 05_positive_rate_binary.png
    ├── 06_bmi_by_class.png
    ├── 07_genhlth_rate.png
    ├── 08_correlation_heatmap.png
    ├── 09_metric_comparison.png
    ├── 10_roc_curves.png
    ├── 11_confusion_matrices.png
    ├── 12_rf_feature_importance.png
    └── dash_*.png                                     # Dashboard UI screenshots (1-13)
```

---

## Technology Stack

| Category | Technology / Library | Purpose |
|---|---|---|
| **Programming Language** | Python 3.12 / 3.13 | Core runtime for analysis, modeling, and dashboard |
| **Machine Learning** | `scikit-learn` (`1.9.1`) | Estimators, Pipelines, `GridSearchCV`, Metrics |
| **Imbalanced Learning** | `imbalanced-learn` (`0.14.2`) | `SMOTENC` oversampling and `imblearn.pipeline` |
| **Data Manipulation** | `pandas` (`3.0.6`), `numpy` (`2.5.3`) | Tabular data extraction, transformation, and matrix operations |
| **Model Serialization** | `joblib` (`1.6.0`) | Serializing trained estimator pipelines and evaluation arrays |
| **Interactive Dashboard** | `streamlit` | Multi-page web dashboard and reactive UI widgets |
| **Data Visualization** | `plotly` (`express` & `graph_objects`) | Interactive browser charts with hover diagnostics |
| **Static Plotting** | `matplotlib` (`3.11.2`), `seaborn` (`0.13.2`) | Publication-ready visual analysis in Jupyter notebooks |
| **Deployment Platform** | Streamlit Community Cloud (PaaS) | Cloud hosting via Git repository synchronization |

---

## Dataset and Preprocessing

### Source and Scope

The data originates from the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**, cleaned and distributed on Kaggle by Alex Teboul.
- **Original file**: `diabetes_binary_health_indicators_BRFSS2015.csv` (253,680 rows, 21 feature columns).
- **Target variable**: `Diabetes_binary`
  - `0`: No diabetes
  - `1`: Prediabetes or diagnosed diabetes
- **Young adult filtering**: `filter.py` subsets respondents where `Age` $\in \{1, 2, 3\}$:
  - Code `1`: Ages 18–24
  - Code `2`: Ages 25–29
  - Code `3`: Ages 30–34
- **Young adult cohort**: 24,421 rows.

### Severe Class Imbalance

| Cohort | Total Records | Negative Cases (0) | Positive Cases (1) | Positive Prevalence |
|---|---|---|---|---|
| Full BRFSS 2015 | 253,680 | 218,334 | 35,346 | 13.93% |
| Young Adults (18–34) | 24,421 | 23,889 | 532 | **2.18%** |
| Training Set (80%) | 19,536 | 19,110 | 426 | 2.18% |
| Test Set (20%) | 4,885 | 4,779 | 106 | 2.18% |

An unweighted model always predicting negative reaches **97.8% accuracy** while identifying **0 true cases**. Accuracy is therefore completely inadequate as an evaluation metric.

### Feature Inventory (21 Features)

1. **Continuous / Body Measurement** (1): `BMI` (Body Mass Index, observed range: 12.0 to 98.0 kg/m²).
2. **Frequency Counts** (2):
   - `MentHlth`: Days of poor mental health in past 30 days (0–30).
   - `PhysHlth`: Days of physical illness or injury in past 30 days (0–30).
3. **Ordinal Ratings** (4):
   - `GenHlth`: Self-rated general health (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor).
   - `Age`: 5-year bracket (1 = 18–24, 2 = 25–29, 3 = 30–34).
   - `Education`: Education level attained (scale 1–6).
   - `Income`: Annual household income category (scale 1–8).
4. **Binary Indicators** (14):
   - `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` (0 = Female, 1 = Male).

### Preprocessing Pipeline and Leakage Prevention

- **Data Integrity**: Confirmed 0 null/missing values across all rows. 1,823 duplicate rows exist in the young adult cohort and were retained because survey responses on discrete low-cardinality scales naturally produce identical answer profiles across different people.
- **Stratification**: An 80/20 train/test split was performed using `train_test_split(..., stratify=y, random_state=42)` ensuring identical 2.18% positive prevalence in both splits.
- **Leakage Prevention**: Feature standardizers (`StandardScaler`) and oversampling transformers (`SMOTENC`) were strictly encapsulated inside `Pipeline` objects:
  - `StandardScaler` was fitted strictly on training folds and applied to test folds without parameter leakage.
  - Tree-based models (Decision Tree and Random Forest) operate on monotonic split points and were left unscaled.
  - `SMOTENC` was configured with `categorical_features=cat_idx` to synthesize minority samples along categorical and binary constraints without generating fractional values for binary columns.

---

## Models and Hyperparameter Tuning

All models were tuned via `GridSearchCV` on the training set using stratified cross-validation and scored on `roc_auc`:

| Model | Pipeline Architecture | Tuned Hyperparameter Grid | Selected Optimal Configuration | CV ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | `StandardScaler` + `LogisticRegression` | `clf__C`: [0.001, 0.003, 0.01, 0.03, 0.1] | `C = 0.001`, `class_weight='balanced'` | 0.8112 |
| **KNN** | `SMOTENC` + `StandardScaler` + `KNeighborsClassifier` | `clf__n_neighbors`: [31, 51, 101, 201]<br>`clf__weights`: ['uniform', 'distance'] | `n_neighbors = 201`, `weights = 'distance'` | 0.6951 |
| **Decision Tree** | `DecisionTreeClassifier` | `max_depth`: [2, 3, 4, 5]<br>`min_samples_leaf`: [50, 100, 200] | `max_depth = 3`, `min_samples_leaf = 200`, `class_weight='balanced'` | 0.7738 |
| **Random Forest** | `RandomForestClassifier` (300 trees) | `max_depth`: [3, 4, 6]<br>`min_samples_leaf`: [20, 50, 100] | `max_depth = 6`, `min_samples_leaf = 100`, `class_weight='balanced_subsample'` | 0.8042 |
| **SVM** | `StandardScaler` + `SVC(kernel='rbf')` | `clf__C`: [0.01, 0.03, 0.1, 0.3] (3-fold CV) | `C = 0.03`, `kernel = 'rbf'`, `class_weight='balanced'` | 0.7997 |

---

## Empirical Results and Performance Analysis

### Held-Out Test Set Performance

Evaluated on the independent test split ($N = 4,885$; 106 positive cases, 4,779 negative cases):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | True Pos (TP) | False Pos (FP) | False Neg (FN) | True Neg (TN) |
|---|---|---|---|---|---|---|---|---|---|---|
| **SVM** | 0.755 | 0.067 | **0.792** | 0.123 | **0.840** | 0.155 | **84** | 1,177 | **22** | 3,602 |
| **Random Forest** | 0.815 | **0.084** | 0.755 | **0.151** | 0.829 | 0.178 | 80 | **877** | 26 | **3,902** |
| **Logistic Regression** | 0.778 | 0.070 | 0.755 | 0.128 | 0.830 | **0.194** | 80 | 1,060 | 26 | 3,719 |
| **Decision Tree** | **0.831** | 0.078 | 0.632 | 0.139 | 0.786 | 0.088 | 67 | 789 | 39 | 3,990 |
| **KNN** | 0.754 | 0.049 | 0.557 | 0.089 | 0.736 | 0.079 | 59 | 1,155 | 47 | 3,624 |
| *Baseline (All-No)* | *0.978* | *0.000* | *0.000* | *0.000* | *0.500* | *0.022* | *0* | *0* | *106* | *4,779* |

### Cross-Validation Performance

5-fold cross-validation results on the training set ($\text{mean} \pm \text{standard deviation}$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | $0.780 \pm 0.004$ | $0.067 \pm 0.006$ | $0.700 \pm 0.065$ | $0.122 \pm 0.011$ | $0.811 \pm 0.017$ |
| **Random Forest** | $0.810 \pm 0.006$ | $0.073 \pm 0.007$ | $0.655 \pm 0.058$ | $0.131 \pm 0.013$ | $0.804 \pm 0.027$ |
| **SVM** | $0.760 \pm 0.007$ | $0.061 \pm 0.004$ | $0.697 \pm 0.044$ | $0.113 \pm 0.008$ | $0.798 \pm 0.025$ |
| **Decision Tree** | $0.803 \pm 0.094$ | $0.073 \pm 0.020$ | $0.587 \pm 0.076$ | $0.128 \pm 0.032$ | $0.774 \pm 0.026$ |
| **KNN** | $0.739 \pm 0.016$ | $0.042 \pm 0.004$ | $0.507 \pm 0.055$ | $0.078 \pm 0.007$ | $0.695 \pm 0.016$ |

### Training and Inference Latency

Measured across training and test inference sets:

| Model | Training Time (s) | Prediction Time (s, $N=4,885$) | Inference Speed Profile |
|---|---|---|---|
| **Decision Tree** | **0.03 s** | **0.002 s** | Real-time (<5 ms) |
| **Logistic Regression** | 0.29 s | 0.020 s | Real-time (<25 ms) |
| **Random Forest** | 1.05 s | 0.060 s | High throughput |
| **KNN** | 3.74 s | 0.553 s | Moderate (pair distance computations) |
| **SVM (RBF)** | 15.39 s | 6.038 s | Computationally heavy (kernel matrix evaluations) |

### Why Recall and ROC-AUC Matter

1. **Clinical Screening Goal**: In early disease risk screening, missing an at-risk individual (False Negative) delays preventative lifestyle changes or medical follow-ups. In contrast, a False Positive leads to a simple, non-invasive confirmatory blood test (e.g., HbA1c or fasting blood glucose). Therefore, **Recall** is prioritized.
2. **The Precision Math**: Because positive prevalence is only $2.18\%$, even an outstanding model will flag several false alarms for every true positive. Random Forest achieved a precision of $0.084$—nearly **$4\times$ the base population rate**—catching 80 of 106 cases while clearing 3,902 healthy respondents.
3. **Top Tier**: **SVM**, **Random Forest**, and **Logistic Regression** form a high-performing cluster (ROC-AUC 0.83–0.84). 
   - **SVM** achieved the highest recall ($0.792$, catching 84 cases).
   - **Random Forest** provided the best precision ($0.084$) and F1 ($0.151$), producing 183 fewer false alarms than Logistic Regression while matching its 80 detected cases.
   - **Logistic Regression** provided the highest PR-AUC ($0.194$) and direct interpretability.

### Feature Importance Highlights

1. **`GenHlth` (Self-Rated General Health)**: Dominates across models as the strongest predictor of metabolic risk.
2. **`BMI` (Body Mass Index)**: Consistently ranked second; non-linear risk escalation occurs prominently above $30\text{ kg/m}^2$.
3. **`HighChol` and `HighBP`**: Strongest clinical indicators in survey responses.
4. **`Stroke` and `HeartDiseaseorAttack`**: Very high conditional correlation with diabetes (~11%), but contribute minimal global tree split importance because acute cardiovascular events occur in $<1\%$ of this young cohort.

---

## Interactive Dashboard

The Streamlit dashboard (`dashboard/app.py`) is organized into six functional views:

```text
+-----------------------------------------------------------------------------------+
| 🩺 Diabetes Risk Dashboard                                                        |
+---------------------+-------------------------------------------------------------+
| [Navigation]        |  Main Display Area                                          |
| (•) Overview        |  • KPI Summary: 24,421 Respondents, 2.18% Prevalence        |
| ( ) Dataset         |  • Headline Rankings (SVM ROC-AUC: 0.840, RF F1: 0.151)     |
| ( ) Model Comparison|                                                             |
| ( ) ROC & CM        |  Interactive Views:                                         |
| ( ) Feature Import. |  - Precision-Recall trade-off scatter plot                  |
| ( ) Risk Predictor  |  - Interactive What-If Decision Threshold Slider            |
|                     |  - Custom survey profile percentile scoring                 |
+---------------------+-------------------------------------------------------------+
```

1. **Overview**: Key project statistics, total respondent counts, and clinical trade-off explanations.
2. **Dataset**: Class balance bar chart, stratified split tables, interactive feature distribution histograms, and 21x21 correlation matrix heatmap.
3. **Model Comparison**: Grouped metric bar charts (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC), precision-recall scatter plot with bubble sizing by F1, 5-fold CV error bar graphs, and timing comparisons.
4. **ROC Curves & Confusion Matrices**: Interactive multi-trace ROC curves and confusion matrix heatmaps with an interactive **What-If Threshold Slider** that simulates changing decision cutoffs in real time.
5. **Feature Importance**: Comparative bar charts of tree impurity metrics (Random Forest, Decision Tree) and standardized regression coefficients (Logistic Regression) colored by risk direction.
6. **Risk Predictor (Demo)**: A three-column user interface accepting health indicators and outputting model risk flags, raw decision scores, and population percentiles relative to test respondents.

---

## Prerequisites

- **Operating System**: Windows, macOS, or Linux.
- **Python**: Version **3.12** or **3.13** (tested and verified on Python 3.12.4 and Python 3.13.4).
- **Package Manager**: `pip` (Python package manager).
- **Web Browser**: Modern browser (Chrome, Edge, Firefox, Safari) for dashboard interaction.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SKSingh0703/Lab_Project.git
cd Lab_Project
```

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the pinned dashboard and model dependencies:

```bash
pip install -r dashboard/requirements.txt
```

If you plan to execute or modify the Jupyter notebooks (`eda.ipynb` or `models.ipynb`), also install Jupyter and plotting packages:

```bash
pip install notebook matplotlib seaborn
```

---

## Usage Instructions

### 1. Launching the Interactive Streamlit Dashboard

Run the Streamlit application from the repository root:

```bash
streamlit run dashboard/app.py
```

Or from within the `dashboard/` directory:

```bash
cd dashboard
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

### 2. Running Data Filtering

To regenerate `diabetes_young_adults.csv` from the raw BRFSS dataset:

```bash
python filter.py
```

### 3. Running Exploratory Data Analysis & Modeling Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

- Open `eda.ipynb` to inspect univariate/bivariate distributions and generate static figures saved to `figures/`.
- Open `models.ipynb` to rerun the stratified train/test split, cross-validation, hyperparameter tuning grid search, and export the trained pipelines to `dashboard/best_models.joblib`.

---

## Environment Variables

This application is entirely self-contained and operates on local serialized artifacts and CSV datasets.

| Variable | Purpose | Required | Default / Example |
|---|---|---|---|
| *None* | No external API keys, database URLs, or secret credentials are required | **No** | N/A |

Optional Streamlit configuration can be customized via `.streamlit/config.toml` (e.g., custom primary theme color, headless mode, or port binding).

---

## Cloud Deployment

The interactive dashboard is architected for deployment on **Streamlit Community Cloud** (PaaS):

1. **Repository Structure**: `dashboard/` contains all assets required by `app.py` (`requirements.txt`, `best_models.joblib`, `test_predictions.joblib`, `results.csv`, `cv_results.csv`, `tuning_results.csv`, `train.csv`, `test.csv`, and `timing.csv`).
2. **Deployment Steps**:
   - Push repository to GitHub (`main` branch).
   - Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
   - Click **New app**, select the repository (`SKSingh0703/Lab_Project`), branch (`main`), and set the main file path to:
     ```text
     dashboard/app.py
     ```
   - Deploy. Streamlit automatically installs packages from `dashboard/requirements.txt` and starts the application.
3. **Version Pinning Note**: The serialized models in `best_models.joblib` require compatible versions of `scikit-learn` and `joblib`. These are explicitly pinned in `dashboard/requirements.txt` (`scikit-learn==1.9.1`, `joblib==1.6.0`, `imbalanced-learn==0.14.2`) to prevent unpickling mismatches during automated cloud builds.

---

## Testing and Troubleshooting

### Verification Commands

Validate Python syntax and package compatibility without executing heavy pipelines:

```bash
# Validate dashboard syntax
python -m py_compile dashboard/app.py

# Validate filter script syntax
python -m py_compile filter.py

# Verify model artifact unpickling
python -c "import joblib; m = joblib.load('dashboard/best_models.joblib'); print('Loaded models:', list(m.keys()))"
```

### Common Issues and Solutions

- **`ModuleNotFoundError: No module named 'imblearn'`**:
  - Install `imbalanced-learn` via `pip install imbalanced-learn==0.14.2`.
- **`InconsistentVersionWarning` or `ValueError` during `joblib.load`**:
  - Occurs if `scikit-learn` or `joblib` versions differ from the serialization environment. Ensure your environment matches `dashboard/requirements.txt` (`scikit-learn==1.9.1`, `joblib==1.6.0`).
- **Slow SVM Predictions**:
  - The SVM classifier uses an RBF kernel. Inference scales with the number of support vectors ($N=4,885$ test predictions require ~6.0 seconds). This is normal behavior.
- **Port Conflict on Streamlit Launch**:
  - If port 8501 is already in use, launch on a custom port:
    ```bash
    streamlit run dashboard/app.py --server.port 8502
    ```

---

## Limitations and Ethical Disclaimer

- **Non-Diagnostic Tool**: This software and its underlying models are designed strictly for educational and academic screening demonstration. They do **not** provide medical advice, diagnosis, or clinical evaluation.
- **Survey Data Constraints**: Data is based on self-reported telephone survey responses (CDC BRFSS 2015), which may carry self-reporting bias or recall inaccuracies.
- **Combined Target**: The target variable (`Diabetes_binary`) combines diagnosed diabetes with prediabetes and does not distinguish between Type 1 and Type 2 diabetes. While Type 2 is lifestyle-associated, Type 1 has autoimmune origins; respondents aged 18–34 likely include both etiologies.
- **Uncalibrated Model Scores**: Due to minority oversampling and cost-sensitive class reweighting, raw output scores do not reflect real-world Bayesian probabilities. The dashboard uses population percentiles relative to the test set to convey comparative relative risk rather than calibrated diagnostic likelihood.

---

## License and Citation

### Academic Attribution

This project was conducted as part of the **Cloud Computing Lab (Semester 7)** coursework at college.

### Data Citation

- **Original Survey**: Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System (BRFSS) 2015 Survey Data*. Atlanta, Georgia: U.S. Department of Health and Human Services, CDC.
- **Curated Dataset**: Teboul, Alex. *Diabetes Health Indicators Dataset*. Kaggle (2021). Derived from BRFSS 2015.
