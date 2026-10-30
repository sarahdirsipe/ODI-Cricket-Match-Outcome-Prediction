#  ODI Cricket Match Outcome Prediction Using Statistical Analysis and Machine Learning

A Data Science project analyzing One Day International (ODI) cricket matches using Exploratory Data Analysis (EDA), statistical hypothesis testing, dimensionality reduction, clustering, and machine learning classification.

**Project Type:** Data Science | Statistical Analysis | Machine Learning  
**Language:** Python  
**Dataset:** 4,644 ODI match records

---

##  Project Overview

Cricket match outcomes depend on multiple factors, including batting performance, bowling efficiency, partnerships, and match conditions. Understanding the relationships between these variables can help reveal meaningful patterns in match performance.

This project analyzes historical ODI cricket data using statistical techniques and machine learning to investigate performance differences, test hypotheses, identify patterns, and classify match outcomes based on match-performance metrics.

The project combines statistical inference with predictive modeling to demonstrate how data science techniques can be applied to sports analytics.

> **Important:** The current classification models use match-performance variables, some of which are available only during or after a match. Therefore, this project demonstrates retrospective match-outcome classification rather than genuine pre-match winner prediction.

##  Project Objectives

- Explore ODI match data and identify important performance patterns.
- Compare batting and bowling performance across match outcomes.
- Apply statistical hypothesis tests to examine relationships between performance metrics and match results.
- Use correlation analysis and multiple linear regression to investigate relationships between variables.
- Apply Principal Component Analysis (PCA) and K-Means clustering to explore underlying data patterns.
- Train and compare multiple machine learning classification models.
- Evaluate model performance using standard classification metrics.

##  Dataset Description

**Dataset file:** `final_odi_dataset.csv`

| Dataset Property | Description |
|---|---|
| Total records | 4,644 |
| Original variables | 16 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Prediction target | `match_outcome` |
| Class 1 | Team 1 wins |
| Class 0 | Team 2 wins |

The dataset contains match-level information relating to teams, runs scored, wickets, strike rates, bowling economy rates, average partnerships, toss information, venue, and match outcome.

The target variable, `match_outcome`, represents the winning outcome from Team 1's perspective.

##  Technologies and Libraries

- **Python** — Programming and analysis
- **Pandas and NumPy** — Data manipulation and numerical computation
- **Matplotlib and Seaborn** — Data visualization
- **SciPy** — Statistical hypothesis testing
- **Scikit-learn** — Regression, classification, PCA, clustering, and model evaluation
- **Jupyter Notebook** — Interactive development and analysis

##  Project Workflow

### 1. Data Preparation and Exploratory Data Analysis

- Inspected the dataset structure and data types.
- Checked for missing values and duplicate records.
- Examined descriptive statistics and distributions.
- Analyzed match outcomes and team scoring patterns.
- Visualized relationships between performance variables.

**Visualizations include:**
- Histograms and count plots
- Scatter plots
- Correlation heatmaps
- Residual plots
- PCA scree and cumulative explained-variance plots
- K-Means elbow-method plot
- Confusion matrices and ROC curves

### 2. Statistical Hypothesis Testing

Several statistical techniques were applied to investigate relationships between match performance and outcomes.

| Technique | Purpose |
|---|---|
| Independent-samples t-test | Compare performance metrics across outcome groups |
| Chi-square test of independence | Examine associations between categorical variables |
| One-way ANOVA | Compare mean bowling-economy advantage across selected teams |
| Pearson correlation | Measure linear relationships between numerical variables |
| Multiple linear regression | Estimate Team 1's total runs using selected performance metrics |

Statistical significance was assessed at the 5% significance level for the documented hypothesis tests.

### 3. Advanced Statistical Analysis

**Principal Component Analysis (PCA)**

PCA was applied to standardized numerical features to examine dimensionality reduction and the amount of variance captured by principal components.

In the notebook, the first two principal components explain approximately **67.55%** of the total variance, while the first four explain approximately **84.36%**.

**K-Means Clustering**

K-Means clustering was used to group matches with similar numerical performance characteristics. The elbow method was examined, and a three-cluster solution was analyzed to explore differences in scoring patterns and match outcomes.

**Multiple Linear Regression**

A multiple linear regression model was used to estimate `total_runs_team1` from selected match-performance variables.

Reported test-set results:
- **R² Score:** 0.4718
- **RMSE:** 45.21 runs

These results describe the regression task of estimating Team 1's total runs, not the classification task of predicting match outcomes.

### 4. Machine Learning Classification

Five classification algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine (RBF kernel)

The models were evaluated using a train-test split and the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- ROC curves

Feature importance was also examined for model interpretation.

##  Model Evaluation Results

The following results are reported by the notebook for its classification test set.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 92.68% | 92.18% | 94.07% | 93.12% | 0.9782 |
| Decision Tree | 89.99% | 88.52% | 93.05% | 90.73% | 0.9280 |
| **Random Forest** | **93.33%** | **92.45%** | **95.09%** | **93.75%** | **0.9820** |
| Gaussian Naive Bayes | 92.03% | 91.58% | 93.46% | 92.51% | 0.9673 |
| SVM (RBF) | 88.70% | 87.21% | 92.02% | 89.55% | 0.9593 |

*Note: These scores reflect the notebook's current feature set and evaluation procedure. They should not be interpreted as evidence of equivalent accuracy for predicting winners before a match begins.*

### Key Observations

- Random Forest achieved the highest accuracy and ROC-AUC among the five evaluated models in the reported test results.
- Logistic Regression also achieved strong classification performance.
- PCA showed that the first four principal components captured approximately 84.36% of the variance in the standardized numerical features.
- The K-Means analysis revealed groups with different scoring patterns and observed Team 1 win rates.
- The tested chi-square analyses did not find statistically significant associations between toss choice and match outcome or between winning the toss and winning the match at the 5% significance level.
- The run-differential and strike-rate-differential tests showed statistically significant differences between the outcome groups.

**Interpretation caution:** Statistical significance does not establish causation. Furthermore, some performance-difference variables are closely related to the match outcome by construction, so their statistical significance should not be interpreted as evidence of independent pre-match predictive power.

##  Repository Structure

```text
ODI-Cricket-Match-Outcome-Prediction/
│
├── ODI_Cricket_Statistical_Analysis_FINAL_CODE.ipynb
├── final_odi_dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

| File | Description |
|---|---|
| `ODI_Cricket_Statistical_Analysis_FINAL_CODE.ipynb` | Complete analysis, visualizations, statistical tests, model training, evaluation, and prediction function |
| `final_odi_dataset.csv` | Dataset used in the analysis |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes specified unnecessary files from version control |

##  Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Jupyter Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/sarahdirsipe/ODI-Cricket-Match-Outcome-Prediction.git
cd ODI-Cricket-Match-Outcome-Prediction
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Notebook

```bash
jupyter notebook ODI_Cricket_Statistical_Analysis_FINAL_CODE.ipynb
```

Open the notebook and execute the cells in order. Keep `final_odi_dataset.csv` in the same directory as the notebook so that the dataset-loading code can locate it.

##  Limitations

- Several model features describe performance during or after a match and may reveal information closely related to the final outcome.
- The reported classification scores therefore do not establish the model's ability to predict match winners before play begins.
- Statistical significance does not imply causation.
- Correlation and feature importance do not, by themselves, establish independent causal effects.
- Results depend on the selected features, train-test split, preprocessing, and model settings.

##  Future Improvements

- Develop a separate pre-match prediction model using only information available at prediction time.
- Explore team form, historical head-to-head records, venue statistics, and other appropriate pre-match features.
- Use chronological or season-aware validation to evaluate generalization to future matches.
- Apply leakage checks and feature-selection safeguards.
- Evaluate probability calibration and prediction uncertainty.
- Deploy the model through a web application or API.

##  Author

**D. Sarah**  
M.Sc. Data Science

**GitHub Repository:**  
https://github.com/sarahdirsipe/ODI-Cricket-Match-Outcome-Prediction

##  License

No license has been specified. All rights remain with the copyright holder unless a license is added to the repository.
