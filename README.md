# ODI Cricket Match Outcome Prediction

A data science project exploring One Day International (ODI) cricket match performance through exploratory data analysis, statistical hypothesis testing, advanced statistical techniques, and machine learning classification.

## Project overview

The project investigates how batting, bowling, partnership, and toss-related metrics relate to ODI match outcomes. It combines descriptive and inferential statistics with machine learning to analyze historical match records and build a model that classifies outcomes from match-performance metrics.

**Important scope note:** The notebook uses match-performance variables such as total runs, wickets, strike rate, economy rate, and partnerships. These are generally known during or after a match, so the current model should be understood as a retrospective outcome-classification demonstration—not a pre-match forecast based only on information available before the match.

## Dataset

- File: `final_odi_dataset.csv`
- Records: 4,644 ODI match records
- Original variables: 16
- Missing values: 0 in the supplied CSV
- Duplicate rows: 0 in the supplied CSV
- Target: `match_outcome` (`1` = Team 1 wins; `0` = Team 2 wins)

The dataset contains team identifiers, runs, wickets, strike rates, bowling economy rates, average partnerships, toss information, venue, and the match outcome.

## Methods used

### Data preparation and exploration
- Data inspection and cleaning checks
- Feature engineering for toss-related and performance-difference variables
- Descriptive statistics
- Histograms, count plots, scatter plots, and correlation heatmaps
- Team performance and scoring-pattern analysis

### Statistical analysis
- Independent-samples t-tests
- Chi-square tests of independence
- One-way ANOVA
- Pearson correlation analysis
- Multiple linear regression

### Advanced analysis
- Principal Component Analysis (PCA)
- K-Means clustering and elbow-method inspection

### Machine learning
The notebook compares classification models including Logistic Regression, Decision Tree, Random Forest, and other tree/ensemble baselines defined in the notebook. Evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Feature importance is also examined.

## Repository contents

- `ODI_Cricket_Statistical_Analysis_FINAL_CODE.ipynb` — end-to-end analysis, visualizations, statistical tests, model evaluation, and prediction function.
- `final_odi_dataset.csv` — dataset used by the notebook.
- `requirements.txt` — Python dependencies.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ODI-Cricket-Match-Outcome-Prediction.git
cd ODI-Cricket-Match-Outcome-Prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment (recommended)

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook ODI_Cricket_Statistical_Analysis_FINAL_CODE.ipynb
```

Run the notebook cells from top to bottom. Keep `final_odi_dataset.csv` in the same directory as the notebook so the dataset-loading cell can find it.

## Reproducibility and interpretation notes

- Model scores depend on the train/test split, preprocessing, feature selection, and installed library versions.
- Statistical significance does not by itself establish causation.
- Correlation and feature importance describe relationships used by the model; they do not prove that a variable independently causes a win.
- For a genuine pre-match prediction system, retrain and evaluate using only information available before the match starts. Avoid using final match statistics as input features because they can reveal the outcome.

## Future improvements

- Add a strict pre-match feature set (team form, venue history, toss result only after the toss, and other information available at prediction time).
- Use time-aware or season-aware validation in addition to a random train/test split.
- Add calibration and confidence intervals for predicted probabilities.
- Package the prediction function as a small web application or API.

## Author

**D. Sarah**  
M.Sc. Data Science project

## License

No license has been specified yet. Add a license file if you want to explicitly permit reuse of this project.
