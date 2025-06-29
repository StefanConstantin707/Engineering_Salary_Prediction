## Technical Methodology

### Table of Contents

1. [Problem Overview](#problem-overview)
2. [Data Understanding](#data-understanding)
3. [Feature Engineering](#feature-engineering)
4. [Modeling Approach](#modeling-approach)
5. [Optimization Strategy](#optimization-strategy)
6. [Key Innovations](#key-innovations)
7. [Results](#results)
8. [Lessons Learned](#lessons-learned)
9. [Future Improvements](#future-improvements)
10. [Code Structure](#code-structure)
11. [Reproducibility](#reproducibility)

---

## Problem Overview

The challenge involves predicting engineering salaries categorized into three ordinal classes:

* **Low**: Entry-level salaries
* **Medium**: Mid-level salaries
* **High**: Senior-level salaries

The ordinal nature of the target (Low < Medium < High) informs our choice of modeling techniques.

**Dataset:**

* Training samples: 1,281
* Test samples: 320
* Features: 313 (including 300 sparse job description features)
* Balanced target distribution across classes

---

## Data Understanding

1. **Job Description Features** (300 columns)

   * Word2Vec type job description encoding 
   * Very high dimensionality relative to dataset size

2. **Categorical Features**

   * `job_title`: Position titles
   * `job_state`: US state abbreviations
   * `feature_1`: Binary category (A/B/C)

3. **Boolean Features**

   * `feature_3`–`feature_9`, `feature_11`, `feature_12`
   * Binary flags (job characteristics)

4. **Quantitative Features**

   * `feature_2`: Numeric 
   * `feature_10`: Numeric (years of experience)

5. **Temporal Feature**

   * `job_posted_date`: YYYY/MM format

---

## Feature Engineering

1. **Job Description Clustering**

   * **Problem:** 300 features ⇒ high dimensionality issues
   * **Solution:** KMeans clustering of job descriptions into 3–5 groups

     ```python
     # Fit KMeans on the 300 binary features
     # Replace 300 columns with cluster membership probabilities
     ```
   * **Benefits:**

     * Reduces 300 → 3–6 features
     * Interpretable job categories
     * Better generalization & handles missing values

2. **State-based Salary Encoding**

   * **Insight:** Average salaries vary by US state
   * **Implementation:**

     ```python
     state_to_avg_salary = {
         "NH": 97046, "MA": 94651, "OR": 94286, ...
     }
     df['state_salary_avg'] = df['job_state'].map(state_to_avg_salary)
     ```
   * **Impact:** Captures geographic salary variation as a single numeric feature

3. **Temporal Features**

   * `months_since_first`: Months since the earliest posting (captures market trends)
   * `month_of_year`: Seasonality in hiring patterns
   * `month_target_stats`: Historical salary distribution by month

4. **Feature Interactions (Optional)**

   * Polynomial terms for key numeric features
   * Interaction between experience (`feature_10`) and job clusters
   * Added selectively to avoid overfitting

---

## Modeling Approach

Treating salary prediction as an ordinal classification problem:

* Transform 3 classes into 2 binary thresholds (Frank & Hall approach):

  1. Is salary > Low? (Medium/High vs. Low)
  2. Is salary > Medium? (High vs. Medium/Low)

**Algorithms Compared:**

1. **XGBoost + Ordinal** (Best)

   * CV Accuracy: 75.6% ±1.2%
   * Robust to mixed data types and outliers
2. **LightGBM + Ordinal**

   * CV Accuracy: 70.8%
   * Faster training
3. **Logistic Regression + Poly Features**

   * CV Accuracy: 68.5%
4. **Neural Network**

   * CV Accuracy: 67.3%
   * Prone to overfitting on small data

---

## Optimization Strategy

1. **Outlier Detection**

   * Ensemble of Isolation Forest, LOF, and Z-score
   * Removed \~5% outliers ⇒ +1.5% CV gain

2. **Feature Selection**

   * Tree-based importance with cross-validation
   * Retained 60–70% of features
   * Eliminated redundant job description features

3. **Hyperparameter Optimization**

   * Bayesian optimization (100 iterations)
   * Key parameters:

     ```yaml
     n_estimators: 800
     max_depth: 10
     learning_rate: 0.005
     subsample: 0.8
     colsample_bytree: 0.7
     ```

---

## Key Innovations

1. **Unified Pipeline**

   ```python
   Pipeline([
       ('preprocessing', DataHandler()),
       ('feature_selection', FeatureSelector()),
       ('classifier', OrdinalClassifier(XGBClassifier()))
   ])
   ```

2. **Experiment Tracking System**

   * Automatic logging of parameters & metrics
   * Visualization of iteration history
   * Ensures reproducibility

3. **Robust Validation**

   * 10-fold stratified cross-validation
   * Consistent preprocessing per fold
   * Prevented data leakage

---

## Results

* **Best CV Accuracy:** 75.6% (±1.2%)
* **Training Accuracy:** 74.5%
* **Feature Reduction:** 313 → \~200 features

**Per-Class Metrics:**

| Class  | Precision | Recall | F1-Score |
| ------ |-----------|--------| -------- |
| Low    | 0.75      | 0.73   | 0.72     |
| Medium | 0.68      | 0.70   | 0.69     |
| High   | 0.74      | 0.76   | 0.73     |

---

## Lessons Learned

1. Ordinal classification outperforms standard multiclass for ordered targets
2. Feature engineering (clustering) has greater impact than model complexity
3. Careful outlier removal consistently improves performance
4. Systematic experiments with tracking are essential

---

## Code Structure

```
src/
├── data/           # Data handling & transforms
├── models/         # Model definitions
├── preprocessing/  # Feature engineering & selection
├── optimization/   # Hyperparameter tuning
└── utils/          # Tracking & visualization
```

---

## Reproducibility

1. Install: `pip install -r requirements.txt`
2. Place data in `data/raw/`
3. Train: `python scripts/train_best_model.py`
4. Submit: `python scripts/generate_submission.py`

*Top 10 finish among 600+ participants demonstrates the effectiveness of combining domain insight, feature engineering, and advanced modeling.*

