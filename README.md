# Engineering Salary Prediction 🏆

**Top 10 out of 600+ participants** in the Engineering Salary Prediction competition

A comprehensive machine learning solution for predicting engineering salaries using advanced feature engineering, ordinal classification, and ensemble methods.

## 🎯 Project Overview

This project tackles the challenge of predicting engineering salaries (Low/Medium/High) based on job postings data. The solution combines sophisticated preprocessing techniques with state-of-the-art machine learning models to achieve exceptional performance.

### Key Achievements
- **Competition Rank**: Top 10 out of 600+ participants
- **Best CV Accuracy**: 75.6% (10-fold cross-validation)
- **Novel Approach**: Ordinal classification with custom job description clustering

## 🚀 Features

### Advanced Data Processing
- **Job Description Clustering**: Reduces 300 sparse job description features to meaningful cluster representations
- **Temporal Feature Engineering**: Extracts time-based patterns from job posting dates
- **State-based Salary Encoding**: Maps US states to average salary values
- **Intelligent Imputation**: Handles missing data with cluster-aware strategies

### Sophisticated Modeling Techniques
- **Ordinal Classification**: Custom implementation treating salary as ordered categories (Low < Medium < High)
- **Model Selection**: XGBoost, LightGBM, and Neural Networks with Bayesian optimization
- **Feature Selection**: Multiple strategies including tree importance and mutual information
- **Outlier Detection**: Ensemble approach combining Isolation Forest, LOF, and statistical methods

### Experiment Tracking
- **Comprehensive Run Tracking**: Automated logging of all experiments with detailed metrics
- **Hyperparameter History**: Complete record of Bayesian optimization iterations
- **Result Analysis**: Automated generation of performance reports and visualizations

## 📊 Model Performance

| Model | CV Accuracy | Features Used | Key Technique |
|-------|-------------|---------------|---------------|
| XGBoost + Ordinal | 75.6%       | 300+ with selection | Feature selection + outlier removal |
| LightGBM + Ordinal | 70.8%       | Clustered features | Job description clustering |
| Logistic + Polynomial | 68.5%       | Polynomial interactions | 2nd degree interactions |
| Neural Network | 67.3%       | All features | Deep architecture with dropout |

## 🛠️ Technical Stack

- **Core ML**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Optimization**: scikit-optimize (Bayesian optimization)
- **Data Processing**: pandas, numpy, polars
- **Visualization**: matplotlib, seaborn
- **Tracking**: Custom run tracking system with JSON/CSV export

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data handling and feature engineering
│   ├── models/            # Model implementations
│   ├── preprocessing/     # Outlier detection, feature selection
│   ├── optimization/      # Hyperparameter optimization
│   └── utils/             # Utilities and helpers
├── experiments/           # Experiment configs and results
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Training and evaluation scripts
└── data/                 # Data directory
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/engineering-salary-prediction.git
cd engineering-salary-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the competition data files in the `data/raw/` directory:
- `train.csv`: Training data with salary labels
- `test.csv`: Test data for predictions

### Train Best Model

```bash
# Train the best performing model (XGBoost with optimized hyperparameters)
python scripts/train_best_model.py

# Generate competition submission
python scripts/generate_submission.py
```

### Run Full Experiments

```bash
# Run all model experiments with hyperparameter optimization
python scripts/run_all_experiments.py

# Analyze and compare results
python scripts/analyze_results.py
```

## 📈 Key Innovations

### 1. Job Description Clustering
Instead of using 300 sparse job description features directly, the solution employs KMeans clustering to create meaningful job groups:
- Reduces dimensionality while preserving information
- Creates interpretable job categories
- Improves model generalization

### 2. Ordinal Classification Framework
Custom implementation based on Frank & Hall's approach:
- Transforms 3-class problem into 2 binary problems
- Leverages ordering information (Low < Medium < High)
- Provides calibrated probability estimates

### 3. Comprehensive Feature Engineering
- **Temporal features**: Months since reference date, seasonal patterns
- **Geographic features**: State-based salary averages
- **Interaction features**: Polynomial combinations of key features
- **Target encoding**: For high-cardinality categorical variables

### 4. Robust Preprocessing Pipeline
- **Outlier detection**: Ensemble method identifying ~5% anomalies
- **Feature selection**: Reduces features by 30-40% while maintaining performance
- **Stratified validation**: Ensures reliable performance estimates

## 📊 Results Analysis

### Feature Importance
Top contributing features:
1. Job description cluster probabilities
2. Years of experience (feature_10)
3. State-based salary encoding
4. Temporal features (months since posting)
5. Boolean feature combinations

### Model Insights
- **Cluster Analysis**: Identified 3-5 distinct job categories with varying salary distributions
- **Outlier Impact**: Removing outliers improved CV accuracy by ~1.5%
- **Feature Selection**: Optimal performance with 60-70% of features retained

## 🔧 Configuration

Model hyperparameters and training configurations are stored in `experiments/configs/`. Key parameters:

```yaml
# xgboost_config.yaml
xgboost:
  n_estimators: 800
  max_depth: 10
  learning_rate: 0.005
  subsample: 0.8
  colsample_bytree: 0.7
  
preprocessing:
  n_clusters: 3
  outlier_method: ensemble
  feature_selection_ratio: 0.65
```

## 📝 Documentation

Detailed documentation available in the `docs/` directory:
- `methodology.md`: Complete technical approach
- `feature_engineering.md`: Feature creation process
- `results_analysis.md`: Detailed performance analysis

---

**Author**: Stefan Constantin
**Competition**: Engineering Salary Prediction  
**Rank**: Top 10 / 600+  
**Date**: 15/06/2025

*If you find this solution helpful, please consider giving it a ⭐!*