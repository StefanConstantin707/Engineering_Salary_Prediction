from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV


# 1. Load & preprocess
df = pl.read_csv("./data/raw_data/train.csv")
df = df.with_columns(
    pl.col("job_posted_date").str.strptime(pl.Date, format="%Y/%m"),
    pl.col("job_title").cast(pl.Categorical),
    pl.col("salary_category").cast(pl.Categorical),
    pl.col("job_state").cast(pl.Categorical),
    pl.col("feature_1").cast(pl.Categorical),
    pl.col(pl.Boolean).cast(pl.Int32)
).drop_nulls(subset=["feature_10"])

feature_cols = [f"feature_{i}" for i in range(1, 13)]
job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]
extra_feature = ["job_state", "job_title", ""]

total_features = feature_cols + job_desc_cols + extra_feature

X_pl = df.select(total_features).to_dummies(
    columns=["job_state", "job_title", "feature_1"]
)

# 2. Convert to pandas + NumPy
X_pd = X_pl.to_pandas()
X_np = X_pd.values  # shape (n_samples, n_features)

# 3. Encode target
Y = df["salary_category"].to_pandas()
le = LabelEncoder().fit(Y)
Y_enc = le.transform(Y)  # array of ints

# 4. Build pipeline (note: PolynomialFeatures is passed as the class)
pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=0
    ))
])

# 5. Set up Bayesian optimization over C and number of components
bayes_cv = BayesSearchCV(
    pipe,
    {
      'clf__C':      (1e-4, 1e1, 'log-uniform'),
      'pca__n_components': (20, X_np.shape[1])
    },
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=0,
    n_jobs=8,
)

# 6. Fit on the NumPy array + encoded target
bayes_cv.fit(X_np, Y_enc)

print("Best CV score:", bayes_cv.best_score_)
print("Best params:", bayes_cv.best_params_)