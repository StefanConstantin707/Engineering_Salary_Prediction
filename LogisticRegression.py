import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score

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

feature_cols   = [f"feature_{i}" for i in range(1, 13)]
feature_cols   = [f"feature_{i}" for i in range(1, 13)]
extra_feature  = ["job_state", "job_title"]
total_features = feature_cols + extra_feature

X = df.select(total_features)
X = X.to_dummies(columns=["job_state", "job_title", "feature_1"])

# standardize feature_10 in Polars
X = X.with_columns(
    ((pl.col("feature_10") - pl.col("feature_10").mean()) /
     pl.col("feature_10").std()).alias("feature_10")
)

# 2. Polynomial expansion (no bias column)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_np = poly.fit_transform(X.to_numpy())
poly_feature_names = poly.get_feature_names_out(X.columns)

# 3. Normalize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# 4. Encode target
Y = df["salary_category"].to_pandas()
le = LabelEncoder().fit(Y)
Y_enc = le.transform(Y)

# 5. Cross-validate & fit
clf = LogisticRegressionCV(
    solver='saga',
    max_iter=1000,
    random_state=0,
    penalty='l1',
    cv=10,
    n_jobs=8,
)

clf.fit(X_scaled, Y_enc)

# 6. Build a DataFrame of coefficients for each class
#    coef_.shape = (n_classes, n_features)
coef_matrix = clf.coef_                     # shape (n_classes, n_features)
class_labels = le.classes_                  # e.g. ['Low', 'Medium', 'High']

# Create DataFrame: rows = features, columns = classes
df_coefs = pd.DataFrame(
    coef_matrix.T,
    index=poly_feature_names,
    columns=class_labels
)
df_coefs.index.name = "feature"

print("\nAll coefficients by class:")
print(df_coefs)

# 7. Find top 10 parameters by absolute magnitude across all classes
df_long = (
    df_coefs
    .reset_index()
    .melt(id_vars="feature", var_name="class", value_name="coefficient")
)
df_long["abs_coef"] = df_long["coefficient"].abs()
top10 = df_long.sort_values("abs_coef", ascending=False).head(10)

print("\nTop 10 parameters by |coefficient|:")
print(top10[["feature", "class", "coefficient"]])

mean_scores = clf.scores_[1].mean(axis=0)
C_values = clf.Cs_

plt.plot(C_values, mean_scores, marker='o')
plt.xscale('log')
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Mean CV Accuracy (Class 1)')
plt.title('Cross-Validation Accuracy vs. C')
plt.grid(True)
plt.show()

print(mean_scores.max())
