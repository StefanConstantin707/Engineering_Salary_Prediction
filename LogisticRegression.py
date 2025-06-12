import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, TargetEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from skopt import BayesSearchCV
from skopt.space import Real

from Classes.DataHandler import DataHandler


def test():
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
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_np = poly.fit_transform(X.to_numpy())
    poly_feature_names = poly.get_feature_names_out(X.columns)
    X_np = X.to_numpy()

    # 3. Normalize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # 4. Encode target
    Y = df["salary_category"].to_pandas()
    le = LabelEncoder().fit(Y)
    Y_enc = le.transform(Y)

    # 5. Cross-validate & fit
    clf = LogisticRegressionCV(
        solver='lbfgs',
        max_iter=1000,
        random_state=0,
        penalty='l2',
        cv=10,
        n_jobs=-1,
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


def test_LG():
    dh = DataHandler()
    X, Y = dh.get_train_data()
    month_encoded_columns = sorted([c for c in X.columns if c.startswith("month_")])
    feature_1_encoded_columns = sorted([c for c in X.columns if c.startswith("feature_1_")])
    job_title_encoded_columns = sorted([c for c in X.columns if c.startswith("job_title_")])
    job_state_encoded_columns = sorted([c for c in X.columns if c.startswith("job_state_")])
    bool_columns = [f"feature_{i}" for i in range(3, 10)] + ["feature_11", "feature_12"]
    quantitative_columns = ["feature_2", "feature_10"]
    job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]

    total_feature_columns = (["months_since_ref", "jd_norm"] + job_state_encoded_columns + bool_columns + quantitative_columns + month_encoded_columns + feature_1_encoded_columns + job_title_encoded_columns + job_desc_cols) # 0.649
    # total_feature_columns = (bool_columns + quantitative_columns + month_encoded_columns + feature_1_encoded_columns + job_title_encoded_columns + job_desc_cols) # 0.634
    # total_feature_columns = (job_desc_cols) # 0.52
    # total_feature_columns = (bool_columns) # 0.50
    # total_feature_columns = (quantitative_columns) # 0.56
    # total_feature_columns = (month_encoded_columns) # 0.44
    # total_feature_columns = (feature_1_encoded_columns) # 0.42
    # total_feature_columns = (job_title_encoded_columns) # 0.42
    # total_feature_columns = (["months_since_ref", "jd_norm", "state_avg_salary"]) # 0.47
    # total_feature_columns = (["jd_norm"]) # 0.39
    # total_feature_columns = (["state_avg_salary"]) # 0.395
    total_feature_columns = (job_state_encoded_columns) # 0.391
    # total_feature_columns = (["months_since_ref"]) # 0.46
    # total_feature_columns = (["months_since_ref"] + bool_columns + quantitative_columns + feature_1_encoded_columns + job_title_encoded_columns + job_desc_cols)  # 0.646
    # total_feature_columns = (["months_since_ref"] + bool_columns + quantitative_columns + feature_1_encoded_columns + job_desc_cols)  # 0.633
    # total_feature_columns = (["months_since_ref"] + bool_columns + quantitative_columns + job_title_encoded_columns + job_desc_cols)  # 0.633
    # total_feature_columns = (["months_since_ref"] + bool_columns + quantitative_columns + job_desc_cols)  # 0.623
    total_feature_columns = (["feature_10"]) # 0.397

    X = X[total_feature_columns]
    Y = Y["salary_category"].to_numpy().ravel().astype(int)

    # 1) build a Pipeline with PCA → LogisticRegression
    pipe = Pipeline([
        # ("pca", PCA()),                 # ← new PCA step
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5_000,
            random_state=0,
        )),
    ])

    # 2) define search space, including PCA n_components and Logistic C
    n_features = X.shape[1]
    search_space = {
        "clf__C": Real(1e-10, 1e-9, prior="log-uniform"),
    }

    # 3) wrap in BayesSearchCV
    bayes_cv = BayesSearchCV(
        pipe,
        search_space,
        n_iter=5,
        cv=10,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=3,
    )

    # 4) fit & report
    bayes_cv.fit(X, Y)
    print("Best CV score:", bayes_cv.best_score_)
    print("Best params:", bayes_cv.best_params_)
    print("Total features before PCA:", n_features)

def test_LG_manual_with_bayes_search():
    df = pd.read_csv("./data/raw_data/train.csv")

    # Binary encode 'feature_1'
    df["feature_1"] = (df["feature_1"] == "B").astype(int)

    # Define feature sets
    job_title_col = ["job_title"]
    passthrough_cols = ["feature_1"] + \
        [f"feature_{i}" for i in range(3, 10)] + ["feature_11", "feature_12", "feature_2", "feature_10"]
    X = df[job_title_col + passthrough_cols]

    # Encode target
    _mapping = {"Low": 0, "Medium": 1, "High": 2}
    Y = df["salary_category"].map(_mapping)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("te", TargetEncoder(), job_title_col),
            ("identity", "passthrough", passthrough_cols),
        ]
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))
    ])

    # Bayesian search space for L2 regularization (inverse of strength)
    param_space = {
        "clf__C": Real(1e-4, 10.0, prior="log-uniform")
    }

    # Bayesian optimizer
    opt = BayesSearchCV(
        pipe,
        param_space,
        n_iter=25,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=42,
    )

    # Fit and print best score
    opt.fit(X, Y)

    print("Best C:", opt.best_params_["clf__C"])
    print("Best CV accuracy:", opt.best_score_)

test_LG_manual_with_bayes_search()



def main():
    test_LG_manual()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()