import os

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class DataHandler:
    # Adjust this to your actual absolute data directory
    DATA_DIR = r"C:\Users\StefanConstantin\Documents\Git\Python\Engineering_Salary_Prediction\data\raw_data"

    def __init__(self, n_job_clusters = 0, use_cluster_probabilities=True):
        """
        Reads train.csv and test.csv from DATA_DIR,
        splits into raw X/y, and builds the preprocessing pipeline.
        """

        self.n_job_clusters = n_job_clusters
        self.use_cluster_probabilities = use_cluster_probabilities

        # 1) Read raw CSVs into pandas DataFrames
        train_path = os.path.join(self.DATA_DIR, "train.csv")
        test_path = os.path.join(self.DATA_DIR, "test.csv")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # Capture original indices
        self.train_index = self.train_df.index.copy()
        self.test_index = self.test_df.index.copy() + 1281

        # 2) Prepare y for train
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        self.train_df["salary_category"] = self.train_df["salary_category"].map(mapping)
        self.y_cols = ["salary_category"]

        # 3) Split raw X and y
        self.X_train_raw = self.train_df.drop(columns=["salary_category"])
        self.y_train = self.train_df[self.y_cols].copy()
        self.X_test_raw = self.test_df.copy()

        if self.n_job_clusters == 0:
            # 4) Build and store the pipeline
            self.pipeline = self.create_pipeline()
        else:
            self.pipeline = self.create_cluster_pipeline()

    def create_pipeline(self):
        """
        Builds and returns the preprocessing pipeline:
        - FeatureExtractor
        - ColumnTransformer for feature creation (js_f1, one-hot job_title, date features, passthroughs)
        - SimpleImputer(mean) on all columns
        - ColumnTransformer to scale non-binary numeric columns
        Each step outputs a pandas DataFrame (sklearn ≥1.6).
        """
        # Instantiate FeatureExtractor to get column lists
        fe = FeatureExtractor()
        bool_cols = fe.bool_columns
        quant_cols = fe.quantitative_columns
        job_desc_cols = fe.job_desc_cols

        # 1) preprocessor: feature creation
        preprocessor_ct = ColumnTransformer(
            transformers=[
                ("js_f1", Pipeline([("custom", CustomJobStateFeature1Transformer())]),
                 ["job_state", "feature_1"]),
                ("title", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                 ["job_title"]),
                ("date", DateFeaturesTransformer(), ["job_posted_date"]),
                ("bools", "passthrough", bool_cols),
                ("quants", "passthrough", quant_cols),
                ("jobdesc", "passthrough", job_desc_cols),
            ],
            remainder="drop"
        ).set_output(transform="pandas")

        # 2) imputer: mean on all columns
        imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")

        # 3) scaler: scale only non-binary numeric columns
        scale_cols = (
                ["js_f1__job_state", "date__months_since_first", "date__month_of_year"] +
                [f"quants__{col}" for col in quant_cols] +
                [f"jobdesc__{col}" for col in job_desc_cols]
        )
        scaler_ct = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), scale_cols),
            ],
            remainder="passthrough"
        ).set_output(transform="pandas")

        pipeline = Pipeline([
            ("feature_extractor", FeatureExtractor()),  # returns DataFrame
            ("preprocessor", preprocessor_ct),          # returns DataFrame
            ("imputer", imputer),                      # returns DataFrame
            ("scaler", scaler_ct),                     # returns DataFrame
        ])
        return pipeline

    def create_cluster_pipeline(self):
        """
        Builds and returns the preprocessing pipeline:
        - FeatureExtractor
        - ColumnTransformer for feature creation (including job clustering)
        - SimpleImputer(mean) on all columns
        - ColumnTransformer to scale non-binary numeric columns dynamically.
        """
        # This import is needed for the dynamic selector
        from sklearn.compose import make_column_selector

        # Instantiate FeatureExtractor to get column lists
        fe = FeatureExtractor()
        bool_cols = fe.bool_columns
        quant_cols = fe.quantitative_columns
        job_desc_cols = fe.job_desc_cols

        # 1) preprocessor: feature creation with clustering
        preprocessor_ct = ColumnTransformer(
            transformers=[
                ("js_f1", Pipeline([("custom", CustomJobStateFeature1Transformer())]),
                 ["job_state", "feature_1"]),
                ("title", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                 ["job_title"]),
                ("date", DateFeaturesTransformer(), ["job_posted_date"]),
                ("bools", "passthrough", bool_cols),
                ("quants", "passthrough", quant_cols),
                ("jobdesc_cluster", JobDescriptionClusterTransformer(
                    n_clusters=self.n_job_clusters,  # This is varied by BayesSearch
                    use_probabilities=self.use_cluster_probabilities,
                    random_state=42
                ), job_desc_cols),
            ],
            remainder="drop"
        ).set_output(transform="pandas")

        # 2) imputer: mean on all columns
        imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")

        # 3) scaler: scale only non-binary numeric columns using a dynamic selector
        # This regex pattern matches all numeric columns that should be scaled,
        # including the cluster probability columns, regardless of how many there are.
        scaler_selector = make_column_selector(
            pattern=(r"js_f1__job_state|date__|quants__|jobdesc_cluster__"),
            dtype_include=np.number
        )

        scaler_ct = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), scaler_selector),
            ],
            remainder="passthrough"  # Automatically passes through binary/categorical columns
        ).set_output(transform="pandas")

        pipeline = Pipeline([
            ("feature_extractor", FeatureExtractor()),
            ("preprocessor", preprocessor_ct),
            ("imputer", imputer),
            ("scaler", scaler_ct),  # Use the new dynamic scaler
        ])
        return pipeline

    def get_train_data_raw(self):
        """
        Returns raw X_train, y_train, and train_index (before any processing).
        """
        return self.X_train_raw.copy(), self.y_train.copy(), self.train_index

    def get_test_data_raw(self):
        """
        Returns raw X_test and test_index (before any processing).
        """
        return self.X_test_raw.copy(), self.test_index

    def analyze_clusters(self):
        """
        Analyze the job description clusters to understand what they represent.
        Must be called after fitting the pipeline.
        """
        # Get the cluster transformer from the pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        cluster_transformer = None

        for name, transformer, _ in preprocessor.transformers_:
            if name == 'jobdesc_cluster':
                cluster_transformer = transformer
                break

        if cluster_transformer is None:
            print("Cluster transformer not found in pipeline")
            return

        if not hasattr(cluster_transformer, 'kmeans') or not hasattr(cluster_transformer.kmeans, 'cluster_centers_'):
            print("Pipeline must be fitted before analyzing clusters")
            return

        # Get cluster assignments for training data
        X_train_raw, y_train, _ = self.get_train_data_raw()
        job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]
        X_job_desc = X_train_raw[job_desc_cols]

        # Preprocess job descriptions
        X_job_desc_array = X_job_desc.values
        X_job_desc_array = np.where(X_job_desc_array == 0, np.nan, X_job_desc_array)
        X_imputed = cluster_transformer.imputer.transform(X_job_desc_array)
        X_scaled = cluster_transformer.scaler.transform(X_imputed)

        # Get cluster assignments
        cluster_labels = cluster_transformer.kmeans.predict(X_scaled)

        # Analyze salary distribution per cluster
        print("\nCluster Analysis:")
        print("-" * 50)

        salary_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        y_values = y_train.values.ravel()

        for cluster_id in range(self.n_job_clusters):
            mask = cluster_labels == cluster_id
            cluster_size = np.sum(mask)

            print(f"\nCluster {cluster_id} (n={cluster_size}):")

            cluster_salaries = y_values[mask]
            for salary_val, salary_name in salary_map.items():
                count = np.sum(cluster_salaries == salary_val)
                pct = 100 * count / cluster_size if cluster_size > 0 else 0
                print(f"  {salary_name}: {count} ({pct:.1f}%)")

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract features from a pandas DataFrame."""
    def __init__(self):
        self.data_columns = ["job_posted_date"]
        self.categorical_columns = ["job_title", "feature_1", "job_state"]
        self.bool_columns = [f"feature_{i}" for i in range(3, 10)] + ["feature_11", "feature_12"]
        self.quantitative_columns = ["feature_2", "feature_10"]
        self.job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]

        self.all_features = (
            self.data_columns
            + self.categorical_columns
            + self.bool_columns
            + self.quantitative_columns
            + self.job_desc_cols
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: pandas DataFrame. Selects needed columns, replaces zeros in job_desc cols with NaN,
        ensures job_posted_date is datetime, and returns a DataFrame.
        """
        # Select only columns that exist in X
        missing = [col for col in self.all_features if col not in X.columns]
        if missing:
            raise ValueError(f"Missing columns in input DataFrame: {missing}")
        df = X[self.all_features].copy()

        # Replace zeros with NaN for job_desc columns
        for col in self.job_desc_cols:
            # Replace exactly 0; keep other falsy values if any
            df[col] = df[col].replace(0, np.nan)

        # Ensure job_posted_date is datetime
        if not np.issubdtype(df["job_posted_date"].dtype, np.datetime64):
            df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], format="%Y/%m", errors="coerce")

        return df

    def set_output(self, transform=None):
        return self


class CustomJobStateFeature1Transformer(BaseEstimator, TransformerMixin):
    """
    Transforms:
    - feature_1: binarize to 1 if == "B", else 0 (including NaN → 0 or keep as 0).
    - job_state: map US state abbreviation to mean value via abbr_to_mean; unmapped → NaN.
    """
    def __init__(self):
        # mapping dict for job_state
        self.abbr_to_mean = {
            "NH": 97046, "MA": 94651, "OR": 94286, "PA": 93864, "NY": 93391,
            "MD": 92244, "WV": 90701, "TX": 89407, "VT": 89074, "NV": 88635,
            "CA": 87957, "ND": 87795, "VA": 86648, "ME": 86128, "WI": 85718,
            "DE": 85112, "NM": 84814, "KS": 84508, "OK": 84153, "WA": 83605,
            "AZ": 83199, "TN": 82354, "ID": 82316, "MS": 80873, "AR": 79907,
            "KY": 78545, "SC": 78256, "WY": 77540, "UT": 77455, "AL": 76867,
            "RI": 76628, "GA": 76574, "IL": 76205, "MN": 75504, "MT": 75309,
            "NJ": 74480, "IN": 73188, "IA": 72526, "CT": 72447, "NC": 71590,
            "CO": 71342, "MO": 70645, "FL": 70393, "OH": 68590, "HI": 67412,
            "LA": 67027, "AK": 65498, "NE": 65123, "SD": 62823, "MI": 61400
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: pandas DataFrame with columns 'feature_1' and 'job_state'.
        Returns DataFrame with:
          - 'feature_1': 1 if original == "B", else 0.
          - 'job_state': mapped to abbr_to_mean, unmapped → NaN.
        """
        X_t = X.copy()

        # feature_1 → binary
        # Treat missing or non-"B" as 0
        X_t["feature_1"] = (X_t["feature_1"] == "B").astype(int)

        # job_state → mean mapping
        # Map via dict; unmapped (including NaN) → NaN
        X_t["job_state"] = X_t["job_state"].map(self.abbr_to_mean)

        return X_t[["job_state", "feature_1"]]

    def set_output(self, transform=None):
        return self


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    From 'job_posted_date', add:
      - months_since_first: months between job_posted_date and earliest date seen in fit.
      - month_target_mean: mean(y) for that month in training data (with optional smoothing).
      - month_target_std: std(y) for that month in training data.
    """
    def __init__(self, col="job_posted_date", smoothing=0.0, ddof=0):
        self.col = col
        self.smoothing = smoothing
        self.ddof = ddof
        # to be set in fit:
        self.min_date_ = None
        self.global_mean_ = None
        self.global_std_ = None
        self.month_stats_ = None  # dict: month -> (enc_mean, enc_std, count)

    def fit(self, X, y):
        # X: DataFrame-like, y: array-like
        dates = pd.to_datetime(X[self.col], errors="coerce")
        valid = dates.dropna()
        # Determine min_date_ for months_since_first
        if len(valid) == 0:
            self.min_date_ = pd.Timestamp.today().normalize()
        else:
            self.min_date_ = valid.min()

        # Prepare month and target
        months = dates.dt.month  # 1..12 or NaN
        df = pd.DataFrame({"month": months, "target": y})
        df_valid = df.dropna(subset=["month"])
        if df_valid.shape[0] == 0:
            # No valid months: global stats from y
            arr = np.asarray(y)
            self.global_mean_ = np.nanmean(arr)
            self.global_std_ = np.nanstd(arr, ddof=self.ddof)
            self.month_stats_ = {}
            return self

        # Compute global mean/std on valid
        self.global_mean_ = df_valid["target"].mean()
        self.global_std_ = df_valid["target"].std(ddof=self.ddof)

        # Group by month
        grp = df_valid.groupby("month")["target"]
        stats = grp.agg(['mean', 'std', 'count']).rename(
            columns={'mean': 'month_mean', 'std': 'month_std', 'count': 'count'}
        )
        # Replace NaN std (e.g., single sample) with 0
        stats['month_std'] = stats['month_std'].fillna(0.0)

        # Smoothing for mean if requested
        if self.smoothing and self.smoothing > 0:
            stats['month_mean_smoothed'] = (
                stats['count'] * stats['month_mean'] + self.smoothing * self.global_mean_
            ) / (stats['count'] + self.smoothing)
            stats['enc_mean'] = stats['month_mean_smoothed']
        else:
            stats['enc_mean'] = stats['month_mean']
        # Keep std un-smoothed (or implement smoothing if desired)
        stats['enc_std'] = stats['month_std']

        # Store mapping month -> (enc_mean, enc_std)
        self.month_stats_ = {
            int(month): (row.enc_mean, row.enc_std)
            for month, row in stats.iterrows()
        }
        return self

    def transform(self, X):
        X_t = X.copy()
        dates = pd.to_datetime(X_t[self.col], errors="coerce")
        # Compute months_since_first
        def compute_months_since(d):
            if pd.isna(d) or pd.isna(self.min_date_):
                return np.nan
            year_diff = d.year - self.min_date_.year
            month_diff = d.month - self.min_date_.month
            return year_diff * 12 + month_diff

        X_t['months_since_first'] = dates.apply(compute_months_since)

        # Compute month_target_mean and month_target_std per row
        months = dates.dt.month
        mean_list = []
        std_list = []
        for m in months:
            if pd.isna(m):
                mean_list.append(self.global_mean_)
                std_list.append(self.global_std_)
            else:
                m_int = int(m)
                if m_int in self.month_stats_:
                    enc_mean, enc_std = self.month_stats_[m_int]
                    mean_list.append(enc_mean)
                    std_list.append(enc_std)
                else:
                    # unseen month (unlikely): fallback
                    mean_list.append(self.global_mean_)
                    std_list.append(self.global_std_)
        X_t['month_target_mean'] = mean_list
        X_t['month_target_std'] = std_list

        # return X_t[['months_since_first', 'month_target_mean', 'month_target_std']]
        return X_t[['months_since_first']]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'months_since_first',
            # 'month_target_mean',
            # 'month_target_std'
        ])

    def set_output(self, transform=None):
        return self


class ValidationTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy transformer to validate the preprocessing pipeline output.
    Prints useful information about the data after preprocessing.
    """

    def __init__(self, step_name="Validation"):
        self.step_name = step_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\n=== {self.step_name} Step ===")
        print(f"Data type: {type(X)}")
        print(f"Shape: {X.shape}")
        print(f"Columns ({len(X.columns)}): {list(X.columns)}")

        # Check for missing values
        missing_count = X.isnull().sum().sum() if hasattr(X, 'isnull') else np.isnan(X).sum()
        print(f"Missing values: {missing_count}")

        # Check data types and ranges
        if hasattr(X, 'dtypes'):
            print(f"Data types: {X.dtypes.value_counts().to_dict()}")
            print(f"Numeric summary:")
            print(X.describe().iloc[[0, 1, 3, 7]].round(3))  # count, mean, 50%, max
        else:
            print(f"Array min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}")

        # Check for infinite values
        if hasattr(X, 'values'):
            inf_count = np.isinf(X.values).sum()
        else:
            inf_count = np.isinf(X).sum()
        print(f"Infinite values: {inf_count}")

        print("=" * 50)
        return X

    def set_output(self, transform=None):
        return self


class JobDescriptionClusterTransformer(BaseEstimator, TransformerMixin):
    """
    Transform job description columns into cluster-based features.
    Replaces 300 job_desc columns with cluster probabilities and distances.
    """

    def __init__(self, n_clusters=5, use_probabilities=True, random_state=42):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        use_probabilities : bool
            If True, return cluster probabilities (soft assignment)
            If False, return one-hot encoded cluster assignments
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.use_probabilities = use_probabilities
        self.random_state = random_state

        # Initialize preprocessing and clustering
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    def fit(self, X, y=None):
        """Fit the clustering model on job description features."""
        # X should already be the job description columns only
        X_array = X.values if hasattr(X, 'values') else X

        # Replace zeros with NaN (already done in FeatureExtractor, but ensure)
        X_array = np.where(X_array == 0, np.nan, X_array)

        # Preprocess
        X_imputed = self.imputer.fit_transform(X_array)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Fit clustering
        self.kmeans.fit(X_scaled)

        return self

    def transform(self, X):
        """Transform job descriptions into cluster features."""
        X_array = X.values if hasattr(X, 'values') else X

        # Replace zeros with NaN
        X_array = np.where(X_array == 0, np.nan, X_array)

        # Preprocess
        X_imputed = self.imputer.transform(X_array)
        X_scaled = self.scaler.transform(X_imputed)

        # Get column names from input if it's a DataFrame
        if hasattr(X, 'columns'):
            index = X.index
        else:
            index = None

        if self.use_probabilities:
            # Calculate distances to each cluster center
            distances = self.kmeans.transform(X_scaled)

            # Convert distances to probabilities (soft assignment)
            neg_distances = -distances
            exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
            probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

            # Create DataFrame with meaningful column names
            feature_names = [f'job_cluster_prob_{i}' for i in range(self.n_clusters)]
            result_df = pd.DataFrame(probabilities, columns=feature_names, index=index)

            # Add minimum distance as additional feature
            result_df['job_cluster_min_dist'] = np.min(distances, axis=1)

        else:
            # Hard cluster assignment (one-hot encoding)
            cluster_labels = self.kmeans.predict(X_scaled)

            # Create one-hot encoding
            result_array = np.zeros((X_array.shape[0], self.n_clusters))
            result_array[np.arange(len(cluster_labels)), cluster_labels] = 1

            feature_names = [f'job_cluster_{i}' for i in range(self.n_clusters)]
            result_df = pd.DataFrame(result_array, columns=feature_names, index=index)

        return result_df

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if self.use_probabilities:
            names = [f'job_cluster_prob_{i}' for i in range(self.n_clusters)]
            names.append('job_cluster_min_dist')
        else:
            names = [f'job_cluster_{i}' for i in range(self.n_clusters)]
        return np.array(names)