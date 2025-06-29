"""
Custom transformers for feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomJobStateFeature1Transformer(BaseEstimator, TransformerMixin):
    """
    Transforms job_state and feature_1 columns.

    - feature_1: binarize to 1 if == "B" or "C", else 0
    - job_state: map US state abbreviation to average salary value
    """

    def __init__(self):
        # State to average salary mapping
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
        """Transform job_state and feature_1."""
        X_t = X.copy()

        # Transform feature_1 to binary
        X_t["feature_1"] = X_t["feature_1"].isin(["B", "C"]).astype(int)

        # Map job_state to average salary
        X_t["job_state"] = X_t["job_state"].map(self.abbr_to_mean)

        return X_t[["job_state", "feature_1"]]

    def get_feature_names_out(self, input_features=None):
        return ["job_state", "feature_1"]

    def set_output(self, transform=None):
        return self


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from job_posted_date.

    Creates:
    - months_since_first: months between date and earliest date in training
    - month_of_year: month number (1-12) - optional
    - month_target_mean/std: target statistics per month - optional
    """

    def __init__(self, extract_month_stats=False):
        self.extract_month_stats = extract_month_stats
        self.min_date_ = None
        self.month_stats_ = None
        self.global_mean_ = None
        self.global_std_ = None

    def fit(self, X, y=None):
        """Fit to learn min date and optionally month statistics."""
        dates = pd.to_datetime(X["job_posted_date"], errors="coerce")
        valid = dates.dropna()

        # Set minimum date
        self.min_date_ = valid.min() if len(valid) > 0 else pd.Timestamp.today()

        # Optionally compute month statistics
        if self.extract_month_stats and y is not None:
            months = dates.dt.month
            df = pd.DataFrame({"month": months, "target": y})
            df_valid = df.dropna(subset=["month"])

            if df_valid.shape[0] > 0:
                self.global_mean_ = df_valid["target"].mean()
                self.global_std_ = df_valid["target"].std()

                # Group statistics by month
                grp = df_valid.groupby("month")["target"]
                stats = grp.agg(['mean', 'std', 'count'])
                stats['std'] = stats['std'].fillna(0.0)

                self.month_stats_ = {
                    int(month): (row['mean'], row['std'])
                    for month, row in stats.iterrows()
                }
            else:
                self.global_mean_ = np.nanmean(y) if y is not None else 0
                self.global_std_ = np.nanstd(y) if y is not None else 1
                self.month_stats_ = {}

        return self

    def transform(self, X):
        """Transform dates to features."""
        X_t = X.copy()
        dates = pd.to_datetime(X_t["job_posted_date"], errors="coerce")

        # Calculate months since first date
        def compute_months_since(d):
            if pd.isna(d) or pd.isna(self.min_date_):
                return np.nan
            year_diff = d.year - self.min_date_.year
            month_diff = d.month - self.min_date_.month
            return year_diff * 12 + month_diff

        X_t['months_since_first'] = dates.apply(compute_months_since)

        # Optionally add month statistics
        if self.extract_month_stats and self.month_stats_ is not None:
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
                        mean_val, std_val = self.month_stats_[m_int]
                        mean_list.append(mean_val)
                        std_list.append(std_val)
                    else:
                        mean_list.append(self.global_mean_)
                        std_list.append(self.global_std_)

            X_t['month_target_mean'] = mean_list
            X_t['month_target_std'] = std_list
            X_t['month_of_year'] = months

            return X_t[['months_since_first', 'month_of_year',
                        'month_target_mean', 'month_target_std']]
        else:
            return X_t[['months_since_first']]

    def get_feature_names_out(self, input_features=None):
        if self.extract_month_stats:
            return ['months_since_first', 'month_of_year',
                    'month_target_mean', 'month_target_std']
        else:
            return ['months_since_first']

    def set_output(self, transform=None):
        return self


class PolynomialFeaturesDataFrame(TransformerMixin, BaseEstimator):
    """
    Polynomial features transformer that preserves DataFrame structure.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_ = None
        self.output_feature_names_ = None

    def fit(self, X, y=None):
        from sklearn.preprocessing import PolynomialFeatures

        # Get input feature names
        if hasattr(X, "columns"):
            self.input_feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            n_features = X_arr.shape[1]
            self.input_feature_names_ = [f"feature_{i}" for i in range(n_features)]

        # Fit polynomial features
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.poly_.fit(X_arr)

        # Generate output feature names
        self.output_feature_names_ = list(
            self.poly_.get_feature_names_out(self.input_feature_names_)
        )

        return self

    def transform(self, X):
        if hasattr(X, "values"):
            X_arr = X.values
            index = X.index
        else:
            X_arr = X
            index = None

        arr_poly = self.poly_.transform(X_arr)

        return pd.DataFrame(
            arr_poly,
            columns=self.output_feature_names_,
            index=index
        )

    def get_feature_names_out(self, input_features=None):
        return self.output_feature_names_