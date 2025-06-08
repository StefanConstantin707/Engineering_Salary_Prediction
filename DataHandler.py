import numpy
import numpy as np
import polars as pl
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.backends.opt_einsum import strategy


class DataHandler:
    def __init__(self, ordinal_classification=False):
        self.data = pl.read_csv("../data/raw_data/train.csv")
        self.test_data = pl.read_csv("../data/raw_data/test.csv")
        self.ordinal_classification = ordinal_classification

        self.data_columns = ["job_posted_date"]
        self.categorical_columns = ["job_title", "feature_1", "job_state"]
        self.bool_columns = [f"feature_{i}" for i in range(3, 10)] + ["feature_11", "feature_12"]
        self.quantitative_columns = ["feature_2", "feature_10"]
        self.job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]
        self.responder_column = ["salary_category"]


        self.data = self.data.with_columns(
            pl.col("job_posted_date").str.strptime(pl.Date, format="%Y/%m"),
            pl.col("salary_category").cast(pl.Categorical)
        )

        self.test_data = self.test_data.with_columns(
            pl.col("job_posted_date").str.strptime(pl.Date, format="%Y/%m")
        )

        # Encode Y label column
        if self.ordinal_classification:
            # self.label_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
            # self.data = self.data.with_columns(
            #     pl.col("salary_category").cast(pl.Utf8).replace(self.label_map).cast(pl.Float32).alias("salary_category")
            # )
            n_thresholds = 5

            self.data = self.data.with_columns([
                # y_0 = 1.0 for Medium or High, else 0.0
                pl.col("salary_category")
                .is_in(["Medium", "High"])
                .cast(pl.Float32)
                .alias("y_0"),

                # y_1 = 1.0 for High, else 0.0
                (pl.col("salary_category") == "High")
                .cast(pl.Float32)
                .alias("y_1"),
            ]).drop("salary_category")
        else:
        # Fit LabelEncoder on numpy array
            values = self.data["salary_category"]
            self.le = LabelEncoder().fit(values)
            labels = self.le.transform(values)
            self.data = self.data.with_columns(
                pl.Series("salary_category", labels)
            )

        # Compute the earliest posted date across both datasets
        min_data = self.data.select(pl.col("job_posted_date").min()).item()
        min_test = self.test_data.select(pl.col("job_posted_date").min()).item()
        self.first_posted_date = min(min_data, min_test)

        self.data = self.preprocess_data(self.data)
        self.test_data = self.preprocess_data(self.test_data)

        self.scaler = StandardScaler()
        self.data = self.normalize_data(self.data, fit=True)
        self.test_data = self.normalize_data(self.test_data, fit=False)

    def preprocess_data(self, data):
        # change state column to state salary column and fill nulls
        data = self.state_column(data)

        # fill missing values
        data = self.fill_missing_data(data)

        data = data.with_columns(
            pl.col("job_title").cast(pl.Categorical),
            pl.col("feature_1").cast(pl.Categorical),
            pl.col(pl.Boolean).cast(pl.Int32)
        )

        # add months since reference column and month column
        data = self.date_column(data)

        # Encode categorical columns
        data = data.to_dummies(
            columns=["job_title", "feature_1", "month"]
        )

        # fill job_desc columns
        data = self.fill_job_desc_columns(data)

        # add job_desc_length column and unitize the job_description columns
        data = self.job_desc_columns(data)

        return data

    def fill_missing_data(self, data):
        # missing test data:
        # job_state: 13/854, feature_10: 314/854, job_desc: 112/854

        # missing train data:
        # job_posted_date: 1/1253, job_state: 27/1253, feature_10: 446/1253, job_desc: 166/1253

        # 1) Fill job_posted_date nulls with the most frequent date (mode)
        mode_date = data.select(
            pl.col("job_posted_date").mode()
        ).to_series()[0]
        data = data.with_columns(
            pl.col("job_posted_date").fill_null(mode_date)
        )

        # 2) Fill feature_10 nulls with the mean
        mean_feature = data.select(pl.col("feature_10").mean()).to_series()[0]
        data = data.with_columns(
            pl.col("feature_10").fill_null(mean_feature)
        )
        return data

    def fill_job_desc_columns(self, data):
        # 3) Replace zeros in job_desc_cols with nulls
        data = data.with_columns([
            pl.when(pl.col(col) == 0)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in self.job_desc_cols
        ])

        # 4) Convert to pandas for KNN imputation
        df_pd = data.to_pandas()

        imputer = KNNImputer(
            missing_values=np.nan,
            n_neighbors=5,
            weights="distance"
        )
        filled_array = imputer.fit_transform(df_pd)

        # 5) Convert back to Polars
        filled_pd = df_pd.copy()
        filled_pd[:] = filled_array
        data = pl.from_pandas(filled_pd)

        return data

    def state_column(self, data):
        # abbr_to_mean = {
        #     "NH": 97046, "MA": 94651, "OR": 94286, "PA": 93864, "NY": 93391,
        #     "MD": 92244, "WV": 90701, "TX": 89407, "VT": 89074, "NV": 88635,
        #     "CA": 87957, "ND": 87795, "VA": 86648, "ME": 86128, "WI": 85718,
        #     "DE": 85112, "NM": 84814, "KS": 84508, "OK": 84153, "WA": 83605,
        #     "AZ": 83199, "TN": 82354, "ID": 82316, "MS": 80873, "AR": 79907,
        #     "KY": 78545, "SC": 78256, "WY": 77540, "UT": 77455, "AL": 76867,
        #     "RI": 76628, "GA": 76574, "IL": 76205, "MN": 75504, "MT": 75309,
        #     "NJ": 74480, "IN": 73188, "IA": 72526, "CT": 72447, "NC": 71590,
        #     "CO": 71342, "MO": 70645, "FL": 70393, "OH": 68590, "HI": 67412,
        #     "LA": 67027, "AK": 65498, "NE": 65123, "SD": 62823, "MI": 61400
        # }
        # global_mean = sum(abbr_to_mean.values()) / len(abbr_to_mean)
        # data = data.with_columns(
        #     pl.col("job_state")
        #     .cast(pl.Utf8)
        #     .replace(abbr_to_mean, default=global_mean)  # map codes → mean salary or null
        #     .fill_null(global_mean)  # fill unmapped or null with global average
        #     .alias("state_avg_salary")
        # ).drop("job_state")

        data = data.to_dummies(
            columns=["job_state"]
        )

        return data

    def job_desc_columns(self, data):
        data = data.with_columns(
            np.sqrt(sum(pl.col(c) ** 2 for c in self.job_desc_cols)).alias("jd_norm")
        )

        return data

    def date_column(self, data):
        ref_total = self.first_posted_date.year * 12 + self.first_posted_date.month - 1

        data = data.with_columns([
            # Linear trend: months since ref
            (
                    (pl.col("job_posted_date").dt.year() * 12 +
                     pl.col("job_posted_date").dt.month() - 1)
                    - ref_total
            ).alias("months_since_ref"),
            pl.col("job_posted_date").dt.month().alias("month"),
        ]).drop("job_posted_date")

        return data

    def normalize_data(self, data: pl.DataFrame, fit: bool) -> pl.DataFrame:
        # select numeric cols: months_since_ref, quantitative, job_desc
        # num_cols = ["months_since_ref", "jd_norm", "state_avg_salary"] + self.quantitative_columns + self.job_desc_cols
        num_cols = ["months_since_ref", "jd_norm"] + self.quantitative_columns + self.job_desc_cols
        df_pd = data.select(num_cols).to_pandas()
        if fit:
            scaled = self.scaler.fit_transform(df_pd)
        else:
            scaled = self.scaler.transform(df_pd)

        # build Polars columns for scaled data
        scaled_series = [pl.Series(col, scaled[:, i]) for i, col in enumerate(num_cols)]
        # drop old and add scaled
        return data.drop(num_cols).with_columns(scaled_series)

    def get_train_data(self):
        month_encoded_columns = sorted([c for c in self.data.columns if c.startswith("month_")])
        feature_1_encoded_columns = sorted([c for c in self.data.columns if c.startswith("feature_1_")])
        job_title_encoded_columns = sorted([c for c in self.data.columns if c.startswith("job_title_")])
        job_state_encoded_columns = sorted([c for c in self.data.columns if c.startswith("job_state_")])

        total_feature_columns = (["months_since_ref", "jd_norm"] + job_state_encoded_columns + self.bool_columns + self.quantitative_columns +
                                 month_encoded_columns + feature_1_encoded_columns + job_title_encoded_columns + self.job_desc_cols)

        X = self.data[total_feature_columns]
        if self.ordinal_classification:
            Y = self.data.select(["y_0", "y_1"])
        else:
            Y = self.data[self.responder_column]

        return X, Y

    def get_test_data(self):
        month_encoded_columns = sorted([c for c in self.test_data.columns if c.startswith("month_")])
        feature_1_encoded_columns = sorted([c for c in self.test_data.columns if c.startswith("feature_1_")])
        job_title_encoded_columns = sorted([c for c in self.test_data.columns if c.startswith("job_title_")])
        job_state_encoded_columns = sorted([c for c in self.test_data.columns if c.startswith("job_state_")])

        total_feature_columns = (["months_since_ref", "jd_norm"] + job_state_encoded_columns + self.bool_columns + self.quantitative_columns +
                                 month_encoded_columns + feature_1_encoded_columns + job_title_encoded_columns + self.job_desc_cols)

        X = self.test_data[total_feature_columns]

        return X

def remap_salary_category(code):
    label_map = {"Low": 0, "Medium": 0.5, "High": 1}
    if code is None:
        return label_map[None]
    return label_map.get(code, "unknown")

