import numpy
import numpy as np
import polars as pl
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class DataHandler:
    def __init__(self, fill_data, train_data):
        if train_data:
            self.data = pl.read_csv("./data/raw_data/train.csv")
        else:
            self.data = pl.read_csv("./data/raw_data/test.csv")
        self.indexes = self.data["obs"]


        self.data = self.data.with_columns(
            pl.col("job_posted_date").str.strptime(pl.Date, format="%Y/%m"),
            pl.col("job_title").cast(pl.Categorical),
            pl.col("job_state").cast(pl.Categorical),
            pl.col("feature_1").cast(pl.Categorical),
            pl.col(pl.Boolean).cast(pl.Int32)
        )
        if train_data:
            self.data = self.data.with_columns(
                pl.col("salary_category").cast(pl.Categorical),
            )

        feature_cols = [f"feature_{i}" for i in range(1, 13)]
        job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]
        extra_feature = ["state_avg_salary", "job_title", "months_since_ref", "month", "jd_norm"]

        ref = self.data["job_posted_date"].min()
        ref_total = ref.year * 12 + ref.month - 1

        self.data = self.data.with_columns([
            # Linear trend: months since ref
            (
                (pl.col("job_posted_date").dt.year() * 12 +
                 pl.col("job_posted_date").dt.month() - 1)
                - ref_total
            ).alias("months_since_ref"),
            pl.col("job_posted_date").dt.month().alias("month"),
        ])

        self.data = self.data.with_columns(
            np.sqrt(sum(pl.col(c) ** 2 for c in job_desc_cols)).alias("jd_norm")
        )

        self.state_column()

        total_features = feature_cols + job_desc_cols + extra_feature

        self.X = self.data.select(total_features).to_dummies(
            columns=["job_title", "feature_1"]
        )

        if train_data:
            # 4. Encode target
            Y = self.data["salary_category"]
            self.le = LabelEncoder().fit(Y)
            self.Y = self.le.transform(Y)

        if fill_data:
            self.X = self.fill_missing_data(self.X)
        else:
            # self.data = self.data.drop_nulls(subset=["feature_10"])
            # self.X = self.X.fill_null(0)
            a=2

    def fill_missing_data(self, X: pl.DataFrame) -> pl.DataFrame:
        # 1) Split columns into job_desc vs. everything else
        job_desc_cols = [c for c in X.columns if c.startswith("job_desc_")]
        other_cols = [c for c in X.columns if c not in job_desc_cols]

        # 2) Prepare the job_desc block: mask zeros → null so KNN knows what to fill
        jd_masked = X.select(job_desc_cols).with_columns([
            pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in job_desc_cols
        ])

        # 3) Impute that block with KNN
        arr_jd = KNNImputer(
            missing_values=np.nan,
            n_neighbors=5,
            weights="distance"
        ).fit_transform(jd_masked.to_numpy())
        jd_filled = pl.DataFrame({
            col: arr_jd[:, i]
            for i, col in enumerate(job_desc_cols)
        })

        # 4) Fill the other columns with their global mean
        other = X.select(other_cols)
        other_filled = other.with_columns([
            pl.col(c).fill_null(pl.col(c).mean()).alias(c)
            for c in other_cols
        ])

        # 5) Horizontally stitch them back together in original order
        result = pl.concat([other_filled, jd_filled], how="horizontal")
        return result.select(X.columns)

    def state_column(self):
        abbr_to_mean = {
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
        global_mean = sum(abbr_to_mean.values()) / len(abbr_to_mean)
        self.data = self.data.with_columns(
            pl.col("job_state")
            .cast(pl.Utf8)
            .replace(abbr_to_mean, default=global_mean)  # map codes → mean salary or null
            .fill_null(global_mean)  # fill unmapped or null with global average
            .alias("state_avg_salary")
        ).drop("job_state")