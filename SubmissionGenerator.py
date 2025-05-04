import numpy as np
import pandas as pd
import polars as pl


class SubmissionGenerator:
    def __init__(self, pipeline, label_encoder, id_col="obs"):
        """
        pipeline     : fitted sklearn/skorch estimator with .predict()
        label_encoder: fitted LabelEncoder to inverse_transform
        id_col       : name of the ID column in test DataFrame
        """
        self.pipe = pipeline
        self.le = label_encoder
        self.id_col = id_col

    def generate(self, X_test, ids=None, submission_csv_path="./data/submissions/submission.csv"):
        """
        Generate submission file using predictions from the model

        Parameters:
        -----------
        data_handler: DataHandler
            Instance of DataHandler used to process training data
        X_test: pl.DataFrame or np.ndarray
            Processed test features, formatted like the training data
        ids: list, optional
            List of IDs for the test data. If None, extracts from X_test
        submission_csv_path: str
            Path where the submission CSV will be saved
        """
        # If X_test is a DataFrame, convert to numpy array
        if isinstance(X_test, pl.DataFrame):
            # If ids not provided, extract from DataFrame
            if ids is None and self.id_col in X_test.columns:
                ids = X_test[self.id_col].to_list()
            X_test_np = X_test.to_numpy().astype(np.float32)
        else:
            X_test_np = X_test.astype(np.float32)

        # Make predictions
        y_num = self.pipe.predict(X_test_np)

        # Transform to category names
        y_cat = self.le.inverse_transform(y_num.astype(int))

        # Create submission file
        sub = pd.DataFrame({
            self.id_col: ids,
            "salary_category": y_cat
        })
        sub.to_csv(submission_csv_path, index=False)
        print(f"Submission saved to {submission_csv_path}")
