"""
Unit tests for data handling components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_handler import DataHandler
from src.data.feature_extractor import FeatureExtractor
from src.data.transformers import (
    CustomJobStateFeature1Transformer,
    DateFeaturesTransformer
)


class TestFeatureExtractor:
    """Test the FeatureExtractor class."""

    def setup_method(self):
        """Set up test data."""
        self.fe = FeatureExtractor()

        # Create sample data
        n_samples = 10
        data = {
            'job_posted_date': ['2024/01'] * n_samples,
            'job_title': ['Engineer'] * n_samples,
            'feature_1': ['A', 'B', 'C'] * (n_samples // 3) + ['A'] * (n_samples % 3),
            'job_state': ['CA', 'NY', 'TX'] * (n_samples // 3) + ['CA'] * (n_samples % 3),
        }

        # Add boolean features
        for i in range(3, 10):
            data[f'feature_{i}'] = np.random.randint(0, 2, n_samples)
        data['feature_11'] = np.random.randint(0, 2, n_samples)
        data['feature_12'] = np.random.randint(0, 2, n_samples)

        # Add quantitative features
        data['feature_2'] = np.random.randn(n_samples)
        data['feature_10'] = np.random.randint(0, 20, n_samples)

        # Add job description features
        for i in range(1, 301):
            data[f'job_desc_{i:03d}'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

        self.df = pd.DataFrame(data)

    def test_feature_extraction(self):
        """Test basic feature extraction."""
        result = self.fe.transform(self.df)

        # Check all features are present
        assert len(result.columns) == len(self.fe.all_features)

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['job_posted_date'])

    def test_zero_replacement(self):
        """Test that zeros are replaced with NaN in job_desc columns."""
        result = self.fe.transform(self.df)

        # Check that zeros are replaced
        for col in self.fe.job_desc_cols:
            if col in result.columns:
                assert (result[col] == 0).sum() == 0


class TestCustomTransformers:
    """Test custom transformer classes."""

    def test_job_state_feature1_transformer(self):
        """Test the job state and feature_1 transformer."""
        transformer = CustomJobStateFeature1Transformer()

        # Create test data
        data = pd.DataFrame({
            'job_state': ['CA', 'NY', 'TX', 'INVALID'],
            'feature_1': ['A', 'B', 'C', None]
        })

        result = transformer.transform(data)

        # Check feature_1 binarization
        assert result.loc[0, 'feature_1'] == 0  # A -> 0
        assert result.loc[1, 'feature_1'] == 1  # B -> 1
        assert result.loc[2, 'feature_1'] == 1  # C -> 1
        assert result.loc[3, 'feature_1'] == 0  # None -> 0

        # Check job_state mapping
        assert result.loc[0, 'job_state'] == 87957  # CA
        assert result.loc[1, 'job_state'] == 93391  # NY
        assert pd.isna(result.loc[3, 'job_state'])  # INVALID -> NaN

    def test_date_features_transformer(self):
        """Test the date features transformer."""
        transformer = DateFeaturesTransformer()

        # Create test data
        dates = pd.DataFrame({
            'job_posted_date': ['2024/01', '2024/06', '2024/12', None]
        })

        # Fit and transform
        transformer.fit(dates)
        result = transformer.transform(dates)

        # Check months_since_first calculation
        assert result.loc[0, 'months_since_first'] == 0  # First date
        assert result.loc[1, 'months_since_first'] == 5  # 5 months later
        assert result.loc[2, 'months_since_first'] == 11  # 11 months later
        assert pd.isna(result.loc[3, 'months_since_first'])  # None


class TestDataHandler:
    """Test the main DataHandler class."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a temporary data directory with sample files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create sample train.csv
        n_samples = 50
        train_data = {
            'job_posted_date': ['2024/01'] * n_samples,
            'job_title': ['Engineer'] * n_samples,
            'feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'job_state': np.random.choice(['CA', 'NY', 'TX'], n_samples),
            'salary_category': np.random.choice(['Low', 'Medium', 'High'], n_samples)
        }

        # Add other features
        for i in range(2, 13):
            if i == 2 or i == 10:
                train_data[f'feature_{i}'] = np.random.randn(n_samples)
            else:
                train_data[f'feature_{i}'] = np.random.randint(0, 2, n_samples)

        # Add job description features
        for i in range(1, 301):
            train_data[f'job_desc_{i:03d}'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

        train_df = pd.DataFrame(train_data)
        train_df.to_csv(data_dir / "train.csv", index=False)

        # Create sample test.csv (without target)
        test_data = train_data.copy()
        del test_data['salary_category']
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(data_dir / "test.csv", index=False)

        return str(data_dir)

    def test_data_loading(self, sample_data_dir):
        """Test data loading functionality."""
        dh = DataHandler(data_dir=sample_data_dir)

        # Check data shapes
        X_train, y_train, train_idx = dh.get_train_data_raw()
        X_test, test_idx = dh.get_test_data_raw()

        assert X_train.shape[0] == 50
        assert y_train.shape[0] == 50
        assert X_test.shape[0] == 50

        # Check target encoding
        assert y_train['salary_category'].isin([0, 1, 2]).all()

    def test_pipeline_creation(self, sample_data_dir):
        """Test preprocessing pipeline creation."""
        # Without clustering
        dh1 = DataHandler(data_dir=sample_data_dir, n_job_clusters=0)
        assert len(dh1.pipeline.steps) == 4  # feature_extractor, preprocessor, imputer, scaler

        # With clustering
        dh2 = DataHandler(data_dir=sample_data_dir, n_job_clusters=3)
        assert len(dh2.pipeline.steps) == 4

    def test_preprocessing(self, sample_data_dir):
        """Test data preprocessing."""
        dh = DataHandler(data_dir=sample_data_dir, n_job_clusters=0)

        X_train, y_train, _ = dh.get_train_data_raw()
        X_processed, _ = dh.get_train_data_processed()

        # Check preprocessing results
        assert isinstance(X_processed, pd.DataFrame)
        assert X_processed.shape[0] == X_train.shape[0]
        assert X_processed.isna().sum().sum() == 0  # No missing values after imputation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])