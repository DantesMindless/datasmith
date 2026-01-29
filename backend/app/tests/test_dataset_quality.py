"""
Tests for Dataset Quality Analysis Functions
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from django.test import TestCase
from datetime import datetime

from app.models.main import Dataset
from app.models.choices import DatasetType, DatasetPurpose, DataQuality


class TestDatasetQualityAnalysis(TestCase):
    """Test suite for dataset quality analysis"""

    def create_sample_dataset(self, with_issues=False):
        """Create a sample dataset for testing"""
        if with_issues:
            # Dataset with quality issues
            data = {
                'col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
                'col2': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # Low variance
                'col3': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
                'col4': [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],  # Duplicate pattern
                'email': ['test@example.com', 'invalid', 'user@test.com', '',
                         'another@example.com', np.nan, 'valid@mail.com',
                         'test2@example.com', 'bad_email', 'last@example.com']
            }
        else:
            # Clean dataset
            data = {
                'col1': np.random.randn(100),
                'col2': np.random.randint(0, 10, 100),
                'col3': np.random.choice(['A', 'B', 'C', 'D'], 100),
                'col4': np.random.randn(100) * 10 + 50,
                'target': np.random.randint(0, 2, 100)
            }

        return pd.DataFrame(data)

    def test_null_value_detection(self):
        """Test detection of null/missing values"""
        df = self.create_sample_dataset(with_issues=True)

        null_counts = df.isnull().sum()
        null_percentage = (df.isnull().sum() / len(df) * 100)

        # col1 should have 2 nulls (20%)
        self.assertEqual(null_counts['col1'], 2)
        self.assertAlmostEqual(null_percentage['col1'], 20.0, places=1)

    def test_duplicate_detection(self):
        """Test detection of duplicate rows"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 1, 2],
            'b': [4, 5, 6, 4, 5],
            'c': [7, 8, 9, 7, 8]
        })

        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()

        # Should detect 2 duplicates
        self.assertEqual(duplicate_count, 2)

    def test_data_type_detection(self):
        """Test automatic data type detection"""
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'C', 'A', 'B'],
            'text': ['hello world', 'test string', 'another text', 'more text', 'final text'],
            'datetime': pd.date_range('2024-01-01', periods=5),
            'email': ['test@example.com', 'user@test.com', 'admin@site.com', 'info@company.com', 'support@help.com']
        })

        # Test numeric detection
        self.assertTrue(pd.api.types.is_numeric_dtype(df['numeric_int']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['numeric_float']))

        # Test datetime detection
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))

        # Test categorical detection (low cardinality)
        self.assertEqual(df['categorical'].nunique(), 3)

    def test_completeness_score(self):
        """Test dataset completeness scoring"""
        # Perfect dataset (no missing values)
        perfect_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [6, 7, 8, 9, 10]
        })

        completeness_perfect = (1 - perfect_df.isnull().sum().sum() / (perfect_df.shape[0] * perfect_df.shape[1])) * 100
        self.assertEqual(completeness_perfect, 100.0)

        # Dataset with missing values
        incomplete_df = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [6, 7, np.nan, 9, np.nan]
        })

        completeness_incomplete = (1 - incomplete_df.isnull().sum().sum() / (incomplete_df.shape[0] * incomplete_df.shape[1])) * 100
        self.assertAlmostEqual(completeness_incomplete, 60.0, places=1)

    def test_statistical_summary(self):
        """Test statistical summary generation"""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        stats = df['numeric_col'].describe()

        self.assertAlmostEqual(stats['mean'], 5.5, places=1)
        self.assertAlmostEqual(stats['std'], 3.03, places=1)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 10)
        self.assertAlmostEqual(stats['50%'], 5.5, places=1)  # median

    def test_categorical_cardinality(self):
        """Test categorical column cardinality detection"""
        df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'high_cardinality': list(range(60)),
            'medium_cardinality': ['cat' + str(i % 15) for i in range(60)]
        })

        # Low cardinality (3 unique values)
        self.assertEqual(df['low_cardinality'].nunique(), 3)

        # High cardinality (60 unique values)
        self.assertEqual(df['high_cardinality'].nunique(), 60)

        # Medium cardinality (15 unique values)
        self.assertEqual(df['medium_cardinality'].nunique(), 15)

    def test_column_quality_score(self):
        """Test individual column quality scoring"""
        # Perfect column
        perfect_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        missing_pct_perfect = perfect_col.isnull().sum() / len(perfect_col) * 100
        quality_score_perfect = 100 - missing_pct_perfect

        self.assertEqual(quality_score_perfect, 100.0)

        # Column with issues
        problematic_col = pd.Series([1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan])
        missing_pct_problematic = problematic_col.isnull().sum() / len(problematic_col) * 100
        quality_score_problematic = 100 - missing_pct_problematic

        self.assertEqual(quality_score_problematic, 50.0)

    def test_outlier_detection(self):
        """Test outlier detection using IQR method"""
        # Data with outliers
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        series = pd.Series(data)

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        # Should detect the outlier (100)
        self.assertGreater(len(outliers), 0)
        self.assertIn(100, outliers.values)

    def test_variance_analysis(self):
        """Test variance analysis for feature importance"""
        df = pd.DataFrame({
            'high_variance': np.random.randn(100) * 10,
            'low_variance': np.ones(100) * 5 + np.random.randn(100) * 0.1,
            'zero_variance': np.ones(100) * 5
        })

        var_high = df['high_variance'].var()
        var_low = df['low_variance'].var()
        var_zero = df['zero_variance'].var()

        self.assertGreater(var_high, var_low)
        self.assertGreater(var_low, var_zero)
        self.assertAlmostEqual(var_zero, 0.0, places=5)

    def test_correlation_analysis(self):
        """Test correlation analysis between features"""
        # Create correlated features
        np.random.seed(42)
        x = np.random.randn(100)
        y = x * 2 + np.random.randn(100) * 0.1  # Highly correlated
        z = np.random.randn(100)  # Not correlated

        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        corr_matrix = df.corr()

        # x and y should be highly correlated
        self.assertGreater(corr_matrix.loc['x', 'y'], 0.9)

        # x and z should have low correlation
        self.assertLess(abs(corr_matrix.loc['x', 'z']), 0.3)

    def test_data_quality_classification(self):
        """Test overall data quality classification"""
        # Excellent quality
        excellent_df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randint(0, 10, 100),
            'c': np.random.choice(['A', 'B', 'C'], 100)
        })

        completeness_excellent = (1 - excellent_df.isnull().sum().sum() /
                                 (excellent_df.shape[0] * excellent_df.shape[1])) * 100

        if completeness_excellent >= 95:
            quality_excellent = DataQuality.EXCELLENT
        elif completeness_excellent >= 80:
            quality_excellent = DataQuality.GOOD
        elif completeness_excellent >= 60:
            quality_excellent = DataQuality.FAIR
        else:
            quality_excellent = DataQuality.POOR

        self.assertEqual(quality_excellent, DataQuality.EXCELLENT)

    def test_email_validation(self):
        """Test email format validation"""
        emails = pd.Series([
            'valid@example.com',
            'also.valid@test.co.uk',
            'invalid-email',
            'missing@',
            '@nodomain.com',
            'spaces in@email.com',
            'good@domain.org'
        ])

        # Simple email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        valid_emails = emails.str.match(email_pattern)

        # Should identify valid emails
        self.assertTrue(valid_emails.iloc[0])  # valid@example.com
        self.assertTrue(valid_emails.iloc[1])  # also.valid@test.co.uk
        self.assertFalse(valid_emails.iloc[2])  # invalid-email

    def test_url_detection(self):
        """Test URL pattern detection"""
        urls = pd.Series([
            'https://www.example.com',
            'http://test.org/path',
            'www.website.com',
            'not a url',
            'ftp://files.server.net'
        ])

        url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        valid_urls = urls.str.match(url_pattern)

        self.assertTrue(valid_urls.iloc[0])  # https://www.example.com
        self.assertTrue(valid_urls.iloc[1])  # http://test.org/path
        self.assertFalse(valid_urls.iloc[3])  # not a url

    def test_numeric_string_detection(self):
        """Test detection of numeric values stored as strings"""
        data = pd.Series(['123', '456.78', '999', 'not a number', '0.123'])

        numeric_mask = pd.to_numeric(data, errors='coerce').notna()

        # Should identify numeric strings
        self.assertTrue(numeric_mask.iloc[0])  # '123'
        self.assertTrue(numeric_mask.iloc[1])  # '456.78'
        self.assertFalse(numeric_mask.iloc[3])  # 'not a number'

    def test_date_format_detection(self):
        """Test detection of date strings"""
        dates = pd.Series([
            '2024-01-15',
            '01/15/2024',
            '15-Jan-2024',
            'not a date',
            '2024-12-31'
        ])

        # Try to parse as dates
        parsed_dates = pd.to_datetime(dates, errors='coerce')
        valid_dates = parsed_dates.notna()

        self.assertTrue(valid_dates.iloc[0])  # 2024-01-15
        self.assertFalse(valid_dates.iloc[3])  # not a date

    def test_imbalanced_target_detection(self):
        """Test detection of imbalanced target distributions"""
        # Balanced
        balanced_target = pd.Series([0] * 50 + [1] * 50)
        balance_ratio_balanced = balanced_target.value_counts(normalize=True)

        self.assertAlmostEqual(balance_ratio_balanced[0], 0.5, places=2)
        self.assertAlmostEqual(balance_ratio_balanced[1], 0.5, places=2)

        # Imbalanced
        imbalanced_target = pd.Series([0] * 90 + [1] * 10)
        balance_ratio_imbalanced = imbalanced_target.value_counts(normalize=True)

        self.assertAlmostEqual(balance_ratio_imbalanced[0], 0.9, places=2)
        self.assertAlmostEqual(balance_ratio_imbalanced[1], 0.1, places=2)

        # Detect imbalance (ratio > 2:1)
        max_ratio = balance_ratio_imbalanced.max() / balance_ratio_imbalanced.min()
        is_imbalanced = max_ratio > 2

        self.assertTrue(is_imbalanced)

    def test_constant_column_detection(self):
        """Test detection of constant (zero variance) columns"""
        df = pd.DataFrame({
            'constant': [5] * 100,
            'variable': np.random.randn(100)
        })

        # Constant column should have zero variance
        self.assertAlmostEqual(df['constant'].var(), 0.0, places=10)

        # Variable column should have non-zero variance
        self.assertGreater(df['variable'].var(), 0.0)

    def test_memory_usage_analysis(self):
        """Test dataset memory usage analysis"""
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'string_col': ['text'] * 1000
        })

        memory_usage = df.memory_usage(deep=True)

        # Should report memory for each column
        self.assertGreater(memory_usage['int_col'], 0)
        self.assertGreater(memory_usage['float_col'], 0)
        self.assertGreater(memory_usage['string_col'], 0)

    def test_skewness_detection(self):
        """Test detection of skewed distributions"""
        # Right-skewed data
        right_skewed = pd.Series(np.random.exponential(2, 1000))
        skewness_right = right_skewed.skew()

        self.assertGreater(skewness_right, 0)

        # Left-skewed data
        left_skewed = pd.Series(-np.random.exponential(2, 1000))
        skewness_left = left_skewed.skew()

        self.assertLess(skewness_left, 0)

        # Normal distribution (low skew)
        normal = pd.Series(np.random.randn(1000))
        skewness_normal = normal.skew()

        self.assertLess(abs(skewness_normal), 0.5)


class TestDatasetPurposeDetection(TestCase):
    """Test automatic dataset purpose detection"""

    def test_classification_detection(self):
        """Test detection of classification datasets"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['Class A', 'Class B', 'Class C'], 100)
        })

        # Target is categorical with low cardinality -> Classification
        target_unique = df['target'].nunique()
        is_classification = target_unique < 20 and df['target'].dtype == 'object'

        self.assertTrue(is_classification)

    def test_regression_detection(self):
        """Test detection of regression datasets"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100) * 100 + 500
        })

        # Target is continuous numeric -> Regression
        is_numeric_target = pd.api.types.is_numeric_dtype(df['target'])
        target_unique_ratio = df['target'].nunique() / len(df)

        # If most values are unique, likely regression
        is_regression = is_numeric_target and target_unique_ratio > 0.5

        self.assertTrue(is_regression)

    def test_clustering_detection(self):
        """Test detection of clustering datasets (no target)"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })

        # No obvious target column -> Clustering
        # (this is a simplified check)
        has_target_like_column = any(col.lower() in ['target', 'label', 'class']
                                     for col in df.columns)

        self.assertFalse(has_target_like_column)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
