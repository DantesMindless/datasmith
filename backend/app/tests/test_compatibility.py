"""
Tests for Model-Dataset Compatibility Validation System
"""
import pytest
from django.test import TestCase
from app.validators.model_compatibility import ModelCompatibilityValidator
from app.models.main import Dataset
from app.models.choices import ModelType, DatasetType, DatasetPurpose, DataQuality
from unittest.mock import Mock


class TestModelCompatibilityValidator(TestCase):
    """Test suite for ModelCompatibilityValidator"""

    def create_mock_dataset(
        self,
        dataset_type=DatasetType.TABULAR,
        dataset_purpose=DatasetPurpose.CLASSIFICATION,
        row_count=1000,
        column_count=10,
        data_quality=DataQuality.GOOD,
        column_info=None
    ):
        """Helper to create mock dataset"""
        dataset = Mock()
        dataset.dataset_type = dataset_type
        dataset.dataset_purpose = dataset_purpose
        dataset.row_count = row_count
        dataset.column_count = column_count
        dataset.data_quality = data_quality
        dataset.column_info = column_info or {}
        return dataset

    def test_compatible_tabular_model(self):
        """Test that Random Forest is compatible with tabular data"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            row_count=500,
            column_count=15
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.RANDOM_FOREST,
            dataset
        )

        self.assertTrue(result['is_compatible'])
        self.assertEqual(len(result['errors']), 0)
        self.assertGreaterEqual(result['score'], 80)

    def test_incompatible_dataset_type(self):
        """Test that Random Forest is incompatible with image data"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.IMAGE,
            row_count=100
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.RANDOM_FOREST,
            dataset
        )

        self.assertFalse(result['is_compatible'])
        self.assertGreater(len(result['errors']), 0)
        self.assertLess(result['score'], 50)

    def test_cnn_with_image_dataset(self):
        """Test that CNN is compatible with image datasets"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.IMAGE,
            dataset_purpose=DatasetPurpose.CLASSIFICATION,
            row_count=200
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.CNN,
            dataset
        )

        self.assertTrue(result['is_compatible'])
        self.assertEqual(len(result['errors']), 0)

    def test_minimum_samples_warning(self):
        """Test warning for insufficient samples"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            row_count=50,  # Below Random Forest's recommended 100
            column_count=10
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.RANDOM_FOREST,
            dataset
        )

        self.assertTrue(result['is_compatible'])
        self.assertGreater(len(result['warnings']), 0)
        self.assertLess(result['score'], 100)

    def test_too_many_features_warning(self):
        """Test warning for too many features"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            row_count=500,
            column_count=300  # Above SVM's recommended max of 200
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.SVM,
            dataset
        )

        # Should still be compatible but with warnings
        self.assertTrue(result['is_compatible'])
        self.assertGreater(len(result['warnings']), 0)

    def test_rnn_with_time_series(self):
        """Test that LSTM is compatible with time series data"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TIME_SERIES,
            dataset_purpose=DatasetPurpose.REGRESSION,
            row_count=500,
            column_count=5
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.LSTM,
            dataset
        )

        self.assertTrue(result['is_compatible'])
        self.assertEqual(len(result['errors']), 0)

    def test_get_recommended_models(self):
        """Test model recommendation system"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            dataset_purpose=DatasetPurpose.CLASSIFICATION,
            row_count=1000,
            column_count=20
        )

        recommendations = ModelCompatibilityValidator.get_recommended_models(
            dataset,
            top_n=5
        )

        # Should return 5 recommendations
        self.assertEqual(len(recommendations), 5)

        # Each recommendation should have model_type, score, and validation
        for model_type, score, validation in recommendations:
            self.assertIsInstance(model_type, str)
            self.assertIsInstance(score, (int, float))
            self.assertIsInstance(validation, dict)
            self.assertIn('is_compatible', validation)
            self.assertIn('score', validation)

        # Recommendations should be sorted by score (descending)
        scores = [score for _, score, _ in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_transfer_learning_with_few_samples(self):
        """Test that transfer learning models work with fewer samples"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.IMAGE,
            dataset_purpose=DatasetPurpose.CLASSIFICATION,
            row_count=60  # Transfer learning requires only 50+
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.RESNET,
            dataset
        )

        self.assertTrue(result['is_compatible'])
        # Should have fewer warnings than CNN due to transfer learning
        cnn_result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.CNN,
            dataset
        )
        self.assertLessEqual(len(result['warnings']), len(cnn_result['warnings']))

    def test_ensemble_model_compatibility(self):
        """Test ensemble models (XGBoost, LightGBM) compatibility"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            dataset_purpose=DatasetPurpose.CLASSIFICATION,
            row_count=500,
            column_count=50
        )

        # Test XGBoost
        xgb_result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.XGBOOST,
            dataset
        )
        self.assertTrue(xgb_result['is_compatible'])

        # Test LightGBM
        lgb_result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.LIGHTGBM,
            dataset
        )
        self.assertTrue(lgb_result['is_compatible'])

    def test_get_compatible_models_for_dataset_type(self):
        """Test getting all compatible models for a dataset type"""
        tabular_models = ModelCompatibilityValidator.get_compatible_models_for_dataset_type(
            DatasetType.TABULAR
        )
        image_models = ModelCompatibilityValidator.get_compatible_models_for_dataset_type(
            DatasetType.IMAGE
        )

        # Tabular should have more models
        self.assertGreater(len(tabular_models), 0)
        self.assertGreater(len(image_models), 0)

        # Check specific models are in correct categories
        self.assertIn(ModelType.RANDOM_FOREST, tabular_models)
        self.assertIn(ModelType.CNN, image_models)
        self.assertNotIn(ModelType.CNN, tabular_models)

    def test_unknown_model_type(self):
        """Test handling of unknown model type"""
        dataset = self.create_mock_dataset()

        result = ModelCompatibilityValidator.validate_compatibility(
            'unknown_model',
            dataset
        )

        self.assertFalse(result['is_compatible'])
        self.assertGreater(len(result['errors']), 0)
        self.assertEqual(result['score'], 0)

    def test_purpose_mismatch(self):
        """Test incompatibility due to dataset purpose mismatch"""
        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            dataset_purpose=DatasetPurpose.REGRESSION,  # Logistic Regression is for classification
            row_count=500,
            column_count=10
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.LOGISTIC_REGRESSION,
            dataset
        )

        self.assertFalse(result['is_compatible'])
        self.assertGreater(len(result['errors']), 0)

    def test_numeric_requirement(self):
        """Test warning for non-numeric features when model requires numeric"""
        column_info = {
            'feature1': {'type': 'categorical'},
            'feature2': {'type': 'text'},
            'feature3': {'type': 'numeric'},
        }

        dataset = self.create_mock_dataset(
            dataset_type=DatasetType.TABULAR,
            row_count=500,
            column_count=3,
            column_info=column_info
        )

        result = ModelCompatibilityValidator.validate_compatibility(
            ModelType.SVM,  # SVM requires numeric features
            dataset
        )

        # Should be compatible but with warnings about encoding
        self.assertTrue(result['is_compatible'])
        self.assertGreater(len(result['warnings']), 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
