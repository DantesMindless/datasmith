"""
Tests for Scikit-learn Model Training Functions
Covers both Classification and Regression models
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from io import BytesIO
import joblib
from sklearn.model_selection import train_test_split

from app.functions.training import train_sklearn_model, get_model_instance
from app.models.choices import ModelType, REGRESSION_MODELS, CLASSIFICATION_MODELS
from app.tests.fixtures import (
    MockDataGenerator,
    MockModelFactory,
    MockMinIOStorage,
    MockTrainingLogger
)


class TestClassificationModels(TestCase):
    """Test suite for classification model training"""

    def setUp(self):
        """Set up test fixtures"""
        # Patch MinIO and logger
        self.minio_patcher = patch('app.functions.training.upload_to_minio',
                                   side_effect=MockMinIOStorage.mock_upload_to_minio)
        self.logger_patcher = patch('app.functions.training.TrainingLogger',
                                   return_value=MockTrainingLogger())

        self.mock_upload = self.minio_patcher.start()
        self.mock_logger_class = self.logger_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_random_forest_classification(self):
        """Test Random Forest classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=3
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10, 'max_depth': 5}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify training completed successfully
        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertIn('confusion_matrix', model.analytics_data)
        self.assertIn('class_report', model.analytics_data)
        self.assertIn('accuracy', model.analytics_data)
        self.assertNotIn('is_regression', model.analytics_data)  # Classification should not have this

        # Verify model was saved
        model.save.assert_called()

    def test_logistic_regression(self):
        """Test Logistic Regression training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.LOGISTIC_REGRESSION,
            target_column=target,
            training_config={'max_iter': 100}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        # Logistic regression should have feature importance via coef_
        self.assertIn('feature_importance', model.analytics_data)

    def test_decision_tree_classifier(self):
        """Test Decision Tree classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.DECISION_TREE,
            target_column=target,
            training_config={'max_depth': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertIn('feature_importance', model.analytics_data)

    def test_knn_classifier(self):
        """Test K-Nearest Neighbors classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.KNN,
            target_column=target,
            training_config={'n_neighbors': 5}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_svm_classifier(self):
        """Test Support Vector Machine classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2  # Smaller dataset for SVM speed
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.SVM,
            target_column=target,
            training_config={'kernel': 'linear', 'max_iter': 100}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_naive_bayes(self):
        """Test Naive Bayes training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NAIVE_BAYES,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_gradient_boosting_classifier(self):
        """Test Gradient Boosting classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.GRADIENT_BOOSTING,
            target_column=target,
            training_config={'n_estimators': 10, 'learning_rate': 0.1}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertIn('feature_importance', model.analytics_data)

    def test_adaboost_classifier(self):
        """Test AdaBoost classifier training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.ADABOOST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_multiclass_classification(self):
        """Test multi-class classification with 5 classes"""
        df, features, target = MockDataGenerator.create_multiclass_data(
            n_samples=150, n_features=5, n_classes=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        # Confusion matrix should be 5x5
        conf_matrix = model.analytics_data.get('confusion_matrix')
        self.assertEqual(len(conf_matrix), 5)
        self.assertEqual(len(conf_matrix[0]), 5)


class TestRegressionModels(TestCase):
    """Test suite for regression model training"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.training.upload_to_minio',
                                   side_effect=MockMinIOStorage.mock_upload_to_minio)
        self.logger_patcher = patch('app.functions.training.TrainingLogger',
                                   return_value=MockTrainingLogger())

        self.mock_upload = self.minio_patcher.start()
        self.mock_logger_class = self.logger_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_linear_regression(self):
        """Test Linear Regression training with regression metrics"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify training completed
        self.assertIsNotNone(model_path)

        # Verify regression-specific analytics
        self.assertTrue(model.analytics_data.get('is_regression'))
        self.assertIn('r2_score', model.analytics_data)
        self.assertIn('rmse', model.analytics_data)
        self.assertIn('mae', model.analytics_data)
        self.assertIn('mse', model.analytics_data)
        self.assertIn('target_stats', model.analytics_data)

        # Verify target_stats structure
        target_stats = model.analytics_data['target_stats']
        self.assertIn('mean', target_stats)
        self.assertIn('std', target_stats)
        self.assertIn('min', target_stats)
        self.assertIn('max', target_stats)

        # No classification metrics
        self.assertNotIn('confusion_matrix', model.analytics_data)
        self.assertNotIn('class_report', model.analytics_data)

    def test_random_forest_regressor(self):
        """Test Random Forest regressor training"""
        df, features, target = MockDataGenerator.create_housing_data(n_samples=200)

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))
        self.assertIn('feature_importance', model.analytics_data)
        self.assertGreater(len(model.analytics_data['feature_importance']), 0)

    def test_decision_tree_regressor(self):
        """Test Decision Tree regressor training"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.DECISION_TREE_REGRESSOR,
            target_column=target,
            training_config={'max_depth': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))
        self.assertIn('feature_importance', model.analytics_data)

    def test_svr_regressor(self):
        """Test Support Vector Regressor training"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=50, n_features=3  # Smaller for SVR speed
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.SVR,
            target_column=target,
            training_config={'kernel': 'rbf', 'C': 1.0}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))

    def test_knn_regressor(self):
        """Test K-Nearest Neighbors regressor training"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.KNN_REGRESSOR,
            target_column=target,
            training_config={'n_neighbors': 5, 'weights': 'distance'}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))

    def test_gradient_boosting_regressor(self):
        """Test Gradient Boosting regressor training"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.GRADIENT_BOOSTING_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10, 'learning_rate': 0.1}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))
        self.assertIn('feature_importance', model.analytics_data)

    def test_adaboost_regressor(self):
        """Test AdaBoost regressor training"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.ADABOOST_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))

    def test_regression_target_stats_accuracy(self):
        """Test that target stats are calculated correctly"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5, target_range=(100, 500)
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        target_stats = model.analytics_data['target_stats']

        # Verify target stats are reasonable for our data range
        self.assertGreater(target_stats['mean'], 50)
        self.assertGreater(target_stats['std'], 0)
        self.assertLess(target_stats['min'], target_stats['max'])

    def test_regression_metrics_are_numeric(self):
        """Test that all regression metrics are proper numbers"""
        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # All metrics should be floats
        self.assertIsInstance(model.analytics_data['r2_score'], float)
        self.assertIsInstance(model.analytics_data['rmse'], float)
        self.assertIsInstance(model.analytics_data['mae'], float)
        self.assertIsInstance(model.analytics_data['mse'], float)

        # RMSE should be sqrt of MSE
        expected_rmse = np.sqrt(model.analytics_data['mse'])
        self.assertAlmostEqual(model.analytics_data['rmse'], expected_rmse, places=5)


class TestModelTrainingCommon(TestCase):
    """Common training tests for all model types"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.training.upload_to_minio',
                                   side_effect=MockMinIOStorage.mock_upload_to_minio)
        self.logger_patcher = patch('app.functions.training.TrainingLogger',
                                   return_value=MockTrainingLogger())

        self.mock_upload = self.minio_patcher.start()
        self.mock_logger_class = self.logger_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_invalid_model_type_raises_error(self):
        """Test error handling for invalid model type"""
        df, features, target = MockDataGenerator.create_tabular_classification_data()

        model = MockModelFactory.create_mock_mlmodel(
            model_type='INVALID_MODEL',
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        with self.assertRaises(Exception):
            train_sklearn_model(model, X_train, y_train, X_test, y_test)

    def test_prediction_schema_saved(self):
        """Test that prediction schema is saved"""
        df, features, target = MockDataGenerator.create_tabular_classification_data()

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify prediction schema
        self.assertIsNotNone(model.prediction_schema)
        self.assertIn('input_features', model.prediction_schema)
        self.assertIn('feature_count', model.prediction_schema)
        self.assertIn('target_column', model.prediction_schema)

    def test_model_upload_called(self):
        """Test that model is uploaded to MinIO"""
        df, features, target = MockDataGenerator.create_tabular_classification_data()

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify upload was called
        self.mock_upload.assert_called()

    def test_training_samples_tracked(self):
        """Test that training/test sample counts are tracked"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target, test_size=0.2)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertEqual(model.analytics_data['training_samples'], len(X_train))
        self.assertEqual(model.analytics_data['test_samples'], len(X_test))

    def test_random_state_reproducibility(self):
        """Test that same random_state produces same results"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100
        )

        model1 = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )
        model1.random_state = 42

        model2 = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )
        model2.random_state = 42

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        _, accuracy1 = train_sklearn_model(model1, X_train, y_train, X_test, y_test)
        _, accuracy2 = train_sklearn_model(model2, X_train, y_train, X_test, y_test)

        # Results should be identical with same random state
        self.assertEqual(accuracy1, accuracy2)

    def test_feature_importance_sorted(self):
        """Test that feature importances are sorted by importance"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        feature_importance = model.analytics_data['feature_importance']

        # Check that importances are sorted descending
        importances = [f['importance'] for f in feature_importance]
        self.assertEqual(importances, sorted(importances, reverse=True))


class TestGetModelInstance(TestCase):
    """Test the get_model_instance function"""

    def test_all_classification_models_can_be_instantiated(self):
        """Test that all classification models can be created"""
        model = MockModelFactory.create_mock_mlmodel()

        for model_type in CLASSIFICATION_MODELS:
            with self.subTest(model_type=model_type):
                model.model_type = model_type
                try:
                    clf = get_model_instance(model_type, model)
                    self.assertIsNotNone(clf)
                except ImportError:
                    # XGBoost/LightGBM may not be installed
                    pass

    def test_all_regression_models_can_be_instantiated(self):
        """Test that all regression models can be created"""
        model = MockModelFactory.create_mock_mlmodel()

        for model_type in REGRESSION_MODELS:
            with self.subTest(model_type=model_type):
                model.model_type = model_type
                try:
                    reg = get_model_instance(model_type, model)
                    self.assertIsNotNone(reg)
                except ImportError:
                    # XGBoost/LightGBM may not be installed
                    pass


class TestXGBoostLightGBM(TestCase):
    """Test XGBoost and LightGBM models (if installed)"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.training.upload_to_minio',
                                   side_effect=MockMinIOStorage.mock_upload_to_minio)
        self.logger_patcher = patch('app.functions.training.TrainingLogger',
                                   return_value=MockTrainingLogger())

        self.mock_upload = self.minio_patcher.start()
        self.mock_logger_class = self.logger_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_xgboost_classifier(self):
        """Test XGBoost classifier training"""
        try:
            import xgboost
        except ImportError:
            self.skipTest("XGBoost not installed")

        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.XGBOOST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_xgboost_regressor(self):
        """Test XGBoost regressor training"""
        try:
            import xgboost
        except ImportError:
            self.skipTest("XGBoost not installed")

        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.XGBOOST_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))

    def test_lightgbm_classifier(self):
        """Test LightGBM classifier training"""
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM not installed")

        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.LIGHTGBM,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_lightgbm_regressor(self):
        """Test LightGBM regressor training"""
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM not installed")

        df, features, target = MockDataGenerator.create_tabular_regression_data(
            n_samples=100, n_features=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.LIGHTGBM_REGRESSOR,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, r2_score = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(model.analytics_data.get('is_regression'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
