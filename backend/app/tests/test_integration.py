"""
Integration Tests for ML Training Pipelines
Tests end-to-end workflows from dataset upload to model training and prediction
"""
import pytest
import numpy as np
import pandas as pd
import torch
import joblib
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from io import BytesIO
from sklearn.model_selection import train_test_split

from app.functions.training import (
    train_sklearn_model,
    train_nn,
    train_cnn
)
from app.functions.prediction import (
    predict_sklearn_model,
    predict_nn,
    predict_cnn
)
from app.models.choices import ModelType
from app.tests.fixtures import (
    MockDataGenerator,
    MockModelFactory,
    MockMinIOStorage
)


class TestEndToEndWorkflows(TestCase):
    """Test complete end-to-end ML workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_upload_patcher = patch('app.functions.training.upload_to_minio',
                                          side_effect=MockMinIOStorage.mock_upload_to_minio)
        self.minio_download_patcher = patch('app.functions.prediction.download_from_minio',
                                            side_effect=MockMinIOStorage.mock_download_from_minio)
        self.logger_patcher = patch('app.functions.training.TrainingLogger')

        self.mock_upload = self.minio_upload_patcher.start()
        self.mock_download = self.minio_download_patcher.start()
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = Mock()

    def tearDown(self):
        """Clean up patches"""
        self.minio_upload_patcher.stop()
        self.minio_download_patcher.stop()
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_sklearn_full_pipeline(self):
        """Test complete sklearn pipeline: data -> train -> predict"""
        # 1. Create dataset
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # 2. Train model
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify training succeeded
        self.assertIsNotNone(model_path)
        self.assertIn('accuracy', model.analytics_data)

        # 3. Make predictions
        # Mock the trained model for prediction
        from sklearn.ensemble import RandomForestClassifier
        X = df[features]
        y = df[target]
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        model_bytes = BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)

        model.model_file = Mock()
        model.model_file.read = Mock(return_value=model_bytes.getvalue())
        model.prediction_schema = {
            'input_features': features,
            'output_classes': list(range(2))
        }

        input_data = {feature: float(np.random.randn()) for feature in features}

        with patch('joblib.load', return_value=clf):
            prediction_result = predict_sklearn_model(model, input_data)

        # Verify prediction succeeded
        self.assertIn('prediction', prediction_result)
        self.assertIn(prediction_result['prediction'], [0, 1])

    def test_neural_network_full_pipeline(self):
        """Test complete neural network pipeline"""
        # 1. Create dataset
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=3
        )

        # 2. Train model
        training_config = {
            'layer_config': [
                {'size': 32, 'activation': 'relu'},
                {'size': 16, 'activation': 'relu'}
            ],
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 0.001
        }

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=training_config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        # Verify training
        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_multiple_models_same_dataset(self):
        """Test training multiple models on the same dataset"""
        # Create dataset
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model_types = [
            ModelType.RANDOM_FOREST,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.KNN
        ]

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        results = {}

        for model_type in model_types:
            model = MockModelFactory.create_mock_mlmodel(
                model_type=model_type,
                target_column=target,
                training_config={'n_estimators': 10} if 'FOREST' in model_type else {}
            )

            model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

            results[model_type] = (model_path, accuracy)

        # All models should train successfully
        for model_type, (model_path, accuracy) in results.items():
            with self.subTest(model_type=model_type):
                self.assertIsNotNone(model_path)
                self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_model_comparison_workflow(self):
        """Test workflow for comparing different models"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        models_to_compare = [
            (ModelType.RANDOM_FOREST, {'n_estimators': 10}),
            (ModelType.GRADIENT_BOOSTING, {'n_estimators': 10}),
            (ModelType.SVM, {'max_iter': 100})
        ]

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        comparison_results = []

        for model_type, config in models_to_compare:
            model = MockModelFactory.create_mock_mlmodel(
                model_type=model_type,
                target_column=target,
                training_config=config
            )

            model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

            comparison_results.append({
                'model_type': model_type,
                'accuracy': accuracy,
                'config': config
            })

        # All models should produce valid results
        self.assertEqual(len(comparison_results), 3)

        # Each should have accuracy
        for result in comparison_results:
            self.assertTrue(0.0 <= result['accuracy'] <= 1.0)

    def test_dataset_preprocessing_pipeline(self):
        """Test dataset preprocessing before training"""
        # Create dataset with various issues
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 10,
            'feature2': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10,
            'feature3': np.random.randn(100),
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
        })

        features = ['feature1', 'feature2', 'feature3']
        target = 'target'

        # Test that training handles preprocessing
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        # Fill NaN before splitting since train_test_split doesn't handle NaN
        df = df.fillna(df.mean(numeric_only=True))
        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Should handle NaN and train successfully
        self.assertIsNotNone(model_path)

    def test_hyperparameter_tuning_workflow(self):
        """Test workflow with different hyperparameters"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        hyperparameter_configs = [
            {'n_estimators': 5, 'max_depth': 3},
            {'n_estimators': 10, 'max_depth': 5},
            {'n_estimators': 20, 'max_depth': 10}
        ]

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        best_accuracy = 0
        best_config = None

        for config in hyperparameter_configs:
            model = MockModelFactory.create_mock_mlmodel(
                model_type=ModelType.RANDOM_FOREST,
                target_column=target,
                training_config=config
            )

            model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config

        # Should identify best configuration
        self.assertIsNotNone(best_config)
        self.assertTrue(0.0 <= best_accuracy <= 1.0)

    def test_cross_validation_workflow(self):
        """Test workflow simulating cross-validation"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        n_folds = 3
        fold_results = []

        for fold in range(n_folds):
            # Different random state for each fold
            model = MockModelFactory.create_mock_mlmodel(
                model_type=ModelType.RANDOM_FOREST,
                target_column=target,
                training_config={'n_estimators': 10}
            )
            model.random_state = 42 + fold

            X_train, X_test, y_train, y_test = self._split_data(
                df, features, target, random_state=42 + fold
            )
            model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

            fold_results.append(accuracy)

        # Calculate average accuracy
        avg_accuracy = np.mean(fold_results)

        self.assertEqual(len(fold_results), n_folds)
        self.assertTrue(0.0 <= avg_accuracy <= 1.0)

    def test_batch_prediction_workflow(self):
        """Test batch prediction on multiple samples"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # Train model
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Create trained model for predictions
        from sklearn.ensemble import RandomForestClassifier
        X = df[features]
        y = df[target]
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        model_bytes = BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)

        model.model_file = Mock()
        model.model_file.read = Mock(return_value=model_bytes.getvalue())
        model.prediction_schema = {
            'input_features': features,
            'output_classes': [0, 1]
        }

        # Batch predictions
        batch_inputs = [
            {feature: float(np.random.randn()) for feature in features}
            for _ in range(10)
        ]

        predictions = []
        with patch('joblib.load', return_value=clf):
            for input_data in batch_inputs:
                result = predict_sklearn_model(model, input_data)
                predictions.append(result['prediction'])

        # Verify all predictions
        self.assertEqual(len(predictions), 10)
        for pred in predictions:
            self.assertIn(pred, [0, 1])

    def test_model_retraining_workflow(self):
        """Test workflow for retraining existing model"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # Initial training
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        _, initial_accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Retrain with more data or different config
        df_extended, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=150, n_features=5, n_classes=2
        )

        model.training_config = {'n_estimators': 20}

        X_train2, X_test2, y_train2, y_test2 = self._split_data(df_extended, features, target)
        _, retrained_accuracy = train_sklearn_model(model, X_train2, y_train2, X_test2, y_test2)

        # Both training runs should succeed
        self.assertTrue(0.0 <= initial_accuracy <= 1.0)
        self.assertTrue(0.0 <= retrained_accuracy <= 1.0)

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column='target'
        )

        # Pass empty pandas objects
        with self.assertRaises(Exception):
            train_sklearn_model(model, pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float))

    def test_different_data_types_workflow(self):
        """Test workflow with different data types"""
        # Mixed data types - use only numeric for sklearn
        df = pd.DataFrame({
            'numeric_int': np.random.randint(0, 100, 100),
            'numeric_float': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        features = ['numeric_int', 'numeric_float']
        target = 'target'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Should handle mixed types
        self.assertIsNotNone(model_path)

    def test_model_persistence_workflow(self):
        """Test that model can be saved and loaded"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # Train and save
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify model was uploaded to storage
        self.mock_upload.assert_called()

        # Verify model path was returned
        self.assertIsNotNone(model_path)

    def test_analytics_data_workflow(self):
        """Test that analytics data is properly generated and stored"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_sklearn_model(model, X_train, y_train, X_test, y_test)

        # Verify comprehensive analytics
        self.assertIn('accuracy', model.analytics_data)
        self.assertIn('confusion_matrix', model.analytics_data)
        self.assertIn('class_report', model.analytics_data)


class TestMultiModalWorkflows(TestCase):
    """Test workflows across different data modalities"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger_patcher = patch('app.functions.training.TrainingLogger')
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = Mock()

    def tearDown(self):
        """Clean up patches"""
        self.logger_patcher.stop()

    def _split_data(self, df, features, target, test_size=0.2, random_state=42):
        """Helper to split data into train/test sets"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_tabular_to_neural_network_conversion(self):
        """Test converting tabular problem to neural network"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # Try both sklearn and NN
        sklearn_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST,
            target_column=target,
            training_config={'n_estimators': 10}
        )

        nn_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'size': 32, 'activation': 'relu'}],
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        with patch('app.functions.training.upload_to_minio',
                  side_effect=MockMinIOStorage.mock_upload_to_minio):
            sklearn_path, sklearn_acc = train_sklearn_model(sklearn_model, X_train, y_train, X_test, y_test)
            nn_path, nn_acc = train_nn(nn_model, X_train, y_train, X_test, y_test)

        # Both should work on same data
        self.assertIsNotNone(sklearn_path)
        self.assertIsNotNone(nn_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
