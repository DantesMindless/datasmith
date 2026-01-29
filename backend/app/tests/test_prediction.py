"""
Tests for Prediction Functions
Covers sklearn, neural network, and CNN predictions
"""
import pytest
import torch
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from io import BytesIO
from PIL import Image

from app.functions.prediction import (
    predict_sklearn_model,
    predict_nn,
    predict_cnn,
    preprocess_healthcare_data
)
from app.models.choices import ModelType
from app.tests.fixtures import (
    MockDataGenerator,
    MockModelFactory,
    MockMinIOStorage
)


class TestPreprocessHealthcareData(TestCase):
    """Test the preprocessing function for healthcare data"""

    def test_gender_conversion(self):
        """Test gender M/F to 1/0 conversion"""
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F'],
            'age': [30, 25, 40, 35]
        })

        result = preprocess_healthcare_data(df)

        self.assertEqual(list(result['gender']), [1, 0, 1, 0])

    def test_smoking_status_conversion(self):
        """Test smoking_status conversion"""
        df = pd.DataFrame({
            'smoking_status': ['current', 'former', 'never', 'current'],
            'age': [30, 25, 40, 35]
        })

        result = preprocess_healthcare_data(df)

        self.assertEqual(list(result['smoking_status']), [2, 1, 0, 2])

    def test_family_history_conversion(self):
        """Test family_history boolean conversion"""
        df = pd.DataFrame({
            'family_history': ['True', 'False', 'true', 'false'],
            'age': [30, 25, 40, 35]
        })

        result = preprocess_healthcare_data(df)

        self.assertEqual(list(result['family_history']), [1, 0, 1, 0])

    def test_numeric_columns_preserved(self):
        """Test that numeric columns are preserved"""
        df = pd.DataFrame({
            'age': [30, 25, 40, 35],
            'bmi': [22.5, 25.0, 28.3, 23.1]
        })

        result = preprocess_healthcare_data(df)

        self.assertEqual(list(result['age']), [30, 25, 40, 35])
        np.testing.assert_array_almost_equal(result['bmi'].values, [22.5, 25.0, 28.3, 23.1])


class TestSklearnPrediction(TestCase):
    """Test suite for scikit-learn model predictions"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.prediction.download_from_minio')
        self.mock_download = self.minio_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()

    def _create_trained_model(self, model_class, n_features=5, n_classes=2, is_regression=False):
        """Create and train a sklearn model"""
        np.random.seed(42)
        X = np.random.randn(100, n_features)

        if is_regression:
            y = np.random.randn(100) * 100 + 500
        else:
            y = np.random.randint(0, n_classes, 100)

        clf = model_class()
        clf.fit(X, y)

        # Serialize model
        model_bytes = BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)

        return model_bytes.getvalue(), clf

    def test_random_forest_classification_prediction(self):
        """Test Random Forest classification predictions"""
        from sklearn.ensemble import RandomForestClassifier

        model_bytes, clf = self._create_trained_model(
            lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        )

        # Set up mock to return our trained model
        self.mock_download.return_value = model_bytes

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        # Create input data
        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(3)
            for i in range(5)
        })

        result = predict_sklearn_model(mock_model, input_df)

        # Verify result is a DataFrame with predictions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('prediction', result.columns)
        self.assertEqual(len(result), 3)

    def test_prediction_with_dict_input(self):
        """Test prediction with dictionary input"""
        from sklearn.ensemble import RandomForestClassifier

        model_bytes, clf = self._create_trained_model(
            lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        )

        self.mock_download.return_value = model_bytes

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        # Single sample as dict
        input_dict = {f'feature_{i}': np.random.randn() for i in range(5)}

        result = predict_sklearn_model(mock_model, input_dict)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('prediction', result.columns)
        self.assertEqual(len(result), 1)

    def test_regression_prediction(self):
        """Test regression model prediction"""
        from sklearn.ensemble import RandomForestRegressor

        model_bytes, reg = self._create_trained_model(
            lambda: RandomForestRegressor(n_estimators=10, random_state=42),
            is_regression=True
        )

        self.mock_download.return_value = model_bytes

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST_REGRESSOR
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(5)
            for i in range(5)
        })

        result = predict_sklearn_model(mock_model, input_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('prediction', result.columns)
        # Regression predictions should be continuous values
        self.assertTrue(all(isinstance(p, (int, float, np.floating)) for p in result['prediction']))

    def test_prediction_handles_nan_values(self):
        """Test that NaN values are handled properly"""
        from sklearn.ensemble import RandomForestClassifier

        model_bytes, clf = self._create_trained_model(
            lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        )

        self.mock_download.return_value = model_bytes

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        # Input with NaN values
        input_df = pd.DataFrame({
            'feature_0': [1.0, np.nan, 3.0],
            'feature_1': [2.0, 2.0, np.nan],
            'feature_2': [3.0, 3.0, 3.0],
            'feature_3': [4.0, 4.0, 4.0],
            'feature_4': [5.0, 5.0, 5.0]
        })

        # Should not raise error - NaN values should be filled
        result = predict_sklearn_model(mock_model, input_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('prediction', result.columns)
        self.assertEqual(len(result), 3)

    def test_missing_features_raises_error(self):
        """Test that missing required features raises error"""
        from sklearn.ensemble import RandomForestClassifier

        model_bytes, clf = self._create_trained_model(
            lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        )

        self.mock_download.return_value = model_bytes

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }

        # Missing feature_4
        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(3)
            for i in range(4)  # Only 4 features instead of 5
        })

        with self.assertRaises(ValueError) as context:
            predict_sklearn_model(mock_model, input_df)

        self.assertIn('Missing features', str(context.exception))

    def test_scaler_applied_when_present(self):
        """Test that scaler is applied when model requires it"""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        # Train with scaled data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVC()
        clf.fit(X_scaled, y)

        # Serialize model and scaler
        model_bytes = BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)

        scaler_bytes = BytesIO()
        joblib.dump(scaler, scaler_bytes)
        scaler_bytes.seek(0)

        # Mock download to return different files
        def mock_download_side_effect(path):
            if 'scaler' in path:
                return scaler_bytes.getvalue()
            return model_bytes.getvalue()

        self.mock_download.side_effect = mock_download_side_effect

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.SVM
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.prediction_schema = {
            'requires_scaling': True
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(3)
            for i in range(5)
        })

        # This should work without error
        result = predict_sklearn_model(mock_model, input_df)
        self.assertIn('prediction', result.columns)


class TestNeuralNetworkPrediction(TestCase):
    """Test suite for neural network predictions"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.prediction.download_from_minio')
        self.mock_download = self.minio_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()

    def test_nn_classification_prediction(self):
        """Test neural network classification predictions"""
        from app.functions.training import ConfigurableMLP
        from sklearn.preprocessing import LabelEncoder

        # Create a simple model
        input_dim = 5
        output_dim = 3
        layer_config = [{'units': 16, 'activation': 'relu'}]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        # Serialize model
        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        # Create label encoder
        le = LabelEncoder()
        le.fit(['class_a', 'class_b', 'class_c'])
        encoder_bytes = BytesIO()
        joblib.dump(le, encoder_bytes)
        encoder_bytes.seek(0)

        # Mock download to return different files
        def mock_download_side_effect(path):
            if 'encoder' in path:
                return encoder_bytes.getvalue()
            return model_bytes.getvalue()

        self.mock_download.side_effect = mock_download_side_effect

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)],
            'layer_config': layer_config,
            'output_dim': output_dim
        }

        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(3)
            for i in range(5)
        })

        result = predict_nn(mock_model, input_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('prediction', result.columns)
        self.assertEqual(len(result), 3)
        # Predictions should be class names
        self.assertTrue(all(p in ['class_a', 'class_b', 'class_c'] for p in result['prediction']))

    def test_nn_handles_nan_values(self):
        """Test that NN prediction handles NaN values gracefully"""
        from app.functions.training import ConfigurableMLP
        from sklearn.preprocessing import LabelEncoder

        input_dim = 3
        output_dim = 2
        layer_config = [{'units': 8, 'activation': 'relu'}]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        le = LabelEncoder()
        le.fit(['yes', 'no'])
        encoder_bytes = BytesIO()
        joblib.dump(le, encoder_bytes)
        encoder_bytes.seek(0)

        def mock_download_side_effect(path):
            if 'encoder' in path:
                return encoder_bytes.getvalue()
            return model_bytes.getvalue()

        self.mock_download.side_effect = mock_download_side_effect

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK
        )
        mock_model.training_config = {
            'features': ['feature_0', 'feature_1', 'feature_2'],
            'layer_config': layer_config,
            'output_dim': output_dim
        }

        # Input with NaN values
        input_df = pd.DataFrame({
            'feature_0': [1.5, np.nan, 3.0],
            'feature_1': [2.0, 3.0, np.nan],
            'feature_2': [3.0, 4.0, 5.0]
        })

        # Should handle NaN gracefully (filled with 0 or mean)
        result = predict_nn(mock_model, input_df)
        self.assertIn('prediction', result.columns)
        self.assertEqual(len(result), 3)


class TestCNNPrediction(TestCase):
    """Test suite for CNN predictions"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.prediction.download_from_minio')
        self.mock_download = self.minio_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()

    def test_cnn_image_prediction(self):
        """Test CNN image classification predictions"""
        from app.functions.training import ConfigurableCNN

        # Create CNN model
        input_channels = 3
        conv_layers = [{'out_channels': 16, 'kernel_size': 3}]
        fc_layers = [{'units': 32}]
        num_classes = 3
        input_size = 64

        model = ConfigurableCNN(input_channels, conv_layers, fc_layers, num_classes, input_size)

        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.CNN
        )
        mock_model.training_config = {
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'input_size': input_size,
            'num_classes': num_classes,
            'class_names': ['cat', 'dog', 'bird']
        }

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(f, format='JPEG')
            temp_path = f.name

        try:
            result = predict_cnn(mock_model, temp_path)

            # Result should be a class name
            self.assertIn(result, ['cat', 'dog', 'bird'])
        finally:
            os.unlink(temp_path)

    def test_cnn_binary_classification(self):
        """Test CNN binary classification"""
        from app.functions.training import ConfigurableCNN

        input_channels = 3
        conv_layers = [{'out_channels': 8}]
        fc_layers = [{'units': 16}]
        num_classes = 2
        input_size = 32

        model = ConfigurableCNN(input_channels, conv_layers, fc_layers, num_classes, input_size)

        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.CNN
        )
        mock_model.training_config = {
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'input_size': input_size,
            'num_classes': num_classes,
            'class_names': ['negative', 'positive']
        }

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(f, format='PNG')
            temp_path = f.name

        try:
            result = predict_cnn(mock_model, temp_path)
            self.assertIn(result, ['negative', 'positive'])
        finally:
            os.unlink(temp_path)

    def test_cnn_resizes_image(self):
        """Test that CNN properly resizes images of different sizes"""
        from app.functions.training import ConfigurableCNN

        input_channels = 3
        conv_layers = [{'out_channels': 8}]
        fc_layers = [{'units': 16}]
        num_classes = 2
        input_size = 64  # Expected size

        model = ConfigurableCNN(input_channels, conv_layers, fc_layers, num_classes, input_size)

        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.CNN
        )
        mock_model.training_config = {
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'input_size': input_size,
            'num_classes': num_classes,
            'class_names': ['a', 'b']
        }

        # Create a larger image (should be resized)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(f, format='JPEG')
            temp_path = f.name

        try:
            # Should work without error - image gets resized
            result = predict_cnn(mock_model, temp_path)
            self.assertIn(result, ['a', 'b'])
        finally:
            os.unlink(temp_path)

    def test_cnn_returns_index_when_no_class_names(self):
        """Test that CNN returns index when class_names not provided"""
        from app.functions.training import ConfigurableCNN

        input_channels = 3
        conv_layers = [{'out_channels': 8}]
        fc_layers = [{'units': 16}]
        num_classes = 3
        input_size = 32

        model = ConfigurableCNN(input_channels, conv_layers, fc_layers, num_classes, input_size)

        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.CNN
        )
        mock_model.training_config = {
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'input_size': input_size,
            'num_classes': num_classes,
            'class_names': []  # Empty class names
        }

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(f, format='JPEG')
            temp_path = f.name

        try:
            result = predict_cnn(mock_model, temp_path)
            # Should return index (0, 1, or 2)
            self.assertIn(result, [0, 1, 2])
        finally:
            os.unlink(temp_path)


class TestPredictionErrorHandling(TestCase):
    """Test error handling in prediction functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.prediction.download_from_minio')
        self.mock_download = self.minio_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()

    def test_model_file_not_found(self):
        """Test error when model file doesn't exist"""
        self.mock_download.side_effect = Exception("File not found")

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': ['feature_0', 'feature_1']
        }

        input_df = pd.DataFrame({
            'feature_0': [1.0, 2.0],
            'feature_1': [3.0, 4.0]
        })

        with self.assertRaises(Exception):
            predict_sklearn_model(mock_model, input_df)

    def test_cnn_invalid_image_path(self):
        """Test CNN prediction with invalid image path"""
        from app.functions.training import ConfigurableCNN

        model = ConfigurableCNN(3, [{'out_channels': 8}], [{'units': 16}], 2, 32)
        model_bytes = BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.CNN
        )
        mock_model.training_config = {
            'conv_layers': [{'out_channels': 8}],
            'fc_layers': [{'units': 16}],
            'input_size': 32,
            'num_classes': 2,
            'class_names': ['a', 'b']
        }

        with self.assertRaises(Exception):
            predict_cnn(mock_model, '/nonexistent/path/image.jpg')


class TestBatchPrediction(TestCase):
    """Test batch prediction capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.minio_patcher = patch('app.functions.prediction.download_from_minio')
        self.mock_download = self.minio_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.minio_patcher.stop()

    def test_large_batch_prediction(self):
        """Test prediction with large batch"""
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        model_bytes = BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)

        self.mock_download.return_value = model_bytes.getvalue()

        mock_model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.RANDOM_FOREST
        )
        mock_model.training_config = {
            'features': [f'feature_{i}' for i in range(5)]
        }
        mock_model.model_file = None  # Force use of download_from_minio path

        # Large batch (1000 samples)
        input_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000)
            for i in range(5)
        })

        result = predict_sklearn_model(mock_model, input_df)

        self.assertEqual(len(result), 1000)
        self.assertIn('prediction', result.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
