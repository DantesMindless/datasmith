"""
Tests for Neural Network (PyTorch MLP) Training Functions
"""
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from io import BytesIO
from sklearn.model_selection import train_test_split

from app.functions.training import train_nn, ConfigurableMLP, get_device
from app.models.choices import ModelType, ActivationFunction
from app.tests.fixtures import (
    MockDataGenerator,
    MockModelFactory,
    MockMinIOStorage,
    MockTrainingLogger
)


class TestConfigurableMLP(TestCase):
    """Test the ConfigurableMLP neural network architecture"""

    def test_mlp_basic_architecture(self):
        """Test basic MLP architecture creation"""
        input_dim = 10
        output_dim = 3
        layer_config = [
            {'units': 32, 'activation': ActivationFunction.RELU},
            {'units': 16, 'activation': ActivationFunction.RELU}
        ]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        # Test forward pass
        x = torch.randn(5, input_dim)
        output = model(x)

        self.assertEqual(output.shape, (5, output_dim))

    def test_mlp_single_layer(self):
        """Test MLP with single hidden layer"""
        input_dim = 5
        output_dim = 2
        layer_config = [{'units': 8, 'activation': ActivationFunction.RELU}]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        x = torch.randn(3, input_dim)
        output = model(x)

        self.assertEqual(output.shape, (3, output_dim))

    def test_mlp_deep_architecture(self):
        """Test MLP with multiple hidden layers"""
        input_dim = 20
        output_dim = 5
        layer_config = [
            {'units': 64, 'activation': ActivationFunction.RELU},
            {'units': 32, 'activation': ActivationFunction.TANH},
            {'units': 16, 'activation': ActivationFunction.SIGMOID},
            {'units': 8, 'activation': ActivationFunction.LEAKY_RELU}
        ]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        x = torch.randn(10, input_dim)
        output = model(x)

        self.assertEqual(output.shape, (10, output_dim))

    def test_mlp_different_activations(self):
        """Test MLP with different activation functions"""
        input_dim = 5
        output_dim = 2

        activations = [
            ActivationFunction.RELU,
            ActivationFunction.TANH,
            ActivationFunction.SIGMOID,
            ActivationFunction.LEAKY_RELU
        ]

        for activation in activations:
            with self.subTest(activation=activation):
                layer_config = [{'units': 8, 'activation': activation}]
                model = ConfigurableMLP(input_dim, output_dim, layer_config)

                x = torch.randn(3, input_dim)
                output = model(x)

                self.assertEqual(output.shape, (3, output_dim))

    def test_mlp_parameter_count(self):
        """Test that model has expected number of parameters"""
        input_dim = 10
        output_dim = 2
        layer_config = [{'units': 5, 'activation': ActivationFunction.RELU}]

        model = ConfigurableMLP(input_dim, output_dim, layer_config)

        # Calculate expected parameters
        # Layer 1: 10 * 5 weights + 5 biases = 55
        # Layer 2: 5 * 2 weights + 2 biases = 12
        # Total: 67

        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 67)


class TestNeuralNetworkTraining(TestCase):
    """Test suite for neural network training"""

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

    def test_nn_binary_classification(self):
        """Test neural network binary classification training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=5, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [
                    {'units': 16, 'activation': ActivationFunction.RELU},
                    {'units': 8, 'activation': ActivationFunction.RELU}
                ],
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.01,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)

        # Verify analytics data
        self.assertIn('accuracy', model.analytics_data)
        self.assertIn('confusion_matrix', model.analytics_data)
        self.assertIn('training_history', model.analytics_data)

    def test_nn_multiclass_classification(self):
        """Test neural network multi-class classification"""
        df, features, target = MockDataGenerator.create_multiclass_data(
            n_samples=150, n_features=5, n_classes=5
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [
                    {'units': 32, 'activation': ActivationFunction.RELU},
                    {'units': 16, 'activation': ActivationFunction.RELU}
                ],
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.01,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)

        # Confusion matrix should be 5x5
        conf_matrix = model.analytics_data.get('confusion_matrix')
        self.assertEqual(len(conf_matrix), 5)

    def test_nn_training_history_recorded(self):
        """Test that training history is properly recorded"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        epochs = 10
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': epochs,
                'batch_size': 8,
                'learning_rate': 0.01,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_nn(model, X_train, y_train, X_test, y_test)

        training_history = model.analytics_data.get('training_history', [])

        # Should have one entry per epoch
        self.assertEqual(len(training_history), epochs)

        # Each entry should have epoch number
        for i, entry in enumerate(training_history, 1):
            self.assertEqual(entry.get('epoch'), i)

    def test_nn_prediction_schema_saved(self):
        """Test that prediction schema is saved after training"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=3
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 0.01,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_nn(model, X_train, y_train, X_test, y_test)

        # Verify prediction schema
        self.assertIsNotNone(model.prediction_schema)
        self.assertIn('input_features', model.prediction_schema)
        self.assertIn('output_classes', model.prediction_schema)
        self.assertEqual(len(model.prediction_schema['output_classes']), 3)

    def test_nn_class_names_stored(self):
        """Test that class names are stored in training config"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=3
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 3,
                'batch_size': 8,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_nn(model, X_train, y_train, X_test, y_test)

        # Class names should be stored
        self.assertIn('class_names', model.training_config)
        self.assertEqual(len(model.training_config['class_names']), 3)

    def test_nn_handles_non_numeric_features(self):
        """Test that NN handles non-numeric features by converting them"""
        # Create data with some string columns
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(50),
            'feature_1': np.random.randn(50),
            'feature_2': ['1.5'] * 25 + ['2.5'] * 25,  # String numbers
            'target': np.random.randint(0, 2, 50)
        })

        features = ['feature_0', 'feature_1', 'feature_2']
        target = 'target'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 3,
                'batch_size': 8,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        # Should not raise error
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_model_upload_called(self):
        """Test that model and encoder are uploaded to MinIO"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 3,
                'batch_size': 8,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        train_nn(model, X_train, y_train, X_test, y_test)

        # Upload should be called at least twice (model + encoder)
        self.assertGreaterEqual(self.mock_upload.call_count, 2)

    def test_nn_different_activations(self):
        """Test NN with different activation functions"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        activations = [
            ActivationFunction.RELU,
            ActivationFunction.TANH,
            ActivationFunction.SIGMOID,
            ActivationFunction.LEAKY_RELU
        ]

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        for activation in activations:
            with self.subTest(activation=activation):
                model = MockModelFactory.create_mock_mlmodel(
                    model_type=ModelType.NEURAL_NETWORK,
                    target_column=target,
                    training_config={
                        'layer_config': [{'units': 8, 'activation': activation}],
                        'epochs': 2,
                        'batch_size': 8,
                        'use_cuda': False
                    }
                )

                model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
                self.assertIsNotNone(model_path)

    def test_nn_batch_size_variations(self):
        """Test NN with different batch sizes"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        batch_sizes = [4, 8, 16, 32]

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                model = MockModelFactory.create_mock_mlmodel(
                    model_type=ModelType.NEURAL_NETWORK,
                    target_column=target,
                    training_config={
                        'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                        'epochs': 2,
                        'batch_size': batch_size,
                        'use_cuda': False
                    }
                )

                model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
                self.assertIsNotNone(model_path)


class TestGetDevice(TestCase):
    """Test the device selection function"""

    def test_get_device_cpu_when_cuda_disabled(self):
        """Test that CPU is returned when CUDA is disabled"""
        device = get_device(use_cuda=False)
        self.assertEqual(device, torch.device('cpu'))

    def test_get_device_with_logger(self):
        """Test device selection with logger"""
        mock_logger = MockTrainingLogger()
        device = get_device(use_cuda=False, logger=mock_logger)

        self.assertEqual(device, torch.device('cpu'))
        # Logger should have been called
        self.assertTrue(any('CPU' in log for log in mock_logger.logs))


class TestNeuralNetworkEdgeCases(TestCase):
    """Test edge cases for neural network training"""

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
        """Helper to split data"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def test_nn_single_epoch(self):
        """Test training with single epoch"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 1,
                'batch_size': 8,
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertEqual(len(model.analytics_data['training_history']), 1)

    def test_nn_small_batch_size(self):
        """Test training with very small batch size"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 2,
                'batch_size': 2,  # Very small batch
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)

    def test_nn_high_learning_rate(self):
        """Test training with high learning rate"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'layer_config': [{'units': 8, 'activation': ActivationFunction.RELU}],
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 0.1,  # High learning rate
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)

        # Should complete without error
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_default_config(self):
        """Test training with default config values"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=3, n_classes=2
        )

        # Minimal config - should use defaults
        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config={
                'use_cuda': False
            }
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)


class TestNeuralNetworkExtendedOptions(TestCase):
    """Test extended neural network configuration options"""

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
        """Helper to split data"""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _get_base_config(self):
        """Get base config for tests"""
        return {
            'layer_config': [{'units': 16, 'activation': 'relu'}],
            'epochs': 3,
            'batch_size': 16,
            'use_cuda': False
        }

    # ==================== Optimizer Tests ====================

    def test_nn_optimizer_adam(self):
        """Test NN with Adam optimizer"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['optimizer'] = 'adam'
        config['learning_rate'] = 0.001

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_optimizer_adamw(self):
        """Test NN with AdamW optimizer"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['optimizer'] = 'adamw'
        config['weight_decay'] = 0.01

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_optimizer_sgd_with_momentum(self):
        """Test NN with SGD optimizer and momentum"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['optimizer'] = 'sgd'
        config['momentum'] = 0.9
        config['learning_rate'] = 0.01

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_optimizer_rmsprop(self):
        """Test NN with RMSprop optimizer"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['optimizer'] = 'rmsprop'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== LR Scheduler Tests ====================

    def test_nn_scheduler_plateau(self):
        """Test NN with ReduceLROnPlateau scheduler"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 5
        config['lr_scheduler'] = 'plateau'
        config['lr_patience'] = 2
        config['lr_factor'] = 0.5

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_scheduler_cosine(self):
        """Test NN with Cosine Annealing scheduler"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 5
        config['lr_scheduler'] = 'cosine'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_scheduler_step(self):
        """Test NN with Step LR scheduler"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 6
        config['lr_scheduler'] = 'step'
        config['lr_patience'] = 2
        config['lr_factor'] = 0.5

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== Early Stopping Tests ====================

    def test_nn_early_stopping_enabled(self):
        """Test NN with early stopping enabled"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=80, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 50  # High number to trigger early stopping
        config['early_stopping'] = True
        config['early_stopping_patience'] = 3

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_early_stopping_disabled(self):
        """Test NN with early stopping disabled"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 5
        config['early_stopping'] = False

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        # Without early stopping, should run all epochs
        self.assertEqual(len(model.analytics_data['training_history']), 5)

    # ==================== Architecture Tests ====================

    def test_nn_with_batch_normalization(self):
        """Test NN with batch normalization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['use_batch_norm'] = True
        config['layer_config'] = [
            {'units': 32, 'activation': 'relu'},
            {'units': 16, 'activation': 'relu'}
        ]

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_with_dropout(self):
        """Test NN with dropout regularization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['dropout'] = 0.3
        config['layer_config'] = [
            {'units': 32, 'activation': 'relu'},
            {'units': 16, 'activation': 'relu'}
        ]

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_init_method_kaiming(self):
        """Test NN with Kaiming weight initialization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['init_method'] = 'kaiming'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_init_method_xavier(self):
        """Test NN with Xavier weight initialization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['init_method'] = 'xavier'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== Loss Function Tests ====================

    def test_nn_loss_cross_entropy(self):
        """Test NN with CrossEntropy loss"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=3
        )

        config = self._get_base_config()
        config['loss_function'] = 'cross_entropy'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_loss_focal(self):
        """Test NN with Focal loss for imbalanced classes"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['loss_function'] = 'focal'
        config['focal_gamma'] = 2.0

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_with_class_weights(self):
        """Test NN with automatic class weights"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['class_weights'] = True

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_with_label_smoothing(self):
        """Test NN with label smoothing"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=3
        )

        config = self._get_base_config()
        config['label_smoothing'] = 0.1

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== Advanced Options Tests ====================

    def test_nn_with_gradient_clipping(self):
        """Test NN with gradient clipping"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['gradient_clipping'] = 1.0

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_with_warmup_epochs(self):
        """Test NN with learning rate warmup"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['epochs'] = 10
        config['warmup_epochs'] = 3

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== Feature Normalization Tests ====================

    def test_nn_normalize_standard(self):
        """Test NN with standard feature normalization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['normalize_features'] = 'standard'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_normalize_minmax(self):
        """Test NN with MinMax feature normalization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['normalize_features'] = 'minmax'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_normalize_robust(self):
        """Test NN with Robust feature normalization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['normalize_features'] = 'robust'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    def test_nn_normalize_none(self):
        """Test NN without feature normalization"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=50, n_features=5, n_classes=2
        )

        config = self._get_base_config()
        config['normalize_features'] = 'none'

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)
        self.assertIsNotNone(model_path)

    # ==================== Full Configuration Test ====================

    def test_nn_full_configuration(self):
        """Test NN with all configuration options enabled"""
        df, features, target = MockDataGenerator.create_tabular_classification_data(
            n_samples=100, n_features=10, n_classes=3
        )

        config = {
            # Training
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'momentum': 0.9,
            'weight_decay': 0.0001,

            # Architecture
            'layer_config': [
                {'units': 64, 'activation': 'relu'},
                {'units': 32, 'activation': 'relu'},
            ],
            'use_batch_norm': True,
            'init_method': 'kaiming',
            'dropout': 0.2,

            # LR Scheduler
            'lr_scheduler': 'plateau',
            'lr_patience': 3,
            'lr_factor': 0.5,
            'min_lr': 1e-6,

            # Early Stopping
            'early_stopping': True,
            'early_stopping_patience': 5,

            # Loss
            'loss_function': 'cross_entropy',
            'class_weights': True,
            'label_smoothing': 0.1,

            # Advanced
            'gradient_clipping': 1.0,
            'warmup_epochs': 2,
            'normalize_features': 'standard',

            # Disable CUDA for tests
            'use_cuda': False
        }

        model = MockModelFactory.create_mock_mlmodel(
            model_type=ModelType.NEURAL_NETWORK,
            target_column=target,
            training_config=config
        )

        X_train, X_test, y_train, y_test = self._split_data(df, features, target)
        model_path, accuracy = train_nn(model, X_train, y_train, X_test, y_test)

        self.assertIsNotNone(model_path)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertIn('accuracy', model.analytics_data)
        self.assertIn('training_history', model.analytics_data)
        self.assertIn('confusion_matrix', model.analytics_data)


class TestConfigurableMLPExtended(TestCase):
    """Extended tests for ConfigurableMLP architecture"""

    def test_mlp_with_dropout_layers(self):
        """Test MLP with per-layer dropout"""
        layer_config = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.3},
            {'units': 32, 'activation': 'relu', 'dropout': 0.2},
        ]
        model = ConfigurableMLP(
            input_dim=10, output_dim=3, layer_config=layer_config, dropout=0.1
        )

        # Check forward pass works
        model.eval()  # Dropout behaves differently in eval mode
        x = torch.randn(5, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 3))

    def test_mlp_with_batch_norm(self):
        """Test MLP with batch normalization"""
        layer_config = [
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
        ]
        model = ConfigurableMLP(
            input_dim=10, output_dim=3, layer_config=layer_config, use_batch_norm=True
        )

        # Count batch norm layers
        bn_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
        self.assertEqual(bn_count, 2)  # One per hidden layer

    def test_mlp_kaiming_init(self):
        """Test MLP with Kaiming initialization"""
        layer_config = [{'units': 64, 'activation': 'relu'}]
        model = ConfigurableMLP(
            input_dim=10, output_dim=3, layer_config=layer_config, init_method='kaiming'
        )

        # Check weights are not all zeros
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                self.assertFalse(torch.all(m.weight == 0))
                self.assertTrue(torch.all(m.bias == 0))  # Biases should be zero

    def test_mlp_xavier_init(self):
        """Test MLP with Xavier initialization"""
        layer_config = [{'units': 64, 'activation': 'tanh'}]
        model = ConfigurableMLP(
            input_dim=10, output_dim=3, layer_config=layer_config, init_method='xavier'
        )

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                self.assertFalse(torch.all(m.weight == 0))

    def test_mlp_string_activation(self):
        """Test MLP with string activation names (frontend format)"""
        layer_config = [
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'tanh'},
            {'units': 16, 'activation': 'sigmoid'},
        ]
        model = ConfigurableMLP(input_dim=10, output_dim=3, layer_config=layer_config)

        x = torch.randn(5, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 3))

    def test_mlp_enum_activation(self):
        """Test MLP with enum activation names (backwards compatibility)"""
        layer_config = [
            {'units': 64, 'activation': ActivationFunction.RELU},
            {'units': 32, 'activation': ActivationFunction.TANH},
        ]
        model = ConfigurableMLP(input_dim=10, output_dim=3, layer_config=layer_config)

        x = torch.randn(5, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 3))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
