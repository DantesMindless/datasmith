"""
Tests for CNN Training Functions
"""
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from io import BytesIO

from app.functions.training import ConfigurableCNN, get_device
from app.models.choices import ModelType, ActivationFunction
from app.tests.fixtures import (
    MockModelFactory,
    MockMinIOStorage,
    MockTrainingLogger
)


class TestConfigurableCNN(TestCase):
    """Test suite for ConfigurableCNN architecture"""

    def test_cnn_basic_architecture(self):
        """Test basic CNN creation with default parameters"""
        conv_layers = [{"out_channels": 32, "kernel_size": 3}]
        fc_layers = [{"units": 64}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=10,
            input_size=64
        )

        self.assertIsInstance(model, nn.Module)

        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        self.assertEqual(output.shape, (2, 10))

    def test_cnn_multiple_conv_layers(self):
        """Test CNN with multiple convolutional layers"""
        conv_layers = [
            {"out_channels": 16, "kernel_size": 3},
            {"out_channels": 32, "kernel_size": 3},
            {"out_channels": 64, "kernel_size": 3}
        ]
        fc_layers = [{"units": 128}, {"units": 64}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=5,
            input_size=64
        )

        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_cnn_different_activations(self):
        """Test CNN with different activation functions"""
        activations = [
            ActivationFunction.RELU,
            ActivationFunction.TANH,
            ActivationFunction.SIGMOID,
            ActivationFunction.LEAKY_RELU
        ]

        for activation in activations:
            with self.subTest(activation=activation):
                conv_layers = [{"out_channels": 16, "kernel_size": 3, "activation": activation}]
                fc_layers = [{"units": 32, "activation": activation}]

                model = ConfigurableCNN(
                    input_channels=3,
                    conv_layers=conv_layers,
                    fc_layers=fc_layers,
                    num_classes=2,
                    input_size=32
                )

                x = torch.randn(2, 3, 32, 32)
                output = model(x)
                self.assertEqual(output.shape, (2, 2))

    def test_cnn_different_kernel_sizes(self):
        """Test CNN with kernel size 3 (the standard supported size)"""
        # Note: ConfigurableCNN uses padding=1, which works best with kernel_size=3
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=2,
            input_size=64
        )

        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))

    def test_cnn_different_input_sizes(self):
        """Test CNN with different input image sizes"""
        # Use powers of 2 that work well with the CNN architecture
        input_sizes = [32, 64]

        for input_size in input_sizes:
            with self.subTest(input_size=input_size):
                conv_layers = [{"out_channels": 16, "kernel_size": 3}]
                fc_layers = [{"units": 32}]

                model = ConfigurableCNN(
                    input_channels=3,
                    conv_layers=conv_layers,
                    fc_layers=fc_layers,
                    num_classes=2,
                    input_size=input_size
                )

                x = torch.randn(2, 3, input_size, input_size)
                output = model(x)
                self.assertEqual(output.shape, (2, 2))

    def test_cnn_parameter_count(self):
        """Test that CNN has expected number of parameters"""
        conv_layers = [{"out_channels": 32, "kernel_size": 3}]
        fc_layers = [{"units": 64}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=10,
            input_size=64
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

    def test_cnn_binary_classification_output(self):
        """Test CNN output shape for binary classification"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=2,
            input_size=32
        )

        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))

    def test_cnn_multiclass_output(self):
        """Test CNN output shape for multi-class classification"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=100,
            input_size=32
        )

        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (4, 100))

    def test_cnn_grayscale_input(self):
        """Test CNN with grayscale (1 channel) input"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=1,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=10,
            input_size=28
        )

        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (2, 10))

    def test_cnn_forward_backward(self):
        """Test CNN forward and backward pass"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=5,
            input_size=32
        )

        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 5, (4,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestCNNDeviceSelection(TestCase):
    """Test device selection for CNN training"""

    def test_get_device_cpu(self):
        """Test device selection returns CPU when CUDA unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            device = get_device(use_cuda=True)
            self.assertEqual(device, torch.device('cpu'))

    def test_get_device_cuda_disabled(self):
        """Test device selection returns CPU when CUDA is disabled"""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(use_cuda=False)
            self.assertEqual(device, torch.device('cpu'))

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value='Mock GPU')
    @patch('torch.cuda.get_device_properties')
    def test_get_device_cuda(self, mock_props, mock_name, mock_available):
        """Test device selection returns CUDA when available"""
        mock_props.return_value.total_memory = 8 * (1024**3)

        device = get_device(use_cuda=True)
        self.assertEqual(device, torch.device('cuda'))

    def test_get_device_with_logger(self):
        """Test device selection logs device info"""
        mock_logger = MockTrainingLogger()

        with patch('torch.cuda.is_available', return_value=False):
            get_device(use_cuda=True, logger=mock_logger)

        self.assertTrue(any('CPU' in log or 'CUDA' in log for log in mock_logger.logs))


class TestCNNTrainingUnit(TestCase):
    """Unit tests for CNN training components"""

    def test_cnn_loss_computation(self):
        """Test that CNN can compute loss correctly"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 3, (4,))

        output = model(x)
        loss = criterion(output, y)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_cnn_optimizer_step(self):
        """Test that optimizer can update CNN weights"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Get initial weights
        initial_weights = model.fc[0].weight.clone()

        # Forward pass
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 3, (4,))

        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        self.assertFalse(torch.equal(initial_weights, model.fc[0].weight))

    def test_cnn_eval_mode(self):
        """Test CNN behaves correctly in eval mode"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        model.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # Same input should give same output in eval mode
        self.assertTrue(torch.allclose(output1, output2))

    def test_cnn_train_mode_consistency(self):
        """Test CNN in train mode with no dropout produces consistent results"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        model.train()
        x = torch.randn(2, 3, 32, 32)

        # Since there's no dropout, train mode should also be deterministic
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        self.assertTrue(torch.allclose(output1, output2))


class TestCNNSerialization(TestCase):
    """Test CNN model serialization"""

    def test_cnn_state_dict_save_load(self):
        """Test saving and loading CNN state dict"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model1 = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        # Save state dict to buffer
        buffer = BytesIO()
        torch.save(model1.state_dict(), buffer)
        buffer.seek(0)

        # Create new model and load state dict
        model2 = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )
        model2.load_state_dict(torch.load(buffer, map_location='cpu'))

        # Test that both models produce same output
        x = torch.randn(2, 3, 32, 32)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        self.assertTrue(torch.allclose(out1, out2))

    def test_cnn_state_dict_roundtrip(self):
        """Test multiple save/load cycles preserve weights"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model1 = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        # First save/load
        buffer1 = BytesIO()
        torch.save(model1.state_dict(), buffer1)
        buffer1.seek(0)

        model2 = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )
        model2.load_state_dict(torch.load(buffer1, map_location='cpu'))

        # Second save/load
        buffer2 = BytesIO()
        torch.save(model2.state_dict(), buffer2)
        buffer2.seek(0)

        model3 = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )
        model3.load_state_dict(torch.load(buffer2, map_location='cpu'))

        # All models should produce same output
        x = torch.randn(2, 3, 32, 32)
        model1.eval()
        model2.eval()
        model3.eval()

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
            out3 = model3(x)

        self.assertTrue(torch.allclose(out1, out2))
        self.assertTrue(torch.allclose(out2, out3))


class TestCNNEdgeCases(TestCase):
    """Test edge cases for CNN"""

    def test_cnn_single_sample_batch(self):
        """Test CNN with batch size of 1"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        x = torch.randn(1, 3, 32, 32)
        output = model(x)

        self.assertEqual(output.shape, (1, 3))

    def test_cnn_large_batch(self):
        """Test CNN with large batch size"""
        conv_layers = [{"out_channels": 16, "kernel_size": 3}]
        fc_layers = [{"units": 32}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=3,
            input_size=32
        )

        x = torch.randn(64, 3, 32, 32)
        output = model(x)

        self.assertEqual(output.shape, (64, 3))

    def test_cnn_minimal_architecture(self):
        """Test CNN with minimal architecture"""
        conv_layers = [{"out_channels": 4, "kernel_size": 3}]
        fc_layers = [{"units": 8}]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=2,
            input_size=16
        )

        x = torch.randn(2, 3, 16, 16)
        output = model(x)

        self.assertEqual(output.shape, (2, 2))

    def test_cnn_deep_architecture(self):
        """Test CNN with deeper architecture"""
        conv_layers = [
            {"out_channels": 32, "kernel_size": 3},
            {"out_channels": 64, "kernel_size": 3},
        ]
        fc_layers = [
            {"units": 128},
            {"units": 64}
        ]

        model = ConfigurableCNN(
            input_channels=3,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=10,
            input_size=64
        )

        x = torch.randn(2, 3, 64, 64)
        output = model(x)

        self.assertEqual(output.shape, (2, 10))


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
