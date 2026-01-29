import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class ConfigurableMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for classification tasks"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Union[List[int], List[dict], int] = None, 
                 activation: str = 'relu', dropout_rate: float = 0.2):
        super(ConfigurableMLP, self).__init__()
        
        # Handle different input formats for hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [64, 32]  # Default
        elif isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]  # Convert single int to list
        elif isinstance(hidden_sizes, list) and hidden_sizes and isinstance(hidden_sizes[0], dict):
            # Convert dict format to size list
            hidden_sizes = [layer['size'] for layer in hidden_sizes if 'size' in layer]
        elif not isinstance(hidden_sizes, list):
            hidden_sizes = [64, 32]  # Fallback default
        
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                # Apply activation to hidden layers only
                x = layer(x)
                x = self._get_activation()(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                # Output layer
                x = layer(x)
        return x
    
    def _get_activation(self):
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu
        }
        return activations.get(self.activation, F.relu)


class ConfigurableCNN(nn.Module):
    """Configurable CNN for image classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10, 
                 input_size: Union[int, tuple] = 32, conv_layers: List[int] = None, 
                 fc_layers: List[int] = None):
        super(ConfigurableCNN, self).__init__()
        
        if conv_layers is None:
            conv_layers = [32, 64]
        if fc_layers is None:
            fc_layers = [256]
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        # Build convolutional layers
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = out_channels
        
        # Calculate size after conv layers
        # Handle both int and tuple input_size
        if isinstance(input_size, tuple):
            conv_h, conv_w = input_size
        else:
            conv_h = conv_w = input_size
            
        for _ in conv_layers:
            conv_h = conv_h // 2
            conv_w = conv_w // 2
        
        self.fc_input_size = in_channels * conv_h * conv_w
        
        # Build fully connected layers
        fc_layer_list = []
        prev_size = self.fc_input_size
        
        for fc_size in fc_layers:
            fc_layer_list.append(nn.Linear(prev_size, fc_size))
            fc_layer_list.append(nn.ReLU(inplace=True))
            fc_layer_list.append(nn.Dropout(0.5))
            prev_size = fc_size
        
        # Final output layer
        fc_layer_list.append(nn.Linear(prev_size, num_classes))
        
        self.fc = nn.Sequential(*fc_layer_list)
        
    def forward(self, x):
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        return x