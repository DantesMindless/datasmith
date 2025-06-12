import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

from app.models.choices import ActivationFunction


ACTIVATION_MAP = {
    ActivationFunction.RELU: nn.ReLU,
    ActivationFunction.TANH: nn.Tanh,
    ActivationFunction.SIGMOID: nn.Sigmoid,
    ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
}


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation):
        super().__init__()
        act_class = ACTIVATION_MAP.get(activation, nn.ReLU)
        layers = []
        dims = [input_dim] + hidden_layers

        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_class())

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_neural_network(obj, X_train, y_train, X_test, y_test, media_root=None):
    config = obj.training_config or {}

    hidden_layers = config.get("hidden_layers", [32, 32])
    activation = config.get("activation", ActivationFunction.RELU)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train_encoded))

    model = ConfigurableMLP(input_dim, output_dim, hidden_layers, activation)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = data.TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.long),
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        preds = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test_encoded, preds)

    # Save model
    if media_root is None:
        from django.conf import settings

        media_root = settings.MEDIA_ROOT

    model_dir = os.path.join(media_root, "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{obj.id}.pt")
    torch.save(model.state_dict(), model_path)

    return model_path, acc
