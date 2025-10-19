import os
import io
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from app.models.choices import ActivationFunction, ModelType
from core.storage_utils import upload_to_minio

ACTIVATION_MAP = {
    ActivationFunction.RELU: nn.ReLU,
    ActivationFunction.TANH: nn.Tanh,
    ActivationFunction.SIGMOID: nn.Sigmoid,
    ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
}


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_config):
        super().__init__()
        layers = []
        dims = [input_dim]

        for config in layer_config:
            units = config.get("units", 32)
            act = config.get("activation", ActivationFunction.RELU)
            act_class = ACTIVATION_MAP.get(act, nn.ReLU)
            layers.append(nn.Linear(dims[-1], units))
            layers.append(act_class())
            dims.append(units)

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConfigurableCNN(nn.Module):
    def __init__(self, input_channels, conv_layers, fc_layers, num_classes, input_size):
        super().__init__()
        layers = []
        in_channels = input_channels
        for layer in conv_layers:
            out_channels = layer.get("out_channels", 16)
            kernel_size = layer.get("kernel_size", 3)
            act = layer.get("activation", ActivationFunction.RELU)
            act_class = ACTIVATION_MAP.get(act, nn.ReLU)
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
            )
            layers.append(nn.MaxPool2d(2))
            layers.append(act_class())
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        conv_output_size = input_size // (2 ** len(conv_layers))
        self.flatten = nn.Flatten()
        fc_input_dim = in_channels * conv_output_size * conv_output_size

        fc = []
        dims = [fc_input_dim]
        for config in fc_layers:
            units = config.get("units", 64)
            act = config.get("activation", ActivationFunction.RELU)
            act_class = ACTIVATION_MAP.get(act, nn.ReLU)
            fc.append(nn.Linear(dims[-1], units))
            fc.append(act_class())
            dims.append(units)
        fc.append(nn.Linear(dims[-1], num_classes))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


def get_model_instance(model_type, obj):
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(max_iter=obj.max_iter)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(random_state=obj.random_state)
    elif model_type == ModelType.SVM:
        return SVC(max_iter=obj.max_iter)
    elif model_type == ModelType.NAIVE_BAYES:
        return GaussianNB()
    elif model_type == ModelType.KNN:
        return KNeighborsClassifier()
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_sklearn_model(obj, X_train, y_train, X_test, y_test):
    clf = get_model_instance(obj.model_type, obj)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    buffer = io.BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)

    model_path = f"trained_models/{obj.id}.joblib"
    upload_to_minio(buffer.read(), model_path, content_type="application/octet-stream")

    # Save prediction schema
    feature_names = X_train.columns.tolist()
    obj.prediction_schema = {
        "input_features": feature_names,
        "feature_count": len(feature_names),
        "target_column": obj.target_column,
        "example": {col: "numeric_value" for col in feature_names},
        "description": f"Provide values for these {len(feature_names)} features to make a prediction"
    }
    obj.save(update_fields=["prediction_schema"])

    return model_path, acc


def train_nn(obj, X_train, y_train, X_test, y_test):
    config = obj.training_config or {}
    layer_config = config.get(
        "layer_config", [{"units": 32, "activation": ActivationFunction.RELU}]
    )
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train_encoded))

    model = ConfigurableMLP(input_dim, output_dim, layer_config)
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

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        preds = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test_encoded, preds)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_path = f"trained_models/{obj.id}.pt"

    upload_to_minio(buffer.read(), model_path, content_type="application/octet-stream")

    enc_buf = io.BytesIO()
    joblib.dump(le, enc_buf)
    enc_buf.seek(0)
    encoder_path = f"trained_models/{obj.id}_encoder.joblib"

    upload_to_minio(
        enc_buf.read(), encoder_path, content_type="application/octet-stream"
    )

    obj.training_config["output_dim"] = output_dim
    obj.training_config["class_names"] = le.classes_.tolist()

    # Save prediction schema
    feature_names = X_train.columns.tolist()
    obj.prediction_schema = {
        "input_features": feature_names,
        "feature_count": len(feature_names),
        "target_column": obj.target_column,
        "output_classes": le.classes_.tolist(),
        "example": {col: "numeric_value" for col in feature_names},
        "description": f"Provide values for these {len(feature_names)} features to predict one of {output_dim} classes"
    }

    obj.save(update_fields=["training_config", "prediction_schema"])

    return model_path, acc


def train_cnn(obj):
    image_path = None

    if hasattr(obj.dataset, "extracted_path") and obj.dataset.extracted_path:
        image_path = obj.dataset.extracted_path
    elif hasattr(obj.dataset, "image_folder") and obj.dataset.image_folder:
        image_path = obj.dataset.image_folder.path

    if not image_path or not os.path.isdir(image_path):
        raise ValueError(f"Image path is invalid or missing: {image_path}")

    config = obj.training_config or {}
    conv_layers = config.get("conv_layers", [{"out_channels": 32}])
    fc_layers = config.get("fc_layers", [{"units": 128}])
    input_size = config.get("input_size", 64)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(image_path, transform=transform)

    # Split dataset for training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)
    model = ConfigurableCNN(3, conv_layers, fc_layers, num_classes, input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Calculate validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total if total > 0 else 0.0

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_path = f"trained_models/{obj.id}.pt"
    upload_to_minio(buffer.read(), model_path, content_type="application/octet-stream")

    obj.training_config["num_classes"] = num_classes
    obj.training_config["class_names"] = list(dataset.class_to_idx.keys())

    # Save prediction schema for image models
    class_names = list(dataset.class_to_idx.keys())
    obj.prediction_schema = {
        "input_type": "image",
        "input_size": input_size,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif"],
        "output_classes": class_names,
        "num_classes": num_classes,
        "example": "Upload an image file",
        "description": f"Upload an image to classify into one of {num_classes} categories: {', '.join(class_names)}"
    }

    obj.save(update_fields=["training_config", "prediction_schema"])

    return model_path, accuracy
