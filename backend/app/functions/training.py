import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from django.conf import settings
from app.models.choices import ActivationFunction, ModelType

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

    model_dir = os.path.join(settings.MEDIA_ROOT, "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{obj.id}.joblib")
    joblib.dump(clf, model_path)
    return model_path, acc

def train_nn(obj, X_train, y_train, X_test, y_test):
    config = obj.training_config or {}
    layer_config = config.get("layer_config", [{"units": 32, "activation": ActivationFunction.RELU}])
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

    model_dir = os.path.join(settings.MEDIA_ROOT, "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{obj.id}.pt")
    torch.save(model.state_dict(), model_path)
    encoder_path = os.path.join(model_dir, f"{obj.id}_encoder.joblib")
    joblib.dump(le, encoder_path)
    return model_path, acc

