import io
import joblib
import torch
import os
from PIL import Image
from torchvision import transforms
from app.functions.training import ConfigurableMLP, ConfigurableCNN
from app.models.choices import ActivationFunction

from core.storage_utils import download_from_minio


def predict_sklearn_model(obj, df):
    config = obj.training_config or {}
    features = config.get("features") or df.columns.tolist()

    if not all(col in df.columns for col in features):
        raise ValueError("Missing one or more input features in CSV.")

    X = df[features]
    model_path = f"trained_models/{obj.id}.joblib"
    model_bytes = download_from_minio(model_path)
    model = joblib.load(io.BytesIO(model_bytes))

    df["prediction"] = model.predict(X)
    return df


def predict_nn(obj, df):
    config = obj.training_config or {}
    features = config.get("features") or df.columns.tolist()
    layer_config = config.get(
        "layer_config", [{"units": 32, "activation": ActivationFunction.RELU}]
    )

    input_dim = len(features)
    output_dim = config.get("output_dim") or config.get("num_classes") or 3

    model = ConfigurableMLP(input_dim, output_dim, layer_config)
    model_path = f"trained_models/{obj.id}.pt"
    encoder_path = f"trained_models/{obj.id}_encoder.joblib"

    model_bytes = download_from_minio(model_path)
    model.load_state_dict(
        torch.load(io.BytesIO(model_bytes), map_location=torch.device("cpu"))
    )
    model.eval()

    encoder_bytes = download_from_minio(encoder_path)
    le = joblib.load(io.BytesIO(encoder_bytes))

    X_tensor = torch.tensor(df[features].values, dtype=torch.float32)
    with torch.no_grad():
        raw_preds = model(X_tensor).argmax(dim=1).numpy()
        preds = le.inverse_transform(raw_preds)

    df["_model_used"] = os.path.basename(model_path)
    df["prediction"] = preds
    return df


def predict_cnn(model_obj, image_path):
    config = model_obj.training_config or {}
    conv_layers = config.get("conv_layers", [{"out_channels": 32}])
    fc_layers = config.get("fc_layers", [{"units": 128}])
    input_size = config.get("input_size", 64)
    num_classes = config.get("num_classes", 2)

    model = ConfigurableCNN(3, conv_layers, fc_layers, num_classes, input_size)

    model_path = f"trained_models/{model_obj.id}.pt"
    model_bytes = download_from_minio(model_path)
    buffer = io.BytesIO(model_bytes)
    model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = output.argmax(dim=1).item()

    return predicted_idx
