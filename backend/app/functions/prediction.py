import joblib
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from app.functions.training import ConfigurableMLP, ConfigurableCNN
from app.models.choices import ActivationFunction
from django.conf import settings

def predict_sklearn_model(model_obj, df):
    model_path = model_obj.model_file.path
    clf = joblib.load(model_path)
    config = model_obj.training_config or {}
    features = config.get("features") or df.columns.tolist()
    preds = clf.predict(df[features])
    df["prediction"] = preds
    df["_model_used"] = os.path.basename(model_path)
    return df


def predict_nn(model_obj, df):
    config = model_obj.training_config or {}
    features = config.get("features") or df.columns.tolist()
    layer_config = config.get(
        "layer_config", [{"units": 32, "activation": ActivationFunction.RELU}]
    )

    input_dim = len(features)
    output_dim = config.get("output_dim") or 3

    model = ConfigurableMLP(input_dim, output_dim, layer_config)
    model_path = model_obj.model_file.path
    encoder_path = model_path.replace(".pt", "_encoder.joblib")
    le = joblib.load(encoder_path)

    model.load_state_dict(torch.load(model_path))
    model.eval()

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
    
    model_path = os.path.join(settings.MEDIA_ROOT, "trained_models", f"{model_obj.id}.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = output.argmax(dim=1).item()

    return predicted_idx  