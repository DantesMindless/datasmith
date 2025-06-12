
import joblib
import torch
import os
from app.functions.training import ConfigurableMLP
from app.models.choices import ActivationFunction

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
    layer_config = config.get("layer_config", [{"units": 32, "activation": ActivationFunction.RELU}])

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
