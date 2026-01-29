import io
import joblib
import torch
import pandas as pd
import os
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
from app.functions.training import ConfigurableMLP, ConfigurableCNN
from app.models.choices import ActivationFunction

from core.storage_utils import download_from_minio

logger = logging.getLogger(__name__)


def preprocess_healthcare_data(df):
    """
    Convert string data to numerical format for healthcare predictions.
    Handles raw table data with string values.
    """
    logger.info("[PREPROCESS] Starting data preprocessing...")
    df = df.copy()

    # Convert gender: M/F to 1/0
    if 'gender' in df.columns:
        original = df['gender'].tolist()
        df['gender'] = df['gender'].astype(str)
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, '1': 1, '0': 0}).fillna(df['gender'])
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(0)
        logger.info(f"[PREPROCESS] gender: {original} -> {df['gender'].tolist()}")

    # Convert smoking_status: current/former/never to 2/1/0
    if 'smoking_status' in df.columns:
        original = df['smoking_status'].tolist()
        df['smoking_status'] = df['smoking_status'].astype(str)
        df['smoking_status'] = df['smoking_status'].map({
            'current': 2, 'former': 1, 'never': 0, '2': 2, '1': 1, '0': 0
        }).fillna(df['smoking_status'])
        df['smoking_status'] = pd.to_numeric(df['smoking_status'], errors='coerce').fillna(0)
        logger.info(f"[PREPROCESS] smoking_status: {original} -> {df['smoking_status'].tolist()}")

    # Convert boolean family_history: True/False to 1/0
    if 'family_history' in df.columns:
        original = df['family_history'].tolist()
        df['family_history'] = df['family_history'].astype(str)
        df['family_history'] = df['family_history'].map({
            'True': 1, 'False': 0, 'true': 1, 'false': 0, 'TRUE': 1, 'FALSE': 0,
            '1': 1, '0': 0, 1: 1, 0: 0
        }).fillna(df['family_history'])
        df['family_history'] = pd.to_numeric(df['family_history'], errors='coerce').fillna(0)
        logger.info(f"[PREPROCESS] family_history: {original} -> {df['family_history'].tolist()}")

    # Convert disease_risk: High/Low to 1/0 (if present, for reference)
    if 'disease_risk' in df.columns:
        original = df['disease_risk'].tolist()
        df['disease_risk'] = df['disease_risk'].astype(str)
        df['disease_risk'] = df['disease_risk'].map({
            'High': 1, 'Low': 0, 'high': 1, 'low': 0, 'HIGH': 1, 'LOW': 0,
            '1': 1, '0': 0
        }).fillna(df['disease_risk'])
        df['disease_risk'] = pd.to_numeric(df['disease_risk'], errors='coerce').fillna(0)
        logger.info(f"[PREPROCESS] disease_risk: {original} -> {df['disease_risk'].tolist()}")

    # Ensure all remaining columns are numeric
    for col in df.columns:
        if col not in ['gender', 'smoking_status', 'family_history', 'disease_risk']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info("[PREPROCESS] Preprocessing complete")
    return df


def predict_sklearn_model(obj, df):
    logger.info("=" * 60)
    logger.info(f"[PREDICTION] Starting sklearn prediction for model: {obj.name} (ID: {obj.id})")
    logger.info(f"[PREDICTION] Model type: {obj.model_type}")

    config = obj.training_config or {}
    logger.info(f"[PREDICTION] Training config features: {config.get('features', 'Not specified')}")

    # Log raw input
    logger.info(f"[PREDICTION] Raw input type: {type(df)}")
    if isinstance(df, dict):
        logger.info(f"[PREDICTION] Raw input (dict): {df}")
    else:
        logger.info(f"[PREDICTION] Raw input shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")
        logger.info(f"[PREDICTION] Raw input columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")

    # Convert dict to DataFrame if needed
    if isinstance(df, dict):
        df = pd.DataFrame([df])
        logger.info(f"[PREDICTION] Converted dict to DataFrame with shape: {df.shape}")

    # Log before preprocessing
    logger.info(f"[PREDICTION] Before preprocessing:")
    logger.info(f"[PREDICTION]   Columns: {list(df.columns)}")
    logger.info(f"[PREDICTION]   Dtypes: {df.dtypes.to_dict()}")
    logger.info(f"[PREDICTION]   Values (first row): {df.iloc[0].to_dict()}")

    # Preprocess raw string data to numerical format
    df = preprocess_healthcare_data(df)

    # Log after preprocessing
    logger.info(f"[PREDICTION] After preprocessing:")
    logger.info(f"[PREDICTION]   Dtypes: {df.dtypes.to_dict()}")
    logger.info(f"[PREDICTION]   Values (first row): {df.iloc[0].to_dict()}")

    features = config.get("features") or df.columns.tolist()
    logger.info(f"[PREDICTION] Using features: {features}")

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        logger.error(f"[PREDICTION] Missing features: {missing_features}")
        logger.error(f"[PREDICTION] Available columns: {list(df.columns)}")
        raise ValueError(f"Missing features in CSV: {missing_features}")

    X = df[features]
    logger.info(f"[PREDICTION] Feature matrix shape: {X.shape}")
    logger.info(f"[PREDICTION] Feature matrix values:\n{X.to_string()}")

    # Handle NaN values - fill with column means or 0 if column is all NaN
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        logger.warning(f"[PREDICTION] Found NaN values in columns: {nan_cols}")
        # Fill NaN with column mean, fallback to 0 if column is entirely NaN
        for col in nan_cols:
            col_mean = X[col].mean()
            if pd.isna(col_mean):
                X[col] = X[col].fillna(0)
                logger.info(f"[PREDICTION] Filled NaN in '{col}' with 0 (column was all NaN)")
            else:
                X[col] = X[col].fillna(col_mean)
                logger.info(f"[PREDICTION] Filled NaN in '{col}' with mean: {col_mean}")

    # Check if running in test mode with mocked file
    if hasattr(obj.model_file, '_mock_name'):
        logger.info("[PREDICTION] Using mocked model file (test mode)")
        model_bytes = obj.model_file.read()
        if isinstance(model_bytes, io.BytesIO):
            model = joblib.load(model_bytes)
        else:
            model = joblib.load(io.BytesIO(model_bytes))
    else:
        model_path = f"trained_models/{obj.id}.joblib"
        logger.info(f"[PREDICTION] Loading model from MinIO: {model_path}")
        model_bytes = download_from_minio(model_path)
        model = joblib.load(io.BytesIO(model_bytes))

    logger.info(f"[PREDICTION] Loaded model type: {type(model).__name__}")
    if hasattr(model, 'classes_'):
        logger.info(f"[PREDICTION] Model classes: {model.classes_}")
    if hasattr(model, 'n_features_in_'):
        logger.info(f"[PREDICTION] Model expected features: {model.n_features_in_}")

    raw_predictions = model.predict(X)
    logger.info(f"[PREDICTION] Raw predictions: {raw_predictions}")

    # Log prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X)
            logger.info(f"[PREDICTION] Prediction probabilities: {proba}")
        except Exception as e:
            logger.warning(f"[PREDICTION] Could not get probabilities: {e}")

    # Try to load label encoder to decode predictions back to original labels
    decoded_predictions = raw_predictions
    try:
        encoder_path = f"trained_models/{obj.id}_label_encoder.joblib"
        logger.info(f"[PREDICTION] Attempting to load label encoder from: {encoder_path}")
        encoder_bytes = download_from_minio(encoder_path)
        label_encoder = joblib.load(io.BytesIO(encoder_bytes))
        logger.info(f"[PREDICTION] Label encoder classes: {label_encoder.classes_}")

        # Decode predictions back to original labels
        decoded_predictions = label_encoder.inverse_transform(raw_predictions.astype(int))
        logger.info(f"[PREDICTION] Decoded predictions: {decoded_predictions.tolist()}")
    except Exception as e:
        logger.info(f"[PREDICTION] No label encoder found or decode failed: {e}")
        logger.info(f"[PREDICTION] Using raw predictions as final output")

        # Fallback: try to get class names from training config
        class_names = config.get('class_names', [])
        if class_names:
            logger.info(f"[PREDICTION] Using class_names from config: {class_names}")
            try:
                decoded_predictions = [class_names[int(p)] for p in raw_predictions]
                logger.info(f"[PREDICTION] Decoded from config: {decoded_predictions}")
            except (IndexError, ValueError) as decode_err:
                logger.warning(f"[PREDICTION] Could not decode using class_names: {decode_err}")

    df["prediction"] = decoded_predictions
    logger.info(f"[PREDICTION] Final predictions: {list(decoded_predictions)}")
    logger.info("=" * 60)

    return df


def predict_nn(obj, df):
    logger.info("=" * 60)
    logger.info(f"[PREDICTION-NN] Starting neural network prediction for model: {obj.name}")

    config = obj.training_config or {}
    features = config.get("features") or df.columns.tolist()
    layer_config = config.get(
        "layer_config", [{"units": 32, "activation": ActivationFunction.RELU}]
    )

    logger.info(f"[PREDICTION-NN] Features: {features}")
    logger.info(f"[PREDICTION-NN] Input data shape: {df.shape}")
    logger.info(f"[PREDICTION-NN] Input values (first row): {df.iloc[0].to_dict()}")

    input_dim = len(features)
    output_dim = config.get("output_dim") or config.get("num_classes") or 3
    logger.info(f"[PREDICTION-NN] Model architecture: input={input_dim}, output={output_dim}")

    model = ConfigurableMLP(input_dim, output_dim, layer_config)
    model_path = f"trained_models/{obj.id}.pt"
    encoder_path = f"trained_models/{obj.id}_encoder.joblib"

    logger.info(f"[PREDICTION-NN] Loading model from: {model_path}")
    model_bytes = download_from_minio(model_path)
    model.load_state_dict(
        torch.load(io.BytesIO(model_bytes), map_location=torch.device("cpu"))
    )
    model.eval()

    encoder_bytes = download_from_minio(encoder_path)
    le = joblib.load(io.BytesIO(encoder_bytes))
    logger.info(f"[PREDICTION-NN] Label encoder classes: {le.classes_}")

    # Handle NaN values before creating tensor
    X = df[features].copy()
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        logger.warning(f"[PREDICTION-NN] Found NaN values in columns: {nan_cols}")
        for col in nan_cols:
            col_mean = X[col].mean()
            if pd.isna(col_mean):
                X[col] = X[col].fillna(0)
                logger.info(f"[PREDICTION-NN] Filled NaN in '{col}' with 0")
            else:
                X[col] = X[col].fillna(col_mean)
                logger.info(f"[PREDICTION-NN] Filled NaN in '{col}' with mean: {col_mean}")

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    logger.info(f"[PREDICTION-NN] Input tensor shape: {X_tensor.shape}")

    with torch.no_grad():
        raw_preds = model(X_tensor).argmax(dim=1).numpy()
        preds = le.inverse_transform(raw_preds)

    logger.info(f"[PREDICTION-NN] Raw predictions (indices): {raw_preds}")
    logger.info(f"[PREDICTION-NN] Decoded predictions: {preds}")
    logger.info("=" * 60)

    df["_model_used"] = os.path.basename(model_path)
    df["prediction"] = preds
    return df


def predict_cnn(model_obj, image_source):
    """
    Predict class for an image using a trained CNN model.

    Args:
        model_obj: MLModel instance with training_config
        image_source: Can be a file path (str), bytes, or file-like object (BytesIO, UploadedFile)

    Returns:
        Predicted class name or index
    """
    logger.info("=" * 60)
    logger.info(f"[PREDICTION-CNN] Starting CNN prediction for model: {model_obj.name}")
    logger.info(f"[PREDICTION-CNN] Image source type: {type(image_source).__name__}")

    config = model_obj.training_config or {}
    conv_layers = config.get("conv_layers", [{"out_channels": 32}])
    fc_layers = config.get("fc_layers", [{"units": 128}])
    input_size = config.get("input_size", 64)
    num_classes = config.get("num_classes", 2)
    class_names = config.get("class_names", [])

    logger.info(f"[PREDICTION-CNN] Input size: {input_size}x{input_size}")
    logger.info(f"[PREDICTION-CNN] Num classes: {num_classes}")
    logger.info(f"[PREDICTION-CNN] Class names: {class_names}")

    model = ConfigurableCNN(3, conv_layers, fc_layers, num_classes, input_size)

    model_path = f"trained_models/{model_obj.id}.pt"
    logger.info(f"[PREDICTION-CNN] Loading model from: {model_path}")
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

    # Handle different image source types
    if isinstance(image_source, str):
        # File path
        logger.info(f"[PREDICTION-CNN] Loading image from path: {image_source}")
        image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        # Raw bytes
        logger.info("[PREDICTION-CNN] Loading image from bytes")
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        # File-like object (BytesIO, UploadedFile, etc.)
        logger.info("[PREDICTION-CNN] Loading image from file-like object")
        if hasattr(image_source, 'seek'):
            image_source.seek(0)
        image = Image.open(image_source).convert("RGB")

    logger.info(f"[PREDICTION-CNN] Original image size: {image.size}")
    image_tensor = transform(image).unsqueeze(0)
    logger.info(f"[PREDICTION-CNN] Transformed tensor shape: {image_tensor.shape}")

    with torch.no_grad():
        output = model(image_tensor)
        logger.info(f"[PREDICTION-CNN] Raw output: {output}")
        predicted_idx = output.argmax(dim=1).item()

    logger.info(f"[PREDICTION-CNN] Predicted index: {predicted_idx}")

    # Return class name if available, otherwise return index
    if class_names and predicted_idx < len(class_names):
        result = class_names[predicted_idx]
    else:
        result = predicted_idx

    logger.info(f"[PREDICTION-CNN] Final prediction: {result}")
    logger.info("=" * 60)
    return result
