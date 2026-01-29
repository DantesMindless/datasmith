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
import pandas as pd
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.models.choices import ActivationFunction, ModelType, REGRESSION_MODELS
from core.storage_utils import upload_to_minio
from app.utils.training_logger import TrainingLogger

# Map activation names (strings) to PyTorch activation classes
# Supports both string values ("relu") and enum values (ActivationFunction.RELU)
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "softmax": nn.Softmax,
    # Also support enum keys for backwards compatibility
    ActivationFunction.RELU: nn.ReLU,
    ActivationFunction.TANH: nn.Tanh,
    ActivationFunction.SIGMOID: nn.Sigmoid,
    ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
}


def get_device(use_cuda=True, logger=None):
    """
    Get the best available device for PyTorch training.

    Args:
        use_cuda: Whether to use CUDA if available (default True)
        logger: Optional TrainingLogger for logging device info

    Returns:
        torch.device: The device to use for training
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        if logger:
            logger.log_info(f"\n🎮 GPU Acceleration Enabled!")
            logger.log_info(f"  • Device: {gpu_name}")
            logger.log_info(f"  • Memory: {gpu_memory:.1f} GB")
            logger.log_info(f"  • CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if logger:
            if use_cuda:
                logger.log_info("\n⚠️ CUDA not available, using CPU")
            else:
                logger.log_info("\n💻 Using CPU (GPU disabled)")

    return device


def _is_regression_problem(y):
    """
    Determine if this is a regression or classification problem based on target values.
    Returns True for regression, False for classification.
    """
    # Convert pandas Series to numpy array if needed
    if hasattr(y, 'values'):
        y_vals = y.values
    else:
        y_vals = np.array(y)

    # Handle NaN values
    y_clean = y_vals[~np.isnan(y_vals)] if y_vals.dtype.kind in 'fc' else y_vals

    # Get unique values
    unique_values = np.unique(y_clean)
    num_unique = len(unique_values)

    # Rule 1: If target is boolean type, it's classification
    if y.dtype.kind == 'b':  # boolean
        return False

    # Rule 2: If target is string/object type, it's classification
    if y.dtype.kind in 'OSU':  # object, string, unicode
        return False

    # Rule 3: If exactly 2 unique values that are 0 and 1, it's classification (binary)
    if num_unique == 2 and set(unique_values) == {0, 1}:
        return False

    # Rule 4: If exactly 2 unique values that are 0.0 and 1.0, it's classification (boolean as float)
    if num_unique == 2 and set(unique_values) == {0.0, 1.0}:
        return False

    # Rule 5: Check for large numeric values (likely continuous/prices/measurements)
    if y.dtype.kind in 'fc':  # float or complex
        value_range = np.max(unique_values) - np.min(unique_values)
        max_val = np.max(unique_values)

        # If values are large (>1000) and span a wide range, likely regression
        if max_val > 1000 and value_range > 100:
            return True

        # If values have significant decimal places, likely continuous
        decimal_count = sum(1 for val in unique_values if val != int(val) and not np.isnan(val))
        if decimal_count > 0 and max_val > 10:
            return True

    # Rule 6: If 20 or fewer unique values, lean towards classification
    if num_unique <= 20:
        # Check if all float values are actually small integers (like 1.0, 2.0, 3.0)
        if y.dtype.kind in 'fc':  # float or complex
            all_integers = all(val == int(val) for val in unique_values if not np.isnan(val))
            max_int_val = np.max(unique_values) if len(unique_values) > 0 else 0
            if all_integers and max_int_val <= 100:  # Small integer values
                return False
        # Don't immediately return False here, continue with other checks

    # Rule 7: If target is numeric (float/int) and has many unique values relative to sample size
    if y.dtype.kind in 'fc':  # float or complex
        unique_ratio = len(unique_values) / len(y_clean)
        # If more than 15% of values are unique, treat as regression
        if unique_ratio > 0.15:
            return True

    # Rule 8: Check if target values are continuous-looking decimals (like 2.8, 3.2, 5.5)
    if num_unique > 5 and y.dtype.kind in 'fc':  # Lowered threshold from 20 to 5
        # Check if many values have decimal places
        decimal_count = sum(1 for val in unique_values if val != int(val) and not np.isnan(val))
        decimal_ratio = decimal_count / len(unique_values)
        if decimal_ratio > 0.2:  # 20% have decimals
            return True

    # Rule 9: If target values span a large range with reasonable precision
    if y.dtype.kind in 'fc' and num_unique > 10:  # Lowered from 50
        value_range = np.max(unique_values) - np.min(unique_values)
        if value_range > 50:  # Large range suggests continuous values
            return True

    # Rule 10: Final check - if we have more than 10 unique numeric values, lean towards regression
    if num_unique > 10 and y.dtype.kind in 'fc':
        return True

    return False


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_config, dropout=0.0, use_batch_norm=False, init_method='kaiming'):
        super().__init__()
        layers = []
        dims = [input_dim]
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        for config in layer_config:
            units = config.get("units", 32)
            act = config.get("activation", ActivationFunction.RELU)
            act_class = ACTIVATION_MAP.get(act, nn.ReLU)

            # Linear layer
            linear = nn.Linear(dims[-1], units)
            layers.append(linear)

            # Batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))

            # Activation
            layers.append(act_class())

            # Dropout (after activation)
            layer_dropout = config.get("dropout", dropout)
            if layer_dropout > 0:
                layers.append(nn.Dropout(layer_dropout))

            dims.append(units)

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights(init_method)

    def _init_weights(self, method='kaiming'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif method == 'uniform':
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    """Get the appropriate sklearn/boosting model instance based on type"""
    config = obj.training_config or {}

    # ============ CLASSIFICATION MODELS ============
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(max_iter=obj.max_iter)
    elif model_type == ModelType.DECISION_TREE:
        return DecisionTreeClassifier(
            random_state=obj.random_state,
            max_depth=config.get('max_depth', None)
        )
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 100)
        )
    elif model_type == ModelType.SVM:
        return SVC(max_iter=obj.max_iter)
    elif model_type == ModelType.NAIVE_BAYES:
        return GaussianNB()
    elif model_type == ModelType.KNN:
        return KNeighborsClassifier(n_neighbors=config.get('n_neighbors', 5))
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingClassifier(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3)
        )
    elif model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=obj.random_state,
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 6),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
    elif model_type == ModelType.LIGHTGBM:
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                random_state=obj.random_state,
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', -1),
                verbose=-1
            )
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
    elif model_type == ModelType.ADABOOST:
        return AdaBoostClassifier(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 50)
        )

    # ============ REGRESSION MODELS ============
    elif model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegression()
    elif model_type == ModelType.DECISION_TREE_REGRESSOR:
        return DecisionTreeRegressor(
            random_state=obj.random_state,
            max_depth=config.get('max_depth', None)
        )
    elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
        return RandomForestRegressor(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 100)
        )
    elif model_type == ModelType.SVR:
        gamma = config.get('gamma', 'scale')
        # Convert gamma if it's a numeric string
        if isinstance(gamma, str) and gamma not in ['scale', 'auto']:
            try:
                gamma = float(gamma)
            except ValueError:
                gamma = 'scale'
        return SVR(
            kernel=config.get('kernel', 'rbf'),
            C=config.get('C', 1.0),
            epsilon=config.get('epsilon', 0.1),
            gamma=gamma,
            degree=config.get('degree', 3),
            max_iter=config.get('max_iter', -1)
        )
    elif model_type == ModelType.KNN_REGRESSOR:
        return KNeighborsRegressor(
            n_neighbors=config.get('n_neighbors', 5),
            weights=config.get('weights', 'distance'),
            algorithm=config.get('algorithm', 'auto'),
            leaf_size=config.get('leaf_size', 30)
        )
    elif model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR:
        return GradientBoostingRegressor(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3)
        )
    elif model_type == ModelType.XGBOOST_REGRESSOR:
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                random_state=obj.random_state,
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 6)
            )
        except ImportError:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
    elif model_type == ModelType.LIGHTGBM_REGRESSOR:
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                random_state=obj.random_state,
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', -1),
                verbose=-1
            )
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
    elif model_type == ModelType.ADABOOST_REGRESSOR:
        return AdaBoostRegressor(
            random_state=obj.random_state,
            n_estimators=config.get('n_estimators', 50)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_sklearn_model(obj, X_train, y_train, X_test, y_test, label_encoder=None):
    # Initialize logger
    logger = TrainingLogger(obj)

    # Get model configuration
    config = obj.training_config or {}
    config['model_type'] = obj.model_type
    config['test_size'] = obj.test_size
    config['random_state'] = obj.random_state

    try:
        # Start training
        logger.start_training(config)

        # Log data information
        unique_classes = len(np.unique(y_train))
        logger.log_data_loading({
            'Total Training Samples': len(X_train),
            'Total Test Samples': len(X_test),
            'Number of Features': X_train.shape[1],
            'Number of Classes': unique_classes,
            'Class Distribution (Train)': dict(zip(*np.unique(y_train, return_counts=True)))
        })

        logger.log_data_split(len(X_train), len(X_test))

        # Log preprocessing
        preprocessing_steps = [
            f"Feature scaling/normalization (if applicable)",
            f"Encoding categorical variables",
            f"Handling missing values"
        ]
        logger.log_preprocessing(preprocessing_steps)

        # Get and log model instance
        logger.log_info("\n🔨 Initializing model...")
        clf = get_model_instance(obj.model_type, obj)

        # Feature scaling for models that require it (SVR, KNN, SVM)
        scaler = None
        models_requiring_scaling = [ModelType.SVR, ModelType.KNN_REGRESSOR, ModelType.SVM, ModelType.KNN]
        if obj.model_type in models_requiring_scaling:
            logger.log_info("📊 Applying StandardScaler (required for this model type)...")
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Log model architecture/parameters
        model_params = clf.get_params()
        architecture = {
            'Algorithm': obj.model_type,
            'Hyperparameters': {k: str(v) for k, v in model_params.items() if v is not None}
        }
        if scaler:
            architecture['Preprocessing'] = 'StandardScaler applied'
        logger.log_model_architecture(architecture)

        # Training
        logger.log_training_start(total_epochs=1)  # sklearn is single-phase
        logger.log_info("⏳ Fitting model to training data...")

        train_start = time.time()

        # Fit with progress tracking for ensemble methods
        if hasattr(clf, 'n_estimators'):
            n_estimators = getattr(clf, 'n_estimators', 100)
            logger.log_info(f"Training {n_estimators} estimators...")

        clf.fit(X_train, y_train)
        train_time = time.time() - train_start

        logger.log_info(f"✓ Model fitting completed in {logger._format_duration(train_time)}")

        # Determine if this is a regression model
        is_regression = obj.model_type in REGRESSION_MODELS

        # Predictions on training set
        logger.log_info("\n📊 Evaluating on training set...")
        train_pred = clf.predict(X_train)

        if is_regression:
            # Regression metrics
            train_mse = mean_squared_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            logger.log_info(f"  Training MSE: {train_mse:.4f}")
            logger.log_info(f"  Training R² Score: {train_r2:.4f}")
        else:
            train_acc = accuracy_score(y_train, train_pred)
            logger.log_info(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Predictions on test set
        logger.log_validation_start()
        y_pred = clf.predict(X_test)

        if is_regression:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            acc = r2  # Use R² as the "accuracy" metric for regression

            eval_metrics = {
                'MSE': f"{mse:.4f}",
                'RMSE': f"{rmse:.4f}",
                'MAE': f"{mae:.4f}",
                'R² Score': f"{r2:.4f}",
                'Training Time': logger._format_duration(train_time)
            }
            logger.log_evaluation_results(eval_metrics)

            # No per-class metrics for regression
            class_report = None
            conf_matrix = None
        else:
            acc = accuracy_score(y_test, y_pred)

            # Detailed evaluation metrics
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log evaluation results
            eval_metrics = {
                'Test Accuracy': f"{acc:.4f} ({acc*100:.2f}%)",
                'Training Accuracy': f"{train_acc:.4f} ({train_acc*100:.2f}%)",
                'Precision (weighted)': f"{class_report['weighted avg']['precision']:.4f}",
                'Recall (weighted)': f"{class_report['weighted avg']['recall']:.4f}",
                'F1-Score (weighted)': f"{class_report['weighted avg']['f1-score']:.4f}",
                'Training Time': logger._format_duration(train_time)
            }

            # Add per-class metrics
            logger.log_evaluation_results(eval_metrics)

            logger.log_info("\n📋 Per-Class Performance:")
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    logger.log_info(f"  Class '{class_name}':")
                    logger.log_info(f"    Precision: {metrics['precision']:.4f}")
                    logger.log_info(f"    Recall: {metrics['recall']:.4f}")
                    logger.log_info(f"    F1-Score: {metrics['f1-score']:.4f}")
                    logger.log_info(f"    Support: {int(metrics['support'])} samples")

        # Feature importance (if available)
        if hasattr(clf, 'feature_importances_'):
            logger.log_info("\n🎯 Top 10 Most Important Features:")
            feature_names = X_train.columns.tolist()
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            for i, idx in enumerate(indices, 1):
                logger.log_info(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

        # Save model
        logger.log_info("\n💾 Saving model...")
        buffer = io.BytesIO()
        joblib.dump(clf, buffer)
        buffer.seek(0)

        model_path = f"trained_models/{obj.id}.joblib"
        model_bytes = buffer.read()
        upload_to_minio(model_bytes, model_path, content_type="application/octet-stream")

        logger.log_model_save(model_path, len(model_bytes))

        # Save scaler if used
        if scaler is not None:
            scaler_buffer = io.BytesIO()
            joblib.dump(scaler, scaler_buffer)
            scaler_buffer.seek(0)
            scaler_path = f"trained_models/{obj.id}_scaler.joblib"
            upload_to_minio(scaler_buffer.read(), scaler_path, content_type="application/octet-stream")
            logger.log_info(f"  • Scaler saved: {scaler_path}")

        # Save prediction schema
        feature_names = X_train.columns.tolist()
        obj.prediction_schema = {
            "input_features": feature_names,
            "feature_count": len(feature_names),
            "target_column": obj.target_column,
            "example": {col: "numeric_value" for col in feature_names},
            "description": f"Provide values for these {len(feature_names)} features to make a prediction",
            "model_type": obj.model_type,
            "num_classes": unique_classes,
            "requires_scaling": scaler is not None
        }

        # Save detailed metrics for analytics
        y_pred = clf.predict(X_test)

        # Feature importance if available
        feature_importance_data = []
        if hasattr(clf, 'feature_importances_'):
            for fname, importance in zip(feature_names, clf.feature_importances_):
                feature_importance_data.append({
                    "feature": fname,
                    "importance": float(importance)
                })
            feature_importance_data.sort(key=lambda x: x["importance"], reverse=True)
        elif hasattr(clf, 'coef_'):
            # For linear models, use coefficients as feature importance
            coefs = clf.coef_.flatten() if len(clf.coef_.shape) > 1 else clf.coef_
            for fname, coef in zip(feature_names, coefs):
                feature_importance_data.append({
                    "feature": fname,
                    "importance": float(abs(coef))
                })
            feature_importance_data.sort(key=lambda x: x["importance"], reverse=True)

        if is_regression:
            # Regression analytics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate target statistics for the combined train+test target values
            y_all = np.concatenate([y_train.values if hasattr(y_train, 'values') else y_train,
                                    y_test.values if hasattr(y_test, 'values') else y_test])
            target_stats = {
                "mean": float(np.mean(y_all)),
                "std": float(np.std(y_all)),
                "min": float(np.min(y_all)),
                "max": float(np.max(y_all))
            }

            # Log target statistics
            logger.log_info("\n📊 Target Variable Statistics:")
            logger.log_info(f"  Mean: {target_stats['mean']:.4f}")
            logger.log_info(f"  Std Dev: {target_stats['std']:.4f}")
            logger.log_info(f"  Min: {target_stats['min']:.4f}")
            logger.log_info(f"  Max: {target_stats['max']:.4f}")

            obj.analytics_data = {
                "r2_score": float(r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "feature_importance": feature_importance_data,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "is_regression": True,
                "target_stats": target_stats
            }
        else:
            # Classification analytics
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()

            # Get class distribution from predictions
            y_pred_all = clf.predict(X_test)
            unique_labels, counts = np.unique(y_pred_all, return_counts=True)
            prediction_distribution = []
            for label, count in zip(unique_labels, counts):
                if label_encoder is not None:
                    try:
                        decoded_label = label_encoder.inverse_transform([int(label)])[0]
                    except:
                        decoded_label = str(label)
                else:
                    decoded_label = str(label)
                prediction_distribution.append({"label": decoded_label, "value": int(count)})

            obj.analytics_data = {
                "accuracy": float(acc),
                "precision": float(class_report['weighted avg']['precision']),
                "recall": float(class_report['weighted avg']['recall']),
                "f1_score": float(class_report['weighted avg']['f1-score']),
                "confusion_matrix": conf_matrix,
                "feature_importance": feature_importance_data,
                "prediction_distribution": prediction_distribution,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "class_report": class_report
            }

        obj.save(update_fields=["prediction_schema", "analytics_data"])

        # Training complete
        logger.log_training_complete(acc)

        return model_path, acc

    except Exception as e:
        logger.log_error(e, "during sklearn model training")
        raise


def train_nn(obj, X_train, y_train, X_test, y_test):
    # Initialize logger
    logger = TrainingLogger(obj)

    config = obj.training_config or {}

    # Architecture config
    layer_config = config.get(
        "layer_config", [{"units": 128, "activation": "relu"}, {"units": 64, "activation": "relu"}]
    )
    dropout = config.get("dropout", 0.2)
    use_batch_norm = config.get("use_batch_norm", True)
    init_method = config.get("init_method", "kaiming")

    # Training config
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 0.001)
    use_cuda = config.get("use_cuda", True)

    # Optimizer config
    optimizer_name = config.get("optimizer", "adam")
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0001)

    # Learning rate scheduler config
    lr_scheduler_name = config.get("lr_scheduler", "plateau")
    lr_patience = config.get("lr_patience", 5)
    lr_factor = config.get("lr_factor", 0.5)
    min_lr = config.get("min_lr", 1e-6)

    # Early stopping config
    early_stopping = config.get("early_stopping", True)
    early_stopping_patience = config.get("early_stopping_patience", 10)
    early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0001)

    # Loss function config
    loss_function = config.get("loss_function", "cross_entropy")
    label_smoothing = config.get("label_smoothing", 0.0)
    focal_gamma = config.get("focal_gamma", 2.0)
    use_class_weights = config.get("class_weights", True)

    # Advanced config
    gradient_clipping = config.get("gradient_clipping", 1.0)
    warmup_epochs = config.get("warmup_epochs", 0)
    normalize_features = config.get("normalize_features", "standard")

    try:
        # Get device (GPU if available)
        device = get_device(use_cuda, logger)

        # Start training
        training_config = {
            'model_type': 'Neural Network (PyTorch)',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': optimizer_name,
            'loss_function': loss_function,
            'dropout': dropout,
            'batch_norm': use_batch_norm,
            'lr_scheduler': lr_scheduler_name,
            'early_stopping': early_stopping,
            'weight_decay': weight_decay,
            'device': str(device)
        }
        logger.start_training(training_config)

        # Encode labels
        logger.log_info("\n🔧 Encoding target labels...")
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train_encoded))

        # Log data information
        logger.log_data_loading({
            'Input Features': input_dim,
            'Output Classes': output_dim,
            'Class Names': le.classes_.tolist(),
            'Training Samples': len(X_train),
            'Test Samples': len(X_test),
            'Class Distribution': dict(zip(*np.unique(y_train_encoded, return_counts=True)))
        })

        logger.log_data_split(len(X_train), len(X_test))

        # Log preprocessing
        preprocessing_steps = [
            "Label encoding for target variable",
            "Convert data to PyTorch tensors",
            f"Create DataLoader with batch size {batch_size}"
        ]
        logger.log_preprocessing(preprocessing_steps)

        # Build model
        logger.log_info("\n🏗️ Building neural network...")
        model = ConfigurableMLP(
            input_dim, output_dim, layer_config,
            dropout=dropout, use_batch_norm=use_batch_norm, init_method=init_method
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log architecture
        architecture = {
            'Input Dimension': input_dim,
            'Output Dimension': output_dim,
            'Hidden Layers': len(layer_config),
            'Layer Configuration': [
                f"Layer {i+1}: {cfg.get('units')} units, {cfg.get('activation', 'relu')} activation"
                for i, cfg in enumerate(layer_config)
            ],
            'Dropout': dropout,
            'Batch Normalization': use_batch_norm,
            'Weight Initialization': init_method,
            'Total Parameters': f"{total_params:,}",
            'Trainable Parameters': f"{trainable_params:,}"
        }
        logger.log_model_architecture(architecture)

        # Ensure all feature data is numeric
        logger.log_info("\n🔄 Converting features to numeric...")
        non_numeric_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_cols:
            logger.log_info(f"  ⚠️ Found non-numeric columns that need conversion: {non_numeric_cols}")
        X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        X_test_numeric = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

        # Feature normalization
        if normalize_features != "none":
            logger.log_info(f"  📊 Applying {normalize_features} normalization...")
            if normalize_features == "standard":
                scaler = StandardScaler()
            elif normalize_features == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif normalize_features == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            X_train_numeric = pd.DataFrame(
                scaler.fit_transform(X_train_numeric),
                columns=X_train_numeric.columns,
                index=X_train_numeric.index
            )
            X_test_numeric = pd.DataFrame(
                scaler.transform(X_test_numeric),
                columns=X_test_numeric.columns,
                index=X_test_numeric.index
            )
        logger.log_info(f"  ✓ Features prepared for training")

        # Setup optimizer
        logger.log_info(f"\n⚙️ Setting up {optimizer_name} optimizer...")
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "nadam":
            optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Setup loss function with class weights
        class_weights_tensor = None
        if use_class_weights:
            class_counts = np.bincount(y_train_encoded)
            class_weights_np = len(y_train_encoded) / (len(class_counts) * class_counts)
            class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
            logger.log_info(f"  📊 Using class weights: {dict(zip(le.classes_, class_weights_np.round(2)))}")

        if loss_function == "cross_entropy":
            if label_smoothing > 0:
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        elif loss_function == "focal":
            # Focal Loss implementation
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, weight=None):
                    super().__init__()
                    self.gamma = gamma
                    self.weight = weight
                    self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

                def forward(self, inputs, targets):
                    ce_loss = self.ce(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                    return focal_loss.mean()

            criterion = FocalLoss(gamma=focal_gamma, weight=class_weights_tensor)
            logger.log_info(f"  📊 Using Focal Loss with gamma={focal_gamma}")
        elif loss_function == "nll":
            criterion = nn.NLLLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Setup learning rate scheduler
        scheduler = None
        if lr_scheduler_name != "none":
            logger.log_info(f"  📉 Setting up {lr_scheduler_name} LR scheduler...")
            if lr_scheduler_name == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr
                )
            elif lr_scheduler_name == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
            elif lr_scheduler_name == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_patience, gamma=lr_factor)
            elif lr_scheduler_name == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            elif lr_scheduler_name == "one_cycle":
                total_steps = epochs * (len(X_train) // batch_size + 1)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr * 10, total_steps=total_steps
                )

        # Create data loaders
        train_dataset = data.TensorDataset(
            torch.tensor(X_train_numeric.values, dtype=torch.float32),
            torch.tensor(y_train_encoded, dtype=torch.long),
        )
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = data.TensorDataset(
            torch.tensor(X_test_numeric.values, dtype=torch.float32),
            torch.tensor(y_test_encoded, dtype=torch.long)
        )
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        logger.log_training_start(epochs)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Warmup learning rate
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            logger.log_epoch_start(epoch, epochs)

            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            num_batches = len(train_loader)

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                optimizer.step()

                # Step scheduler for one_cycle (per batch)
                if scheduler and lr_scheduler_name == "one_cycle":
                    scheduler.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

                # Log batch progress
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches:
                    logger.log_batch_progress(
                        batch_idx, num_batches, loss.item(),
                        {'accuracy': train_correct / train_total}
                    )

            epoch_loss = running_loss / num_batches
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            # Log epoch end
            logger.log_epoch_end(
                epoch, epochs, epoch_loss, train_acc,
                val_loss, val_acc, current_lr, epoch_time
            )

            # Step scheduler (per epoch, except one_cycle)
            if scheduler and lr_scheduler_name != "one_cycle":
                if lr_scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping check
            if early_stopping and epochs_no_improve >= early_stopping_patience:
                logger.log_info(f"\n⏹️ Early stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
                break

        # Restore best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.log_info(f"  ✓ Restored best model from epoch {best_epoch}")

        # Final evaluation
        logger.log_validation_start()
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_numeric.values, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test_encoded, preds)

            # Detailed metrics
            class_report = classification_report(y_test_encoded, preds,
                                                target_names=[str(c) for c in le.classes_],
                                                output_dict=True, zero_division=0)

        eval_metrics = {
            'Final Test Accuracy': f"{acc:.4f} ({acc*100:.2f}%)",
            'Best Validation Accuracy': f"{best_val_acc:.4f} ({best_val_acc*100:.2f}%)",
            'Best Epoch': best_epoch,
            'Precision (weighted)': f"{class_report['weighted avg']['precision']:.4f}",
            'Recall (weighted)': f"{class_report['weighted avg']['recall']:.4f}",
            'F1-Score (weighted)': f"{class_report['weighted avg']['f1-score']:.4f}"
        }
        logger.log_evaluation_results(eval_metrics)

        # Save model
        logger.log_info("\n💾 Saving model and encoder...")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_path = f"trained_models/{obj.id}.pt"
        model_bytes = buffer.read()

        upload_to_minio(model_bytes, model_path, content_type="application/octet-stream")
        logger.log_model_save(model_path, len(model_bytes))

        enc_buf = io.BytesIO()
        joblib.dump(le, enc_buf)
        enc_buf.seek(0)
        encoder_path = f"trained_models/{obj.id}_encoder.joblib"

        upload_to_minio(
            enc_buf.read(), encoder_path, content_type="application/octet-stream"
        )
        logger.log_info(f"  • Label encoder saved: {encoder_path}")

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

        # Save detailed metrics for analytics
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_numeric.values, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()

        class_report = classification_report(y_test_encoded, preds,
                                            target_names=[str(c) for c in le.classes_],
                                            output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test_encoded, preds).tolist()

        # Get class distribution from predictions
        unique_labels, counts = np.unique(preds, return_counts=True)
        prediction_distribution = [
            {"label": str(le.classes_[label]), "value": int(count)}
            for label, count in zip(unique_labels, counts)
        ]

        # Training history from logger metrics
        training_history = []
        for i, epoch in enumerate(range(1, epochs + 1)):
            history_entry = {"epoch": epoch}
            if i < len(logger.metrics['losses']):
                history_entry['loss'] = float(logger.metrics['losses'][i])
            if i < len(logger.metrics['accuracies']):
                history_entry['accuracy'] = float(logger.metrics['accuracies'][i])
            if i < len(logger.metrics['val_accuracies']):
                history_entry['val_accuracy'] = float(logger.metrics['val_accuracies'][i])
            training_history.append(history_entry)

        obj.analytics_data = {
            "accuracy": float(acc),
            "precision": float(class_report['weighted avg']['precision']),
            "recall": float(class_report['weighted avg']['recall']),
            "f1_score": float(class_report['weighted avg']['f1-score']),
            "confusion_matrix": conf_matrix,
            "feature_importance": [],  # Neural networks don't have direct feature importance
            "prediction_distribution": prediction_distribution,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_history": training_history,
            "class_report": class_report
        }

        obj.save(update_fields=["training_config", "prediction_schema", "analytics_data"])

        logger.log_training_complete(acc)

        return model_path, acc

    except Exception as e:
        logger.log_error(e, "during neural network training")
        raise


def train_cnn(obj):
    # Initialize logger
    logger = TrainingLogger(obj)

    image_path = None
    temp_dir_to_cleanup = None

    # Get image path - prioritize MinIO storage, fall back to legacy paths
    if obj.dataset.minio_images_prefix:
        # Download images from MinIO to temp directory
        logger.log_info("📥 Downloading images from MinIO storage...")
        image_path = obj.dataset.get_image_directory()
        temp_dir_to_cleanup = image_path  # Mark for cleanup after training
        logger.log_info(f"  • Images downloaded to: {image_path}")
    elif hasattr(obj.dataset, "extracted_path") and obj.dataset.extracted_path:
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
    use_cuda = config.get("use_cuda", True)  # Enable CUDA by default

    try:
        # Get device (GPU if available)
        device = get_device(use_cuda, logger)

        # Start training
        training_config = {
            'model_type': 'Convolutional Neural Network (CNN)',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'input_size': f'{input_size}x{input_size}',
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'device': str(device)
        }
        logger.start_training(training_config)

        # Load dataset
        logger.log_info(f"\n📂 Loading image dataset from: {image_path}")
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(image_path, transform=transform)
        num_classes = len(dataset.classes)

        logger.log_data_loading({
            'Image Directory': image_path,
            'Total Images': len(dataset),
            'Number of Classes': num_classes,
            'Class Names': list(dataset.class_to_idx.keys())
        })

        # Split dataset for training and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

        logger.log_data_split(train_size, val_size)

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Log preprocessing
        preprocessing_steps = [
            f"Resize images to {input_size}x{input_size}",
            "Convert to tensor (normalize to [0, 1])",
            f"Create batches of size {batch_size}"
        ]
        logger.log_preprocessing(preprocessing_steps)

        # Build model
        logger.log_info("\n🏗️ Building CNN model...")
        model = ConfigurableCNN(3, conv_layers, fc_layers, num_classes, input_size)
        model = model.to(device)  # Move model to GPU

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log architecture
        architecture = {
            'Input Channels': 3,
            'Input Size': f'{input_size}x{input_size}',
            'Output Classes': num_classes,
            'Convolutional Layers': len(conv_layers),
            'Conv Layer Details': [f"Layer {i+1}: {cfg.get('out_channels', 16)} filters" for i, cfg in enumerate(conv_layers)],
            'Fully Connected Layers': len(fc_layers),
            'FC Layer Details': [f"Layer {i+1}: {cfg.get('units', 64)} units" for i, cfg in enumerate(fc_layers)],
            'Total Parameters': f"{total_params:,}",
            'Trainable Parameters': f"{trainable_params:,}"
        }
        logger.log_model_architecture(architecture)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        logger.log_training_start(epochs)
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            logger.log_epoch_start(epoch, epochs)

            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            num_batches = len(train_loader)

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
                # Move data to device (GPU)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

                # Log batch progress
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches:
                    logger.log_batch_progress(
                        batch_idx, num_batches, loss.item(),
                        {'accuracy': train_correct / train_total}
                    )

            epoch_loss = running_loss / num_batches
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    # Move data to device (GPU)
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            epoch_time = time.time() - epoch_start

            logger.log_epoch_end(
                epoch, epochs, epoch_loss, train_acc,
                val_loss, val_acc, lr, epoch_time
            )

        # Calculate final accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move data to device (GPU)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0.0

        eval_metrics = {
            'Final Validation Accuracy': f"{accuracy:.4f} ({accuracy*100:.2f}%)",
            'Best Validation Accuracy': f"{best_val_acc:.4f} ({best_val_acc*100:.2f}%)",
            'Best Epoch': best_epoch
        }
        logger.log_evaluation_results(eval_metrics)

        # Save model
        logger.log_info("\n💾 Saving CNN model...")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_path = f"trained_models/{obj.id}.pt"
        model_bytes = buffer.read()
        upload_to_minio(model_bytes, model_path, content_type="application/octet-stream")

        logger.log_model_save(model_path, len(model_bytes))

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

        # Calculate confusion matrix and detailed metrics on validation set
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move data to device (GPU)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_report = classification_report(all_labels, all_preds,
                                            target_names=class_names,
                                            output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

        # Get class distribution from predictions
        unique_labels, counts = np.unique(all_preds, return_counts=True)
        prediction_distribution = [
            {"label": class_names[int(label)], "value": int(count)}
            for label, count in zip(unique_labels, counts)
        ]

        # Training history from logger metrics
        training_history = []
        for i, epoch in enumerate(range(1, epochs + 1)):
            history_entry = {"epoch": epoch}
            if i < len(logger.metrics['losses']):
                history_entry['loss'] = float(logger.metrics['losses'][i])
            if i < len(logger.metrics['accuracies']):
                history_entry['accuracy'] = float(logger.metrics['accuracies'][i])
            if i < len(logger.metrics['val_accuracies']):
                history_entry['val_accuracy'] = float(logger.metrics['val_accuracies'][i])
            training_history.append(history_entry)

        obj.analytics_data = {
            "accuracy": float(accuracy),
            "precision": float(class_report['weighted avg']['precision']),
            "recall": float(class_report['weighted avg']['recall']),
            "f1_score": float(class_report['weighted avg']['f1-score']),
            "confusion_matrix": conf_matrix,
            "feature_importance": [],  # CNNs don't have traditional feature importance
            "prediction_distribution": prediction_distribution,
            "training_samples": train_size,
            "test_samples": val_size,
            "training_history": training_history,
            "class_report": class_report
        }

        obj.save(update_fields=["training_config", "prediction_schema", "analytics_data"])

        logger.log_training_complete(accuracy)

        return model_path, accuracy

    except Exception as e:
        logger.log_error(e, "during CNN training")
        raise

    finally:
        # Clean up temp directory if we downloaded from MinIO
        if temp_dir_to_cleanup and os.path.isdir(temp_dir_to_cleanup):
            import shutil
            try:
                shutil.rmtree(temp_dir_to_cleanup)
            except Exception:
                pass  # Ignore cleanup errors


