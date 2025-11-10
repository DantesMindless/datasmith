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
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from app.models.choices import ActivationFunction, ModelType
from core.storage_utils import upload_to_minio
from app.utils.training_logger import TrainingLogger

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
    """Get the appropriate sklearn/boosting model instance based on type"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    # Traditional ML models
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(max_iter=obj.max_iter)
    elif model_type == ModelType.DECISION_TREE:
        return DecisionTreeClassifier(
            random_state=obj.random_state,
            max_depth=obj.training_config.get('max_depth', None)
        )
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            random_state=obj.random_state,
            n_estimators=obj.training_config.get('n_estimators', 100)
        )
    elif model_type == ModelType.SVM:
        return SVC(max_iter=obj.max_iter)
    elif model_type == ModelType.NAIVE_BAYES:
        return GaussianNB()
    elif model_type == ModelType.KNN:
        n_neighbors = obj.training_config.get('n_neighbors', 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    # Ensemble models
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingClassifier(
            random_state=obj.random_state,
            n_estimators=obj.training_config.get('n_estimators', 100)
        )
    elif model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=obj.random_state,
                n_estimators=obj.training_config.get('n_estimators', 100),
                learning_rate=obj.training_config.get('learning_rate', 0.1),
                max_depth=obj.training_config.get('max_depth', 6),
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
                n_estimators=obj.training_config.get('n_estimators', 100),
                learning_rate=obj.training_config.get('learning_rate', 0.1),
                max_depth=obj.training_config.get('max_depth', -1),
                verbose=-1
            )
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
    elif model_type == ModelType.ADABOOST:
        return AdaBoostClassifier(
            random_state=obj.random_state,
            n_estimators=obj.training_config.get('n_estimators', 50)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_sklearn_model(obj, X_train, y_train, X_test, y_test):
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

        # Log model architecture/parameters
        model_params = clf.get_params()
        architecture = {
            'Algorithm': obj.model_type,
            'Hyperparameters': {k: str(v) for k, v in model_params.items() if v is not None}
        }
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

        # Predictions on training set
        logger.log_info("\n📊 Evaluating on training set...")
        train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.log_info(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Predictions on test set
        logger.log_validation_start()
        y_pred = clf.predict(X_test)
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

        # Save prediction schema
        feature_names = X_train.columns.tolist()
        obj.prediction_schema = {
            "input_features": feature_names,
            "feature_count": len(feature_names),
            "target_column": obj.target_column,
            "example": {col: "numeric_value" for col in feature_names},
            "description": f"Provide values for these {len(feature_names)} features to make a prediction",
            "model_type": obj.model_type,
            "num_classes": unique_classes
        }

        # Save detailed metrics for analytics
        y_pred = clf.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance if available
        feature_importance_data = []
        if hasattr(clf, 'feature_importances_'):
            for fname, importance in zip(feature_names, clf.feature_importances_):
                feature_importance_data.append({
                    "feature": fname,
                    "importance": float(importance)
                })
            feature_importance_data.sort(key=lambda x: x["importance"], reverse=True)

        # Get class distribution from predictions
        y_pred_all = clf.predict(X_test)
        unique_labels, counts = np.unique(y_pred_all, return_counts=True)
        prediction_distribution = [
            {"label": str(label), "value": int(count)}
            for label, count in zip(unique_labels, counts)
        ]

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
    layer_config = config.get(
        "layer_config", [{"units": 32, "activation": ActivationFunction.RELU}]
    )
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)

    try:
        # Start training
        training_config = {
            'model_type': 'Neural Network (PyTorch)',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
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
        model = ConfigurableMLP(input_dim, output_dim, layer_config)

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
            'Total Parameters': f"{total_params:,}",
            'Trainable Parameters': f"{trainable_params:,}"
        }
        logger.log_model_architecture(architecture)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train_encoded, dtype=torch.long),
        )
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation dataset
        val_dataset = data.TensorDataset(
            torch.tensor(X_test.values, dtype=torch.float32),
            torch.tensor(y_test_encoded, dtype=torch.long)
        )
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
                best_epoch = epoch

            epoch_time = time.time() - epoch_start

            # Log epoch end
            logger.log_epoch_end(
                epoch, epochs, epoch_loss, train_acc,
                val_loss, val_acc, lr, epoch_time
            )

        # Final evaluation
        logger.log_validation_start()
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            preds = model(X_test_tensor).argmax(dim=1).numpy()
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
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            preds = model(X_test_tensor).argmax(dim=1).numpy()

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

    try:
        # Start training
        training_config = {
            'model_type': 'Convolutional Neural Network (CNN)',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'input_size': f'{input_size}x{input_size}',
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
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
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(batch_y.numpy())

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


def train_transfer_learning(obj):
    """
    Train a transfer learning model (ResNet, VGG, EfficientNet)
    """
    import torchvision.models as models

    image_path = None

    if hasattr(obj.dataset, "extracted_path") and obj.dataset.extracted_path:
        image_path = obj.dataset.extracted_path
    elif hasattr(obj.dataset, "image_folder") and obj.dataset.image_folder:
        image_path = obj.dataset.image_folder.path

    if not image_path or not os.path.isdir(image_path):
        raise ValueError(f"Image path is invalid or missing: {image_path}")

    config = obj.training_config or {}
    input_size = config.get("input_size", 224)  # Standard ImageNet size
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)
    freeze_features = config.get("freeze_features", True)

    # Data transforms for transfer learning
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    dataset = datasets.ImageFolder(image_path, transform=transform)
    num_classes = len(dataset.classes)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained model based on type
    if obj.model_type == ModelType.RESNET:
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif obj.model_type == ModelType.VGG:
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif obj.model_type == ModelType.EFFICIENTNET:
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported transfer learning model: {obj.model_type}")

    # Freeze feature extraction layers if specified
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        if obj.model_type == ModelType.RESNET:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif obj.model_type == ModelType.VGG:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif obj.model_type == ModelType.EFFICIENTNET:
            for param in model.classifier.parameters():
                param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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

    # Validation
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

    # Save model
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_path = f"trained_models/{obj.id}.pt"
    upload_to_minio(buffer.read(), model_path, content_type="application/octet-stream")

    obj.training_config["num_classes"] = num_classes
    obj.training_config["class_names"] = list(dataset.class_to_idx.keys())
    obj.training_config["input_size"] = input_size

    # Save prediction schema
    class_names = list(dataset.class_to_idx.keys())
    obj.prediction_schema = {
        "input_type": "image",
        "input_size": input_size,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif"],
        "output_classes": class_names,
        "num_classes": num_classes,
        "model_architecture": obj.model_type,
        "pretrained": True,
        "example": "Upload an image file",
        "description": f"Upload an image to classify into one of {num_classes} categories using {obj.model_type} transfer learning"
    }

    obj.save(update_fields=["training_config", "prediction_schema"])

    return model_path, accuracy


def train_rnn(obj):
    """
    Train RNN/LSTM/GRU model for time series or sequential data
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(obj.dataset.csv_file.path)
    target = obj.target_column
    config = obj.training_config or {}

    # Get sequence length and features
    sequence_length = config.get("sequence_length", 10)
    features = config.get("features") or [col for col in df.columns if col != target]

    # Prepare sequential data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(df[target].iloc[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split data
    split_idx = int(len(X) * (1 - obj.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Encode labels if classification
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Model parameters
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 0.001)

    input_size = len(features)
    output_size = len(np.unique(y_train_encoded))

    # Build RNN model
    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, model_type):
            super().__init__()
            if model_type == ModelType.LSTM:
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            elif model_type == ModelType.GRU:
                self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            else:  # RNN
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])  # Take last timestep
            return out

    model = RNNModel(input_size, hidden_size, num_layers, output_size, obj.model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors
    train_dataset = data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.long)
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test_encoded, preds)

    # Save model
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_path = f"trained_models/{obj.id}.pt"
    upload_to_minio(buffer.read(), model_path, content_type="application/octet-stream")

    # Save scaler and encoder
    import joblib
    scaler_buf = io.BytesIO()
    joblib.dump(scaler, scaler_buf)
    scaler_buf.seek(0)
    scaler_path = f"trained_models/{obj.id}_scaler.joblib"
    upload_to_minio(scaler_buf.read(), scaler_path, content_type="application/octet-stream")

    enc_buf = io.BytesIO()
    joblib.dump(le, enc_buf)
    enc_buf.seek(0)
    encoder_path = f"trained_models/{obj.id}_encoder.joblib"
    upload_to_minio(enc_buf.read(), encoder_path, content_type="application/octet-stream")

    obj.training_config["output_dim"] = output_size
    obj.training_config["class_names"] = le.classes_.tolist()
    obj.training_config["hidden_size"] = hidden_size
    obj.training_config["num_layers"] = num_layers
    obj.training_config["sequence_length"] = sequence_length

    # Save prediction schema
    obj.prediction_schema = {
        "input_type": "sequential",
        "sequence_length": sequence_length,
        "input_features": features,
        "feature_count": len(features),
        "output_classes": le.classes_.tolist(),
        "model_architecture": obj.model_type,
        "example": f"Provide {sequence_length} timesteps of data with features: {', '.join(features)}",
        "description": f"Sequential prediction using {obj.model_type} model"
    }

    obj.save(update_fields=["training_config", "prediction_schema"])

    return model_path, acc
