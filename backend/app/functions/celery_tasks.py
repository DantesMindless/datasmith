from celery import shared_task
from app.models import MLModel, Dataset
from app.functions.training import (
    train_cnn, train_nn, train_sklearn_model,
    _is_regression_problem
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.models.choices import ModelStatus, ModelType
from app.models.main import TrainingRun
import os
import zipfile
import shutil
from pathlib import Path
from django.conf import settings


def encode_categorical_features(df, features, logger=None):
    """
    Encode categorical features (object/string columns) to numeric values.
    Returns the encoded dataframe and a dict of encoders for each column.
    """
    df_encoded = df[features].copy()
    encoders = {}

    for col in features:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            if logger:
                logger.info(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


@shared_task
def train_sklearn_task(model_id):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting sklearn training task for model {model_id}")

    acc = None  # Initialize accuracy variable

    try:
        obj = MLModel.objects.get(id=model_id)
    except MLModel.DoesNotExist:
        logger.error(f"Model {model_id} does not exist. It may have been deleted.")
        return 0.0

    try:
        # Validate dataset has CSV file
        if not obj.dataset.minio_csv_key:
            error_msg = "This model requires a tabular dataset with a CSV file. The selected dataset appears to be an image dataset."
            logger.error(f"{error_msg} Model: {model_id}, Dataset: {obj.dataset.id}")
            obj.status = ModelStatus.FAILED
            obj.training_log = error_msg
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=error_msg)
            raise ValueError(error_msg)

        csv_obj = obj.dataset.get_csv_file_object()
        df = pd.read_csv(csv_obj)
        target = obj.target_column
        config = obj.training_config or {}

        # Get features, respecting excluded_columns
        excluded_columns = config.get("excluded_columns", [])
        if config.get("features"):
            # Use explicitly specified features
            features = config.get("features")
        else:
            # Auto-select all columns except target and excluded
            features = [col for col in df.columns if col != target and col not in excluded_columns]
            logger.info(f"Auto-selected {len(features)} features (excluded {len(excluded_columns)} columns)")

        # Encode categorical features (e.g., 'M'/'F' -> 0/1)
        X, feature_encoders = encode_categorical_features(df, features, logger)
        if feature_encoders:
            logger.info(f"Encoded {len(feature_encoders)} categorical columns: {list(feature_encoders.keys())}")

        y = df[target]

        # Determine if this is a regression or classification problem
        is_regression = _is_regression_problem(y)
        logger.info(f"Problem type detected: {'Regression' if is_regression else 'Classification'}")

        # Store original class names for prediction_distribution display (classification only)
        # This ensures labels are readable (e.g., "Class A" instead of "0")
        original_classes = sorted(y.unique().tolist())
        if not is_regression:
            obj.training_config['class_names'] = [str(c) for c in original_classes]
        else:
            # For regression, don't store class_names (there are no classes)
            obj.training_config.pop('class_names', None)
        obj.save(update_fields=['training_config'])
        logger.info(f"=== TARGET ANALYSIS ===")
        logger.info(f"Target column: {target}")
        logger.info(f"Target dtype: {y.dtype}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Original unique values ({len(original_classes)}): {original_classes[:20] if len(original_classes) > 20 else original_classes}{'...' if len(original_classes) > 20 else ''}")
        if not is_regression:
            logger.info(f"Target classes: {obj.training_config.get('class_names', [])}")

        # Check class distribution and minimum class size
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        logger.info(f"=== CLASS DISTRIBUTION ===")
        logger.info(f"Class value counts:\n{class_counts}")
        logger.info(f"Min class size: {min_class_size}")
        
        # Get test_size and random_state from training_config with fallback to model defaults
        config = obj.training_config or {}
        test_size = config.get('test_size', obj.test_size)
        random_state = config.get('random_state', obj.random_state)
        
        if not is_regression and min_class_size < 2:
            # For classes with only 1 sample, we need to handle this
            logger.warning(f"Some classes have only {min_class_size} sample(s). Classes: {class_counts[class_counts < 2].index.tolist()}")
            
            # If any class has less than 2 samples, use a minimum test_size that leaves at least 1 sample per class
            min_test_size = max(0.1, 2.0 / len(y))  # At least 10% or enough for 2 samples per class
            if test_size < min_test_size:
                logger.info(f"Adjusting test_size from {test_size} to {min_test_size} due to small classes")
                test_size = min_test_size

        # Use stratified split for classification to ensure balanced representation
        if is_regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"=== REGRESSION SPLIT ===")
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        else:
            # Stratified split for classification
            try:
                logger.info(f"=== ATTEMPTING STRATIFIED SPLIT ===")
                logger.info(f"Test size: {test_size}, Random state: {random_state}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                logger.info(f"Stratified split successful")
            except ValueError as e:
                # Fall back to regular split if stratification fails
                logger.warning(f"Stratified split failed: {e}. Using regular split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            logger.info(f"=== POST-SPLIT ANALYSIS ===")
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            logger.info(f"y_train unique values: {sorted(y_train.unique())}")
            logger.info(f"y_test unique values: {sorted(y_test.unique())}")
            logger.info(f"y_train dtype: {y_train.dtype}")
            logger.info(f"y_test dtype: {y_test.dtype}")

        # Apply label encoding after split for models that require consecutive labels
        label_encoder = None
        models_requiring_consecutive_labels = [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.GRADIENT_BOOSTING]
        
        # CRITICAL FIX: All classification problems with float targets need encoding
        needs_encoding = False
        if not is_regression:
            # Check if target values are floats or strings (need encoding for sklearn classifiers)
            if y_train.dtype.kind in 'fc' or y_train.dtype == 'object' or y_train.dtype.name == 'category':
                needs_encoding = True
                logger.info(f"Classification with {y_train.dtype} target detected - requires label encoding")
            
            # Also check for non-consecutive integer classes
            elif obj.model_type in models_requiring_consecutive_labels:
                train_classes = sorted(y_train.unique())
                test_classes = sorted(y_test.unique())
                all_classes = sorted(set(train_classes) | set(test_classes))
                expected_classes = list(range(len(all_classes)))
                if all_classes != expected_classes:
                    needs_encoding = True
                    logger.info(f"Non-consecutive classes detected for {obj.model_type} - requires label encoding")
        
        if needs_encoding:
            logger.info(f"=== LABEL ENCODING FOR CLASSIFICATION ===")
            
            # Check if we need to encode to make labels consecutive
            train_classes = sorted(y_train.unique())
            test_classes = sorted(y_test.unique())
            all_classes = sorted(set(train_classes) | set(test_classes))
            expected_classes = list(range(len(all_classes)))
            
            logger.info(f"Current classes: {all_classes}")
            logger.info(f"Expected classes: {expected_classes}")
            
            logger.info(f"Target needs encoding, applying LabelEncoder")
            label_encoder = LabelEncoder()
            
            # Log pre-encoding state
            logger.info(f"BEFORE encoding - y_train unique: {sorted(y_train.unique())}")
            logger.info(f"BEFORE encoding - y_test unique: {sorted(y_test.unique())}")
            
            # CRITICAL FIX: Fit on all classes but ensure training classes are consecutive
            # First, create a combined series to know all possible classes
            combined_y = pd.concat([y_train, y_test])
            all_unique_classes = sorted(combined_y.unique())
            logger.info(f"All unique classes in dataset: {all_unique_classes}")
            
            # Fit the encoder on all classes (for consistency in predictions)
            label_encoder.fit([str(c) for c in all_unique_classes])
            
            # Transform both sets
            y_train_encoded = label_encoder.transform(y_train.astype(str))
            y_test_encoded = label_encoder.transform(y_test.astype(str))
            
            logger.info(f"After transform - y_train_encoded unique: {sorted(y_train_encoded)}")
            logger.info(f"After transform - y_test_encoded unique: {sorted(y_test_encoded)}")
            
            # CRITICAL: Re-map training classes to be consecutive starting from 0
            train_unique_encoded = sorted(np.unique(y_train_encoded))
            test_unique_encoded = sorted(np.unique(y_test_encoded))
            
            logger.info(f"Training encoded unique: {train_unique_encoded}")
            logger.info(f"Test encoded unique: {test_unique_encoded}")
            
            # Create a mapping from current encoded values to consecutive values
            train_class_mapping = {old_val: new_val for new_val, old_val in enumerate(train_unique_encoded)}
            logger.info(f"Training class mapping: {train_class_mapping}")
            
            # Apply the remapping to make training classes consecutive [0, 1, 2, ..., n-1]
            y_train_final = np.array([train_class_mapping[val] for val in y_train_encoded])
            
            # For test set, map only the classes that exist in training, others become a special value
            max_train_class = len(train_unique_encoded) - 1
            y_test_final = []
            for val in y_test_encoded:
                if val in train_class_mapping:
                    y_test_final.append(train_class_mapping[val])
                else:
                    # Handle test classes not seen in training - map to closest or special class
                    logger.warning(f"Test class {val} not seen in training, mapping to class 0")
                    y_test_final.append(0)  # or could use max_train_class
            
            y_test_final = np.array(y_test_final)
            
            # Log encoding details
            logger.info(f"=== FINAL ENCODING RESULTS ===")
            logger.info(f"Label encoder classes: {label_encoder.classes_.tolist()}")
            logger.info(f"Number of original classes: {len(label_encoder.classes_)}")
            logger.info(f"Training classes after remapping: {sorted(np.unique(y_train_final))}")
            logger.info(f"Test classes after remapping: {sorted(np.unique(y_test_final))}")
            logger.info(f"Training classes are consecutive: {list(range(len(np.unique(y_train_final)))) == sorted(np.unique(y_train_final))}")
            
            # Convert back to pandas Series with proper index
            y_train = pd.Series(y_train_final, index=y_train.index)
            y_test = pd.Series(y_test_final, index=y_test.index)
            
            logger.info(f"FINAL y_train unique: {sorted(y_train.unique())}")
            logger.info(f"FINAL y_test unique: {sorted(y_test.unique())}")
            logger.info(f"Training set class range: {y_train.min()} to {y_train.max()}")
            logger.info(f"Test set class range: {y_test.min()} to {y_test.max()}")
            logger.info(f"Expected by model: [0, 1, 2, ..., {len(np.unique(y_train)) - 1}]")
            logger.info(f"=================================")
        else:
            if obj.model_type in models_requiring_consecutive_labels and is_regression:
                # For regression, no encoding needed
                logger.info(f"=== REGRESSION MODE FOR {obj.model_type} ===")
                logger.info(f"No label encoding needed for regression")
            else:
                logger.info(f"Classes are already consecutive, no encoding needed")
                # For other models, check if we might still need encoding
                if not is_regression:
                    train_classes = sorted(y_train.unique())
                    test_classes = sorted(y_test.unique())
                    all_classes = sorted(set(train_classes) | set(test_classes))
                    expected_classes = list(range(len(all_classes)))
                    
                    logger.info(f"Current classes: {all_classes}")
                    logger.info(f"Expected classes: {expected_classes}")
                    
                    if all_classes != expected_classes:
                        logger.warning(f"Non-consecutive classes detected for {obj.model_type} - may still cause issues")
                        logger.warning(f"Consider adding {obj.model_type} to models_requiring_consecutive_labels if errors occur")

        s3_path, acc = train_sklearn_model(obj, X_train, y_train, X_test, y_test, label_encoder=label_encoder)
        obj.model_file.name = s3_path
        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log is already set by TrainingLogger in train_sklearn_model
        # Do not overwrite it here
        obj.save()

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        logger.error(f"Training failed for model {model_id}: {str(e)}")
        try:
            obj.status = ModelStatus.FAILED
            obj.training_log = f"Training failed: {str(e)}"
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
        except MLModel.DoesNotExist:
            logger.error(f"Model {model_id} was deleted during training")

    return acc if acc is not None else 0.0


@shared_task
def train_nn_task(model_id):
    import logging
    logger = logging.getLogger(__name__)
    acc = None  # Initialize accuracy variable

    try:
        obj = MLModel.objects.get(id=model_id)
    except MLModel.DoesNotExist:
        logger.error(f"Model {model_id} does not exist. It may have been deleted.")
        return 0.0

    try:
        # Validate dataset has CSV file
        if not obj.dataset.minio_csv_key:
            error_msg = "This model requires a tabular dataset with a CSV file. The selected dataset appears to be an image dataset."
            logger.error(f"{error_msg} Model: {model_id}, Dataset: {obj.dataset.id}")
            obj.status = ModelStatus.FAILED
            obj.training_log = error_msg
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=error_msg)
            raise ValueError(error_msg)

        csv_obj = obj.dataset.get_csv_file_object()
        df = pd.read_csv(csv_obj)
        target = obj.target_column
        config = obj.training_config or {}

        # Get features, respecting excluded_columns
        excluded_columns = config.get("excluded_columns", [])
        if config.get("features"):
            features = config.get("features")
        else:
            features = [col for col in df.columns if col != target and col not in excluded_columns]
            logger.info(f"Auto-selected {len(features)} features (excluded {len(excluded_columns)} columns)")

        # Encode categorical features (e.g., 'M'/'F' -> 0/1)
        X, feature_encoders = encode_categorical_features(df, features, logger)
        if feature_encoders:
            logger.info(f"Encoded {len(feature_encoders)} categorical columns: {list(feature_encoders.keys())}")

        y = df[target]
        
        # Determine if this is a regression or classification problem
        is_regression = _is_regression_problem(y)
        logger.info(f"Problem type detected: {'Regression' if is_regression else 'Classification'}")
        
        # Store original class names and detailed analysis
        original_classes = sorted(y.unique().tolist())
        logger.info(f"=== TARGET ANALYSIS (NN) ===")
        logger.info(f"Target column: {target}")
        logger.info(f"Target dtype: {y.dtype}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Original unique values ({len(original_classes)}): {original_classes}")
        
        # Check class distribution for classification problems
        if not is_regression:
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            logger.info(f"=== CLASS DISTRIBUTION (NN) ===")
            logger.info(f"Class value counts:\n{class_counts}")
            logger.info(f"Min class size: {min_class_size}")
            
            if min_class_size < 2:
                logger.warning(f"Some classes have only {min_class_size} sample(s). Classes: {class_counts[class_counts < 2].index.tolist()}")
                min_test_size = max(0.1, 2.0 / len(y))
                if obj.test_size < min_test_size:
                    logger.info(f"Adjusting test_size from {obj.test_size} to {min_test_size} due to small classes")
                    obj.test_size = min_test_size

        # Use stratified split for classification to ensure balanced representation
        if is_regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=obj.test_size, random_state=obj.random_state
            )
            logger.info(f"=== REGRESSION SPLIT (NN) ===")
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        else:
            # Stratified split for classification
            try:
                logger.info(f"=== ATTEMPTING STRATIFIED SPLIT (NN) ===")
                logger.info(f"Test size: {obj.test_size}, Random state: {obj.random_state}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=obj.test_size, random_state=obj.random_state, stratify=y
                )
                logger.info(f"Stratified split successful")
            except ValueError as e:
                # Fall back to regular split if stratification fails
                logger.warning(f"Stratified split failed: {e}. Using regular split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=obj.test_size, random_state=obj.random_state
                )
            
            logger.info(f"=== POST-SPLIT ANALYSIS (NN) ===")
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            logger.info(f"y_train unique values: {sorted(y_train.unique())}")
            logger.info(f"y_test unique values: {sorted(y_test.unique())}")
            logger.info(f"y_train dtype: {y_train.dtype}")
            logger.info(f"y_test dtype: {y_test.dtype}")
            
            # Check if NN needs label encoding (it usually does for non-consecutive classes)
            train_classes = sorted(y_train.unique())
            test_classes = sorted(y_test.unique())
            all_classes = sorted(set(train_classes) | set(test_classes))
            expected_classes = list(range(len(all_classes)))
            
            logger.info(f"Current classes: {all_classes}")
            logger.info(f"Expected classes for NN: {expected_classes}")
            
            if all_classes != expected_classes:
                logger.info(f"NN requires consecutive classes, applying LabelEncoder")
                label_encoder = LabelEncoder()
                
                # Log pre-encoding state  
                logger.info(f"BEFORE encoding - y_train unique: {sorted(y_train.unique())}")
                logger.info(f"BEFORE encoding - y_test unique: {sorted(y_test.unique())}")
                
                # Create a combined series for fitting to ensure all classes are known
                combined_y = pd.concat([y_train, y_test])
                all_unique_classes = sorted(combined_y.unique())
                logger.info(f"All unique classes in NN dataset: {all_unique_classes}")
                
                # Fit the encoder on all classes
                label_encoder.fit([str(c) for c in all_unique_classes])
                
                # Transform both sets
                y_train_encoded = label_encoder.transform(y_train.astype(str))
                y_test_encoded = label_encoder.transform(y_test.astype(str))
                
                logger.info(f"After transform - y_train_encoded unique: {sorted(y_train_encoded)}")
                logger.info(f"After transform - y_test_encoded unique: {sorted(y_test_encoded)}")
                
                # CRITICAL: Re-map training classes to be consecutive starting from 0
                train_unique_encoded = sorted(np.unique(y_train_encoded))
                test_unique_encoded = sorted(np.unique(y_test_encoded))
                
                logger.info(f"NN Training encoded unique: {train_unique_encoded}")
                logger.info(f"NN Test encoded unique: {test_unique_encoded}")
                
                # Create a mapping from current encoded values to consecutive values for NN
                train_class_mapping = {old_val: new_val for new_val, old_val in enumerate(train_unique_encoded)}
                logger.info(f"NN Training class mapping: {train_class_mapping}")
                
                # Apply the remapping to make training classes consecutive [0, 1, 2, ..., n-1]
                y_train_final = np.array([train_class_mapping[val] for val in y_train_encoded])
                
                # For test set, map only the classes that exist in training
                y_test_final = []
                for val in y_test_encoded:
                    if val in train_class_mapping:
                        y_test_final.append(train_class_mapping[val])
                    else:
                        logger.warning(f"NN: Test class {val} not seen in training, mapping to class 0")
                        y_test_final.append(0)
                
                y_test_final = np.array(y_test_final)
                
                logger.info(f"=== NN FINAL ENCODING RESULTS ===")
                logger.info(f"NN Label encoder classes: {label_encoder.classes_.tolist()}")
                logger.info(f"NN Training classes after remapping: {sorted(np.unique(y_train_final))}")
                logger.info(f"NN Test classes after remapping: {sorted(np.unique(y_test_final))}")
                logger.info(f"NN Training classes are consecutive: {list(range(len(np.unique(y_train_final)))) == sorted(np.unique(y_train_final))}")
                
                # Convert back to pandas Series
                y_train = pd.Series(y_train_final, index=y_train.index)
                y_test = pd.Series(y_test_final, index=y_test.index)
                
                logger.info(f"FINAL NN y_train unique: {sorted(y_train.unique())}")
                logger.info(f"FINAL NN y_test unique: {sorted(y_test.unique())}")
                logger.info(f"=================================")
            else:
                logger.info(f"Classes are already consecutive for NN")
        
        logger.info(f"=== SENDING TO NN TRAINING ===")
        logger.info(f"Final y_train range: {y_train.min() if not is_regression else 'N/A'} to {y_train.max() if not is_regression else 'N/A'}")
        logger.info(f"Final y_test range: {y_test.min() if not is_regression else 'N/A'} to {y_test.max() if not is_regression else 'N/A'}")
        logger.info(f"Is regression: {is_regression}")

        s3_path, acc = train_nn(obj, X_train, y_train, X_test, y_test)
        obj.model_file.name = s3_path
        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log is already set by TrainingLogger in train_nn
        # Do not overwrite it here
        obj.save()

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        logger.error(f"NN training failed for model {model_id}: {str(e)}")
        try:
            obj.status = ModelStatus.FAILED
            obj.training_log = f"Training failed: {str(e)}"
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
        except MLModel.DoesNotExist:
            logger.error(f"Model {model_id} was deleted during training")

    return acc if acc is not None else 0.0


@shared_task
def train_cnn_task(model_id):
    import logging
    logger = logging.getLogger(__name__)
    acc = None  # Initialize accuracy variable

    try:
        obj = MLModel.objects.get(id=model_id)
    except MLModel.DoesNotExist:
        logger.error(f"Model {model_id} does not exist. It may have been deleted.")
        return 0.0

    try:
        # Validate dataset is an image dataset
        if not obj.dataset.is_image_dataset and not obj.dataset.minio_images_prefix:
            raise ValueError("CNN models require an image dataset. The selected dataset appears to be a tabular dataset.")

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.TRAINING)
        obj.status = ModelStatus.TRAINING
        obj.save()

        model_path, acc = train_cnn(obj)

        # Set model file path with logging
        logger.info(f"🔗 Linking model file to database: {model_path}")
        obj.model_file.name = model_path
        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log is already set by TrainingLogger in train_cnn
        # Do not overwrite it here
        obj.save()
        
        # Verify the save was successful
        logger.info(f"✅ CNN training complete - Model file: {obj.model_file.name}, Status: {obj.status}, Accuracy: {acc}")

        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        logger.error(f"CNN training failed for model {model_id}: {str(e)}")
        try:
            obj.status = ModelStatus.FAILED
            obj.training_log = f"CNN training failed: {str(e)}"
            obj.save()
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
        except MLModel.DoesNotExist:
            logger.error(f"Model {model_id} was deleted during training")

    return acc if acc is not None else 0.0


@shared_task
def extract_image_dataset_task(dataset_id):
    """Extract image dataset ZIP file in background - MinIO only"""
    import logging
    import tempfile
    logger = logging.getLogger(__name__)

    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        logger.error(f"Dataset {dataset_id} does not exist.")
        return

    if not dataset.minio_zip_key:
        logger.warning(f"Dataset {dataset_id} does not have a MinIO ZIP key.")
        return

    try:
        logger.info(f"Starting extraction for dataset {dataset_id}")

        # Download ZIP from MinIO
        logger.info(f"[EXTRACT] Downloading ZIP from MinIO: {dataset.minio_zip_key}")
        zip_bytes = dataset.get_zip_bytes()
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.write(zip_bytes)
        temp_zip.close()
        zip_path = temp_zip.name
        logger.info(f"[EXTRACT] ZIP downloaded to temp file: {zip_path}")

        # Create temp extraction directory
        extract_to = tempfile.mkdtemp(prefix=f'dataset_{dataset.id}_')
        logger.info(f"[EXTRACT] Extracting to temp directory: {extract_to}")

        # Extract ZIP
        temp_dir = os.path.join(extract_to, "tmp_extraction")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        logger.info(f"[EXTRACT] ZIP extracted to: {temp_dir}")

        # Handle nested directory structure
        top = next(Path(temp_dir).iterdir(), None)
        if top and top.is_dir() and len(list(Path(temp_dir).iterdir())) == 1:
            for item in top.iterdir():
                shutil.move(str(item), extract_to)
            shutil.rmtree(temp_dir)
            logger.info(f"[EXTRACT] Unwrapped nested directory")
        else:
            for item in Path(temp_dir).iterdir():
                shutil.move(str(item), extract_to)
            shutil.rmtree(temp_dir)

        # Count images and calculate total size
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_count = 0
        total_size = 0
        for root, dirs, files in os.walk(extract_to):
            level = root.replace(extract_to, '').count(os.sep)
            if level < 3:
                for f in files:
                    file_path = Path(root) / f
                    if file_path.suffix.lower() in image_extensions:
                        image_count += 1
                        total_size += file_path.stat().st_size
            else:
                del dirs[:]

        logger.info(f"[EXTRACT] Found {image_count} images, total size: {total_size} bytes")

        # Upload extracted images to MinIO
        from core.storage_utils import upload_directory_to_minio
        minio_prefix = f"datasets/{dataset.id}/images/"
        logger.info(f"[EXTRACT] Uploading extracted images to MinIO: {minio_prefix}")
        upload_directory_to_minio(extract_to, minio_prefix)
        logger.info(f"[EXTRACT] Images uploaded to MinIO successfully")

        # Update dataset
        dataset.minio_images_prefix = minio_prefix
        dataset.row_count = image_count
        dataset.file_size = total_size
        dataset.is_processed = True
        dataset.processing_errors = None

        # Clean up temp files
        shutil.rmtree(extract_to)
        os.unlink(zip_path)
        logger.info(f"[EXTRACT] Cleaned up temp files")

        dataset.save(update_fields=["minio_images_prefix", "row_count", "file_size", "is_processed", "processing_errors"])
        logger.info(f"Successfully extracted {image_count} images for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"Extraction failed for dataset {dataset_id}: {str(e)}")
        dataset.processing_errors = f"Extraction failed: {str(e)}"
        dataset.save(update_fields=["processing_errors"])
