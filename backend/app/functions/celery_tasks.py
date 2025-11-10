from celery import shared_task
from app.models import MLModel, Dataset
from app.functions.training import (
    train_cnn, train_nn, train_sklearn_model,
    train_transfer_learning, train_rnn
)
import pandas as pd
from sklearn.model_selection import train_test_split
from app.models.choices import ModelStatus, ModelType
from app.models.main import TrainingRun
import os
import zipfile
import shutil
from pathlib import Path
from django.conf import settings


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
        if not obj.dataset.csv_file or not obj.dataset.csv_file.name:
            error_msg = "This model requires a tabular dataset with a CSV file. The selected dataset appears to be an image dataset."
            logger.error(f"{error_msg} Model: {model_id}, Dataset: {obj.dataset.id}")
            obj.status = ModelStatus.FAILED
            obj.training_log = error_msg
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=error_msg)
            raise ValueError(error_msg)

        df = pd.read_csv(obj.dataset.csv_file.path)
        target = obj.target_column
        config = obj.training_config or {}
        features = config.get("features") or df.drop(columns=[target]).columns.tolist()

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=obj.test_size, random_state=obj.random_state
        )

        s3_path, acc = train_sklearn_model(obj, X_train, y_train, X_test, y_test)
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
        if not obj.dataset.csv_file or not obj.dataset.csv_file.name:
            error_msg = "This model requires a tabular dataset with a CSV file. The selected dataset appears to be an image dataset."
            logger.error(f"{error_msg} Model: {model_id}, Dataset: {obj.dataset.id}")
            obj.status = ModelStatus.FAILED
            obj.training_log = error_msg
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=error_msg)
            raise ValueError(error_msg)

        df = pd.read_csv(obj.dataset.csv_file.path)
        target = obj.target_column
        config = obj.training_config or {}
        features = config.get("features") or df.drop(columns=[target]).columns.tolist()

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=obj.test_size, random_state=obj.random_state
        )

        s3_path, acc = train_nn(obj, X_train, y_train, X_test, y_test)
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
        if not obj.dataset.is_image_dataset and not obj.dataset.image_folder:
            raise ValueError("CNN models require an image dataset. The selected dataset appears to be a tabular dataset.")

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.TRAINING)
        obj.status = ModelStatus.TRAINING
        obj.save()

        model_path, acc = train_cnn(obj)

        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log is already set by TrainingLogger in train_cnn
        # Do not overwrite it here
        obj.save()

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
def train_transfer_learning_task(model_id):
    """Train transfer learning models (ResNet, VGG, EfficientNet)"""
    import logging
    logger = logging.getLogger(__name__)
    acc = None

    try:
        obj = MLModel.objects.get(id=model_id)
    except MLModel.DoesNotExist:
        logger.error(f"Model {model_id} does not exist. It may have been deleted.")
        return 0.0

    try:
        # Validate dataset is an image dataset
        if not obj.dataset.is_image_dataset and not obj.dataset.image_folder:
            raise ValueError("Transfer learning models require an image dataset.")

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.TRAINING)
        obj.status = ModelStatus.TRAINING
        obj.save()

        model_path, acc = train_transfer_learning(obj)

        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log should be set by TrainingLogger in train_transfer_learning
        # Do not overwrite it here
        obj.save()

        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        logger.error(f"Transfer learning training failed for model {model_id}: {str(e)}")
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
def train_rnn_task(model_id):
    """Train RNN/LSTM/GRU models for time series"""
    import logging
    logger = logging.getLogger(__name__)
    acc = None

    try:
        obj = MLModel.objects.get(id=model_id)
    except MLModel.DoesNotExist:
        logger.error(f"Model {model_id} does not exist. It may have been deleted.")
        return 0.0

    try:
        # Validate dataset has CSV file
        if not obj.dataset.csv_file or not obj.dataset.csv_file.name:
            error_msg = "RNN/LSTM/GRU models require a tabular dataset with a CSV file."
            logger.error(f"{error_msg} Model: {model_id}, Dataset: {obj.dataset.id}")
            obj.status = ModelStatus.FAILED
            obj.training_log = error_msg
            obj.save()
            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.FAILED, error=error_msg)
            raise ValueError(error_msg)

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.TRAINING)
        obj.status = ModelStatus.TRAINING
        obj.save()

        model_path, acc = train_rnn(obj)

        obj.status = ModelStatus.COMPLETE
        obj.accuracy = acc
        # Note: training_log should be set by TrainingLogger in train_rnn
        # Do not overwrite it here
        obj.save()

        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        logger.error(f"RNN training failed for model {model_id}: {str(e)}")
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
def extract_image_dataset_task(dataset_id):
    """Extract image dataset ZIP file in background"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        logger.error(f"Dataset {dataset_id} does not exist.")
        return

    if not dataset.image_folder or not dataset.image_folder.name.endswith(".zip"):
        logger.warning(f"Dataset {dataset_id} does not have a valid ZIP file.")
        return

    try:
        logger.info(f"Starting extraction for dataset {dataset_id}")

        extract_to = os.path.join(
            settings.MEDIA_ROOT, "image_datasets", str(dataset.id)
        )
        os.makedirs(extract_to, exist_ok=True)

        temp_dir = os.path.join(extract_to, "tmp_extraction")
        with zipfile.ZipFile(dataset.image_folder.path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Handle nested directory structure
        top = next(Path(temp_dir).iterdir(), None)
        if top and top.is_dir() and len(list(Path(temp_dir).iterdir())) == 1:
            for item in top.iterdir():
                shutil.move(str(item), extract_to)
            shutil.rmtree(temp_dir)
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

        dataset.extracted_path = extract_to
        dataset.row_count = image_count
        dataset.file_size = total_size
        dataset.is_processed = True
        dataset.processing_errors = None
        dataset.save(update_fields=["extracted_path", "row_count", "file_size", "is_processed", "processing_errors"])

        logger.info(f"Successfully extracted {image_count} images for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"Extraction failed for dataset {dataset_id}: {str(e)}")
        dataset.processing_errors = f"Extraction failed: {str(e)}"
        dataset.save(update_fields=["processing_errors"])
