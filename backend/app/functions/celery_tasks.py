from celery import shared_task
from app.models import MLModel, Dataset
from app.functions.training import train_cnn, train_nn, train_sklearn_model
import pandas as pd
from sklearn.model_selection import train_test_split
from app.models.choices import ModelStatus
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
        obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
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
        obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
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
        obj.training_log = f"Training complete for CNN. Validation Accuracy: {acc:.4f}" if acc else "Training complete for CNN."
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
