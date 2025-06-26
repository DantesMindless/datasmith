from celery import shared_task
from django.conf import settings
from app.models import MLModel
from app.functions.training import train_cnn, train_nn, train_sklearn_model
import pandas as pd
from sklearn.model_selection import train_test_split
from app.models.choices import ModelStatus
from app.models.main import TrainingRun


@shared_task
def train_sklearn_task(model_id):
    obj = MLModel.objects.get(id=model_id)
    df = pd.read_csv(obj.dataset.csv_file.path)
    target = obj.target_column
    config = obj.training_config or {}
    features = config.get("features") or df.drop(columns=[target]).columns.tolist()

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=obj.test_size, random_state=obj.random_state
    )

    try:
        s3_path, acc = train_sklearn_model(obj, X_train, y_train, X_test, y_test)
        obj.model_file.name = s3_path
        obj.status = ModelStatus.COMPLETE
        obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
        obj.save()

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        obj.status = ModelStatus.FAILED
        obj.training_log = f"Training failed: {str(e)}"
        obj.save()
        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.FAILED, error=str(e))

    return acc


@shared_task
def train_nn_task(model_id):
    obj = MLModel.objects.get(id=model_id)
    df = pd.read_csv(obj.dataset.csv_file.path)
    target = obj.target_column
    config = obj.training_config or {}
    features = config.get("features") or df.drop(columns=[target]).columns.tolist()

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=obj.test_size, random_state=obj.random_state
    )
    try:
        s3_path, acc = train_nn(obj, X_train, y_train, X_test, y_test)
        obj.status = ModelStatus.COMPLETE
        obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
        obj.save()

        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
    except Exception as e:
        obj.status = ModelStatus.FAILED
        obj.training_log = f"Training failed: {str(e)}"
        obj.save()
        run, _ = TrainingRun.objects.get_or_create(model=obj)
        run.add_entry(status=ModelStatus.FAILED, error=str(e))


@shared_task
def train_cnn_task(model_id):
    obj = MLModel.objects.get(id=model_id)
    run, _ = TrainingRun.objects.get_or_create(model=obj)

    try:
        run.add_entry(status=ModelStatus.TRAINING)
        obj.status = ModelStatus.TRAINING
        obj.save()

        model_path, acc = train_cnn(obj)

        obj.status = ModelStatus.COMPLETE
        obj.training_log = (
            "Training complete for CNN. Accuracy not calculated (no validation set)."
        )
        obj.save()

        run.add_entry(status=ModelStatus.COMPLETE)
    except Exception as e:
        obj.status = ModelStatus.FAILED
        obj.training_log = f"CNN training failed: {str(e)}"
        obj.save()
        run.add_entry(status=ModelStatus.FAILED, error=str(e))
        raise  
