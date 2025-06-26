import os
import zipfile
import shutil
from pathlib import Path
from django.db import models
from django.conf import settings
from core.models import (
    BaseModel,
)
from app.models.choices import ModelStatus, ModelType


class Dataset(BaseModel):
    name = models.CharField(max_length=255, default="Unnamed Dataset")
    csv_file = models.FileField(upload_to="csv_datasets/", blank=True, null=True)
    image_folder = models.FileField(upload_to="image_zips/", blank=True, null=True)
    extracted_path = models.CharField(max_length=512, blank=True, null=True)
    is_image_dataset = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.image_folder and self.image_folder.name.endswith(".zip"):
            extract_to = os.path.join(
                settings.MEDIA_ROOT, "image_datasets", str(self.id)
            )
            os.makedirs(extract_to, exist_ok=True)

            temp_dir = os.path.join(extract_to, "tmp_extraction")
            with zipfile.ZipFile(self.image_folder.path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            top = next(Path(temp_dir).iterdir(), None)
            if top and top.is_dir():
                for item in top.iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)
            else:
                for item in Path(temp_dir).iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)

            self.extracted_path = extract_to
            super().save(update_fields=["extracted_path"])

    def __str__(self):
        return self.name


class MLModel(BaseModel):
    name = models.CharField(max_length=255, default="Unnamed Model")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    target_column = models.CharField(max_length=255, default="target")
    training_config = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=50, choices=ModelStatus.choices, default=ModelStatus.PENDING
    )
    model_file = models.FileField(
        upload_to="trained_models/",
        null=True,
        blank=True,
        help_text="Path to the trained model file",
    )
    training_log = models.TextField(
        blank=True, null=True, default="Training not started."
    )
    model_type = models.CharField(
        max_length=50, choices=ModelType.choices, default=ModelType.LOGISTIC_REGRESSION
    )
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(default=42)
    max_iter = models.IntegerField(default=1000)

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    model = models.OneToOneField(
        "MLModel", on_delete=models.CASCADE, related_name="training_run"
    )
    history = models.JSONField(default=list, blank=True)

    def add_entry(self, status: str, accuracy: float = None, error: str = None):
        from django.utils.timezone import now

        entry = {
            "timestamp": now().isoformat(),
            "status": status,
            "accuracy": accuracy,
            "error": error,
        }
        self.history.append(entry)
        self.save()
