from django.core.validators import FileExtensionValidator
from django.db import models
from core.models import (
    BaseModel,
)
from app.models.choices import ModelStatus, ModelType


class Dataset(BaseModel):
    name = models.CharField(max_length=255, default="Untitled Dataset")
    file = models.FileField(
        upload_to="datasets/",
        validators=[FileExtensionValidator(["csv"])],
        help_text="Upload a CSV file",
    )

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
    max_iter = models.IntegerField(default=1000)  # useful for LR and SVM

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    model = models.OneToOneField(
        "MLModel", on_delete=models.CASCADE, related_name="training_run"
    )
    history = models.JSONField(
        default=list, blank=True
    )  # stores all runs as list of dicts

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
