import os
import uuid
from django.core.validators import FileExtensionValidator
from django.db import models
from core.models import BaseModel  # assuming BaseModel is defined in the same file or imported appropriately


class Dataset(BaseModel):
    name = models.CharField(max_length=255, default="Untitled Dataset")
    file = models.FileField(
        upload_to="datasets/",
        validators=[FileExtensionValidator(["csv"])],
        help_text="Upload a CSV file"
    )

    def __str__(self):
        return self.name
from django.db import models

class ModelStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    TRAINING = "training", "Training"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"

class ModelType(models.TextChoices):
    LOGISTIC_REGRESSION = "logistic_regression", "Logistic Regression"
    RANDOM_FOREST = "random_forest", "Random Forest"
    SVM = "svm", "Support Vector Machine"
    NAIVE_BAYES = "naive_bayes","Naive Bayes"
    KNN = "knn", "k-Nearest Neighbours"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING","Gradient Boosting"
    NEURAL_NETWORK = "neural_network", "Neural Network (PyTorch)"
    
    
class MLModel(BaseModel):
    name = models.CharField(max_length=255, default="Unnamed Model")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    target_column = models.CharField(max_length=255, default="target")
    training_config = models.JSONField(default=dict,blank=True)
    status = models.CharField(
        max_length=50,
        choices=ModelStatus.choices,
        default=ModelStatus.PENDING
    )
    model_file = models.FileField(
        upload_to="trained_models/",
        null=True,
        blank=True,
        help_text="Path to the trained model file"
    )
    training_log = models.TextField(
        blank=True,
        null=True,
        default="Training not started."
    )
    model_type = models.CharField(
        max_length=50,
        choices=ModelType.choices,
        default=ModelType.LOGISTIC_REGRESSION
    )
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(default=42)
    max_iter = models.IntegerField(default=1000)  # useful for LR and SVM
    def __str__(self):
        return self.name

class TrainingRun(models.Model):
    model = models.ForeignKey("MLModel", on_delete=models.CASCADE, related_name="training_runs")
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    algorithm = models.CharField(max_length=255)
    status = models.CharField(max_length=50, default="pending")  
    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.model.name} ({self.status}) @ {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}"