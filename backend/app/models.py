# models.py
from django.db import models
from core.models import BaseModel

class NeuralNetworkModel(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    num_layers = models.IntegerField()
    layer_configurations = models.JSONField(
        help_text="JSON structure: [{'type': 'dense', 'units': 64, 'activation': 'relu'}, ...]"
    )
    input_shape = models.CharField(max_length=255, help_text="e.g., '(64,)'")
    output_shape = models.CharField(max_length=255, help_text="e.g., '(10,)'")
    learning_rate = models.FloatField()
    batch_size = models.IntegerField()
    epochs = models.IntegerField()
    optimizer = models.CharField(
        max_length=50,
        choices=[('sgd', 'SGD'), ('adam', 'Adam')],
    )
    optimizer_params = models.JSONField(null=True, blank=True)
    trained_model_path = models.CharField(max_length=1024, null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Neural Network"
        verbose_name_plural = "Neural Networks"

class TrainingData(BaseModel):
    name = models.CharField(max_length=255)
    data_file = models.FileField(upload_to="training_data/")
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = "Training Data"
        verbose_name_plural = "Training Data"
