from django.db import models
from core.models import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import boto3
import json


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

    def create_model(self):
        """
        Creates a PyTorch model based on the configurations in the database.
        """
        layer_configs = self.layer_configurations
        layers = []
        input_size = eval(self.input_shape)[0]
        output_size = eval(self.output_shape)[0]
        last_size = input_size

        for config in layer_configs:
            layers.append(nn.Linear(config["in_features"], config["out_features"]))
            if config["activation"] == "relu":
                layers.append(nn.ReLU())
            last_size = config["out_features"]

        layers.append(nn.Linear(last_size, output_size))
        return nn.Sequential(*layers)

    def train_model(self, data):
        """
        Train the model with provided data.
        """
        model = self.create_model()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        inputs, outputs = data
        inputs_tensor = torch.from_numpy(inputs).float()
        outputs_tensor = torch.from_numpy(outputs).float()

        num_epochs = self.epochs
        batch_size = self.batch_size
        num_batches = inputs_tensor.size(0) // batch_size

        for epoch in range(num_epochs):
            model.train()
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch_inputs = inputs_tensor[batch_start:batch_end]
                batch_outputs = outputs_tensor[batch_start:batch_end]

                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

        self.save_model_to_minio(model)

    def predict(self, input_data):
        """
        Use the trained model to make predictions.
        """
        model = self.create_model()
        model.load_state_dict(torch.load(self.trained_model_path))
        model.eval()

        input_tensor = torch.from_numpy(np.array(input_data)).float()
        with torch.no_grad():
            predictions = model(input_tensor)
        return predictions

    def save_model_to_minio(self, model, bucket_name="models", key="trained_model.pt"):
        """
        Save the trained model to MinIO.
        """
        temp_path = "/tmp/trained_model.pt"
        torch.save(model.state_dict(), temp_path)

        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        try:
            s3.upload_file(temp_path, bucket_name, key)
            print(f"File uploaded successfully to bucket '{bucket_name}' with key '{key}'.")
        except Exception as e:
            print(f"Error uploading file to MinIO: {e}")


class TrainingData(BaseModel):
    name = models.CharField(max_length=255)
    data_file = models.FileField(upload_to="training_data/")
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = "Training Data"
        verbose_name_plural = "Training Data"

    @staticmethod
    def generate_quadratic_data(num_samples=10000, seed=42):
        """
        Generate synthetic data for training the neural network.
        """
        np.random.seed(seed)
        a = np.random.uniform(1, 2, size=(num_samples, 1))
        b = np.random.uniform(-5, 5, size=(num_samples, 1))
        c = np.random.uniform(-5, 5, size=(num_samples, 1))

        discriminant = b**2 - 4 * a * c
        positive_discriminant = discriminant >= 0
        a = a[positive_discriminant.flatten()]
        b = b[positive_discriminant.flatten()]
        c = c[positive_discriminant.flatten()]
        discriminant = discriminant[positive_discriminant.flatten()]
        sqrt_discriminant = np.sqrt(discriminant)

        x1 = (-b + sqrt_discriminant) / (2 * a)
        x2 = (-b - sqrt_discriminant) / (2 * a)

        ab = a * b
        ac = a * c
        bc = b * c
        a_squared = a**2
        b_squared = b**2
        c_squared = c**2

        inputs = np.hstack([a, b, c, ab, ac, bc, a_squared, b_squared, c_squared])
        outputs = np.hstack([x1, x2])
        return inputs, outputs
