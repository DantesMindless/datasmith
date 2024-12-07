from typing import List, Dict, Any, Optional, Tuple
from django.db import models
from core.models import BaseModel
from datasource.models import DataSource
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import boto3
import io


class NeuralNetworkModel(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    num_layers = models.IntegerField()
    layer_configurations = models.JSONField(
        help_text="JSON structure: [{'type': 'dense', 'in_features': 9, 'out_features': 128, 'activation': 'relu'}, ...]"
    )
    input_shape = models.CharField(max_length=255, help_text="e.g., '(9,)'")
    output_shape = models.CharField(max_length=255, help_text="e.g., '(2,)'")
    learning_rate = models.FloatField()
    batch_size = models.IntegerField()
    epochs = models.IntegerField()
    optimizer = models.CharField(
        max_length=50,
        choices=[("sgd", "SGD"), ("adam", "Adam")],
    )
    optimizer_params = models.JSONField(null=True, blank=True)
    trained_model_path = models.CharField(max_length=1024, null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Neural Network"
        verbose_name_plural = "Neural Networks"

    ### Workflow Integration ###

    def create_or_get_model(
        self,
        name: str,
        input_shape: str,
        output_shape: str,
        num_layers: int,
        layer_details: List[Dict[str, Any]],
        learning_rate: float,
        batch_size: int,
        epochs: int,
        optimizer: str = "adam",
        created_by: Optional[Any] = None,
    ) -> Any:
        """
        Create or retrieve a neural network configuration by name.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if model_instance:
            print(f"Model '{name}' already exists. Returning the existing model.")
            return model_instance

        # Validate layer configurations
        if len(layer_details) != num_layers:
            raise ValueError(
                "The number of layers does not match the layer details provided."
            )

        # Create the model instance
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.layer_configurations = layer_details
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.created_by = created_by
        self.save()
        print(f"Model '{name}' created and saved to the database.")
        return self

    def train_model_by_name(
        self, name: str, training_data: Tuple[np.ndarray, np.ndarray]
    ) -> Any:
        """
        Train the model specified by name.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if not model_instance:
            raise ValueError(f"No model found with the name '{name}'.")

        print(f"Training model '{name}'...")
        model_instance.train_model(*training_data)
        return model_instance

    def predict_by_name(self, name: str, input_data: List[float]) -> torch.Tensor:
        """
        Make predictions using the model specified by name.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if not model_instance:
            raise ValueError(f"No model found with the name '{name}'.")

        print(f"Making predictions with model '{name}'...")
        return model_instance.predict(input_data)

    ### Core Methods ###

    def create_model(self) -> nn.Module:
        """
        Creates a PyTorch model based on the configurations in the database.
        """
        layers = []
        input_size = eval(self.input_shape)[0]  # Use literal_eval for safety
        output_size = eval(self.output_shape)[0]
        last_size = input_size

        for config in self.layer_configurations:
            layers.append(nn.Linear(config["in_features"], config["out_features"]))
            if config["activation"] == "relu":
                layers.append(nn.ReLU())
            last_size = config["out_features"]

        layers.append(nn.Linear(last_size, output_size))
        return nn.Sequential(*layers)

    def get_test_data(
        self, datasource_id: str, table_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch test data from a DataSource.
        """
        datasource = DataSource.objects.get(id=datasource_id)
        data = datasource.query(f"SELECT * FROM {table_name};")

        if not data:
            raise ValueError(
                f"No data retrieved from table '{table_name}' in DataSource '{datasource.name}'."
            )

        inputs = np.array([row["input_data"] for row in data])
        outputs = np.array([row["output_data"] for row in data])
        return inputs, outputs

    def train_model(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """
        Train the model using the provided data.
        """
        model = self.create_model()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        inputs_tensor = torch.from_numpy(inputs).float()
        outputs_tensor = torch.from_numpy(outputs).float()

        num_batches = inputs_tensor.size(0) // self.batch_size

        for epoch in range(self.epochs):
            model.train()
            for i in range(num_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                batch_inputs = inputs_tensor[start:end]
                batch_outputs = outputs_tensor[start:end]

                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.6f}")

        self._save_trained_model(model)
        print("Training completed successfully.")

    def predict(self, input_data: List[float]) -> torch.Tensor:
        """
        Use the trained model to make predictions.
        """
        if not self.trained_model_path:
            raise ValueError("trained_model_path is not set. Train the model first.")

        buffer = self._download_from_minio(
            bucket_name="models", key=self.trained_model_path
        )
        model = self.create_model()
        model.load_state_dict(torch.load(buffer))
        model.eval()

        input_tensor = torch.tensor(input_data).float()
        with torch.no_grad():
            predictions = model(input_tensor)
        return predictions

    def _save_trained_model(self, model: nn.Module) -> None:
        """
        Save the trained model to MinIO.
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self._upload_to_minio(
            buffer, bucket_name="models", key=f"{self.name}_trained_model.pt"
        )
        self.trained_model_path = f"{self.name}_trained_model.pt"
        self.save()

    ### MinIO Helpers ###

    def _upload_to_minio(self, buffer: io.BytesIO, bucket_name: str, key: str) -> None:
        """
        Upload an in-memory buffer to MinIO.
        """
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        try:
            s3.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())
        except Exception as e:
            raise ValueError(f"Error uploading model to MinIO: {e}")

    def _download_from_minio(self, bucket_name: str, key: str) -> io.BytesIO:
        """
        Download a model from MinIO into an in-memory buffer.
        """
        buffer = io.BytesIO()
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        try:
            s3.download_fileobj(bucket_name, key, buffer)
            buffer.seek(0)
        except Exception as e:
            raise ValueError(f"Error downloading model from MinIO: {e}")
        return buffer
