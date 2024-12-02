from django.db import models
from core.models import BaseModel
from app.models.test import QuadraticTrainingData
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
        choices=[('sgd', 'SGD'), ('adam', 'Adam')],
    )
    optimizer_params = models.JSONField(null=True, blank=True)
    trained_model_path = models.CharField(max_length=1024, null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Neural Network"
        verbose_name_plural = "Neural Networks"

    ### Workflow Integration ###

    def create_or_get_model(self, name, input_shape, output_shape, num_layers, layer_details, learning_rate, batch_size, epochs, optimizer="adam", created_by=None):
        """
        Create or retrieve a neural network configuration by name.

        Args:
            name (str): Name of the neural network.
            input_shape (str): Shape of the input tensor, e.g., "(9,)".
            output_shape (str): Shape of the output tensor, e.g., "(2,)".
            num_layers (int): Number of layers in the neural network.
            layer_details (list[dict]): Layer configuration as a list of dictionaries.
            learning_rate (float): Learning rate for training.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            optimizer (str): Optimizer to use ("adam" or "sgd").

        Returns:
            NeuralNetworkModel: The existing or newly created model instance.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if model_instance:
            print(f"Model '{name}' already exists. Returning the existing model.")
            return model_instance

        # Validate layer configurations
        if len(layer_details) != num_layers:
            raise ValueError("The number of layers does not match the layer details provided.")

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

    def train_model_by_name(self, name, training_data):
        """
        Train the model specified by name.

        Args:
            name (str): The name of the neural network.
            training_data (tuple): A tuple containing inputs and outputs (X, y).

        Returns:
            NeuralNetworkModel: The trained model instance.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if not model_instance:
            raise ValueError(f"No model found with the name '{name}'.")

        # Train the model
        print(f"Training model '{name}'...")
        model_instance.train_model(training_data)
        return model_instance

    def predict_by_name(self, name, input_data):
        """
        Make predictions using the model specified by name.

        Args:
            name (str): The name of the neural network.
            input_data (list): A list of input features to predict.

        Returns:
            Tensor: The predicted output from the model.
        """
        model_instance = NeuralNetworkModel.objects.filter(name=name).first()
        if not model_instance:
            raise ValueError(f"No model found with the name '{name}'.")

        print(f"Making predictions with model '{name}'...")
        return model_instance.predict(input_data)

    ### Core Methods ###

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

    def get_test_data(self):
        """
        Fetch test data from the QuadraticTrainingData table in the test database.
        Returns:
            inputs (np.array): Input data as a NumPy array.
            outputs (np.array): Output data as a NumPy array.
        """
        data = QuadraticTrainingData.objects.using("test").all()
        inputs = []
        outputs = []

        for entry in data:
            inputs.append(entry.input_data)  # Assuming input_data is JSON
            outputs.append(entry.output_data)  # Assuming output_data is JSON

        return np.array(inputs), np.array(outputs)

    def train_model(self, datasource_id, table_name):
        """
        Train the model using data fetched from a DataSource.
        """
        datasource = DataSource.objects.get(id=datasource_id)
        data = datasource.query(f"SELECT * FROM {table_name}")

        inputs = np.array([list(row[:-1]) for row in data])
        outputs = np.array([row[-1] for row in data])

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
        
    def predict(self, input_data):
        """
        Use the trained model to make predictions.
        """
        if not self.trained_model_path:
            raise ValueError("trained_model_path is not set. Train the model first.")

        # Download the model from MinIO into an in-memory buffer
        buffer = self._download_from_minio(bucket_name="models", key=self.trained_model_path)

        # Load the model state from the buffer
        model = self.create_model()
        model.load_state_dict(torch.load(buffer))
        model.eval()

        # Prepare input data
        input_tensor = torch.from_numpy(np.array(input_data)).float()
        with torch.no_grad():
            predictions = model(input_tensor)
        return predictions
    def _save_trained_model(self, model):
            """
            Save the trained model to MinIO.
            """
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            self._upload_to_minio(buffer, bucket_name="models", key=f"{self.name}_trained_model.pt")
            self.trained_model_path = f"{self.name}_trained_model.pt"
            self.save()
            print("Model saved and uploaded to MinIO.")
    ### MinIO Helpers ###

    def _upload_to_minio(self, buffer, bucket_name, key):
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
            print(f"Model uploaded successfully to bucket '{bucket_name}' with key '{key}'.")
        except Exception as e:
            raise ValueError(f"Error uploading model to MinIO: {e}")

    def _download_from_minio(self, bucket_name, key):
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
            buffer.seek(0)  # Reset the buffer position
            print(f"Model downloaded successfully from bucket '{bucket_name}' with key '{key}'.")
        except Exception as e:
            raise ValueError(f"Error downloading model from MinIO: {e}")
        return buffer
    
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