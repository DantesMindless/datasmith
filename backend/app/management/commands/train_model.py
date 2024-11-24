from django.core.management.base import BaseCommand
from app.models import NeuralNetworkModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import boto3


class Command(BaseCommand):
    help = "Train a neural network model with coefficients between 1 and 5"

    def handle(self, *args, **kwargs):
        try:
            nn_model = NeuralNetworkModel.objects.first()
            if not nn_model:
                self.stderr.write("No NeuralNetworkModel found in the database.")
                return
        except Exception as e:
            self.stderr.write(f"Error fetching NeuralNetworkModel: {e}")
            return

        # Prepare model architecture
        layer_configs = json.loads(json.dumps(nn_model.layer_configurations))
        layers = []
        input_size = eval(nn_model.input_shape)[0]
        output_size = eval(nn_model.output_shape)[0]

        last_size = input_size
        for config in layer_configs:
            layers.append(nn.Linear(config["in_features"], config["out_features"]))
            if config["activation"] == "relu":
                layers.append(nn.ReLU())
            last_size = config["out_features"]
        layers.append(nn.Linear(last_size, output_size))

        model = nn.Sequential(*layers)

        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=nn_model.learning_rate)
        criterion = nn.MSELoss()

        # Generate training data
        num_samples = 10000
        np.random.seed(42)
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

        inputs_tensor = torch.from_numpy(inputs).float()
        outputs_tensor = torch.from_numpy(outputs).float()

        num_epochs = 200
        batch_size = nn_model.batch_size
        num_batches = inputs_tensor.size(0) // batch_size

        # Training loop
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
                self.stdout.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

        # Save the trained model locally
        temp_path = "/tmp/trained_model.pt"
        torch.save(model.state_dict(), temp_path)
        self.stdout.write(f"Model trained and saved at {temp_path}")

        # Upload the model to MinIO
        self.upload_to_minio(temp_path, "models", "trained_model.pt")

    def upload_to_minio(self, file_path, bucket_name, key):
        """
        Uploads the specified file to the MinIO bucket.
        """
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        try:
            s3.upload_file(file_path, bucket_name, key)
            self.stdout.write(f"File uploaded successfully to bucket '{bucket_name}' with key '{key}'.")
        except Exception as e:
            self.stderr.write(f"Error uploading file to MinIO: {e}")
