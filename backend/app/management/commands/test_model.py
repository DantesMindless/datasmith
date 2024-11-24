from django.core.management.base import BaseCommand
from app.models import NeuralNetworkModel
import torch
import torch.nn as nn
import os
import json
import numpy as np
import boto3

class Command(BaseCommand):
    help = "Test a trained neural network model by calculating quadratic roots"

    def add_arguments(self, parser):
        parser.add_argument("a", type=float, help="Coefficient a of the quadratic equation")
        parser.add_argument("b", type=float, help="Coefficient b of the quadratic equation")
        parser.add_argument("c", type=float, help="Coefficient c of the quadratic equation")

    def handle(self, *args, **kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]

        # Fetch NeuralNetworkModel
        try:
            nn_model = NeuralNetworkModel.objects.first()
            if not nn_model:
                self.stderr.write("No NeuralNetworkModel found in the database.")
                return
        except Exception as e:
            self.stderr.write(f"Error fetching NeuralNetworkModel: {e}")
            return

        # Build model from configuration
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

        # Load the model from MinIO
        temp_path = "trained_model.pt"
        if not self.download_from_minio(temp_path, "models", "trained_model.pt"):
            self.stderr.write(f"Trained model file not found in MinIO. Please train the model first.")
            return
        model.load_state_dict(torch.load(temp_path))
        model.eval()

        ab = a * b
        ac = a * c
        bc = b * c
        a_squared = a ** 2
        b_squared = b ** 2
        c_squared = c ** 2

        input_features = np.array([[a, b, c, ab, ac, bc, a_squared, b_squared, c_squared]])
        input_tensor = torch.from_numpy(input_features).float()

        with torch.no_grad():
            predictions = model(input_tensor)

        x1, x2 = predictions[0].tolist()

        self.stdout.write(f"For the quadratic equation: {a}x² + {b}x + {c} = 0")
        self.stdout.write(f"Predicted roots: x1 = {x1:.6f}, x2 = {x2:.6f}")

    def download_from_minio(self, file_path, bucket_name, key):
        s3 = boto3.client(
            's3',
            endpoint_url="http://minio:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        try:
            s3.download_file(bucket_name, key, file_path)
            return True
        except Exception as e:
            self.stderr.write(f"Error downloading model from MinIO: {e}")
            return False
