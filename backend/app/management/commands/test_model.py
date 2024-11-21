import torch
import numpy as np
from app.models import NeuralNetworkModel
from django.conf import settings
from django.core.management.base import BaseCommand
from storage import load_model_from_minio  # Utility for loading from MinIO

# Example function to calculate quadratic roots using a trained neural network
def predict_roots(model, scaler_inputs, scaler_outputs, a, b, c, device='cpu'):
    # Ensure inputs are in the correct format
    a = np.array([[a]])
    b = np.array([[b]])
    c = np.array([[c]])

    # Calculate additional features
    ab = a * b
    ac = a * c
    bc = b * c
    a_squared = a ** 2
    b_squared = b ** 2
    c_squared = c ** 2

    # Combine all features
    inputs = np.hstack([a, b, c, ab, ac, bc, a_squared, b_squared, c_squared])

    # Scale the inputs using the saved scaler
    inputs_scaled = scaler_inputs.transform(inputs)

    # Convert inputs to tensor
    inputs_tensor = torch.from_numpy(inputs_scaled).float().to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs_scaled = model(inputs_tensor)

    # Inverse transform the outputs to get the original scale
    outputs = scaler_outputs.inverse_transform(outputs_scaled.cpu().numpy())
    return outputs[0][0], outputs[0][1]  # Return roots x1, x2


class Command(BaseCommand):
    help = "Test a trained neural network for quadratic root prediction"

    def handle(self, *args, **kwargs):
        # Load the neural network model from MinIO or local storage
        model = load_model_from_minio("models", "Quadratic Root Predictor.pt")

        # Load scalers (assume they are saved locally or in MinIO)
        import joblib
        scaler_inputs = joblib.load("scaler_inputs.pkl")
        scaler_outputs = joblib.load("scaler_outputs.pkl")

        # Example coefficients (user input or test data)
        a, b, c = 1, -3, 2

        # Predict the roots
        root1, root2 = predict_roots(model, scaler_inputs, scaler_outputs, a, b, c)

        # Output the results
        print(f"For equation {a}x² + {b}x + {c} = 0:")
        print(f"Predicted roots are: x1 = {root1}, x2 = {root2}")
