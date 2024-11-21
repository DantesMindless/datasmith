from django.core.management.base import BaseCommand
from app.models import NeuralNetworkModel
from django.contrib.auth import get_user_model

# Get the custom user model
User = get_user_model()

class Command(BaseCommand):
    help = 'Create a sample neural network model'

    def handle(self, *args, **kwargs):
        # Get or create a default custom user
        user = User.objects.first()  # Adjust as needed

        if not user:
            # Create a default user if no users exist
            user = User.objects.create_user(username='default_user', email='default@example.com', password='password')

        NeuralNetworkModel.objects.create(
            name="Quadratic Root Predictor",
            num_layers=3,
            layer_configurations=[
                {"type": "dense", "in_features": 9, "out_features": 128, "activation": "relu"},
                {"type": "dense", "in_features": 128, "out_features": 64, "activation": "relu"},
                {"type": "dense", "in_features": 64, "out_features": 2, "activation": "linear"},
            ],
            input_shape="(9,)",
            output_shape="(2,)",
            learning_rate=0.001,
            batch_size=32,
            epochs=1000,
            optimizer="adam",
            created_by=user,  # Pass the custom user instance
        )
        self.stdout.write(self.style.SUCCESS("Neural network created successfully!"))
