from django.db import models


class QuadraticTrainingData(models.Model):
    input_data = models.JSONField(help_text="Precomputed input features for training")
    output_data = models.JSONField(help_text="Expected output values for training")

    def __str__(self) -> str:
        return f"Training Data: Inputs {self.input_data}, Outputs {self.output_data}"

    class Meta:
        db_table = (
            "test_quadratictrainingdata"  # Ensure the table name starts with 'test_'
        )
