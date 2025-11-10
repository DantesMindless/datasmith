# Generated manually for analytics_data field
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0016_alter_mlmodel_model_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='mlmodel',
            name='analytics_data',
            field=models.JSONField(blank=True, default=dict, help_text='Detailed analytics data including metrics, confusion matrix, feature importance, etc.'),
        ),
    ]
