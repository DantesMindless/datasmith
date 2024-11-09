from django.contrib.contenttypes.models import ContentType
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.forms.models import model_to_dict

from core.models import AuditModel, BaseModel


@receiver(post_save)
def post_save_handler(sender, instance, created, **kwargs):
    # Ensure the instance is a subclass of BaseModel
    if issubclass(sender, BaseModel) and created:
        # Create an audit entry for the newly created instance
        AuditModel.objects.create(
            objects_id=instance.id,
            content_type=ContentType.objects.get_for_model(sender),
            data=model_to_dict(
                instance, exclude=["id", "created_by", "created_at", "updated_at"]
            ),
            created_by=instance.created_by,
        )
