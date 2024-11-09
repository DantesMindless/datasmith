from typing import Iterable

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.db import models


class AuditModel(models.Model):
    """
    AuditModel is a Django model that represents an audit log entry.

    Attributes:
        objects_id (PositiveIntegerField): The ID of the object being audited.
        content_type (ForeignKey): A reference to the ContentType of the object being audited.
        data (JSONField): A JSON field to store additional data related to the audit entry.
        created_by (ForeignKey): A reference to the User who created the audit entry.
        created_at (DateTimeField): The timestamp when the audit entry was created.

    Meta:
        verbose_name (str): The human-readable name for the model.
        verbose_name_plural (str): The human-readable plural name for the model.
    """

    objects_id = models.PositiveIntegerField()
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    data = models.JSONField()
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="%(class)s_created_by"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Audit"
        verbose_name_plural = "Audits"


class BaseModel(models.Model):
    """
    BaseModel is an abstract base class for other models to inherit from. It provides
    common fields for tracking the creation and modification of model instances.

    Attributes:
        created_by (ForeignKey): A reference to the User who created the instance.
        deleted (BooleanField): A flag indicating whether the instance has been soft deleted.
        created_at (DateTimeField): The date and time when the instance was created.
        updated_at (DateTimeField): The date and time when the instance was last updated.

    Methods:
        soft_delete(): Marks the instance as deleted and saves the change to the database.

    Meta:
        abstract (bool): Indicates that this is an abstract base class and should not
        be used to create any database table.
    """

    deleted = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="%(class)s_created_by"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    def soft_delete(self):
        """
        Marks the object as deleted by setting the 'deleted' attribute to True and saves the change to the database.
        """
        self.deleted = True
        self.save()

    class Meta:
        abstract = True
