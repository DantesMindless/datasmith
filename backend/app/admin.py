from django.contrib import admin
from django.utils.safestring import mark_safe
from .models import Dataset, MLModel, TrainingRun, DataSegment, RowSegmentLabel
from app.functions.admin_actions import (
    make_prediction,
    train_model,
    predict_single_image,
)
import json


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "created_by", "created_at")


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "model_type",
        "status",
        "dataset",
        "created_by",
        "created_at",
    )
    actions = [train_model, make_prediction, predict_single_image]

    def response_action(self, request, queryset):
        """
        Override to allow custom non-redirect behavior from make_prediction.
        """
        if request.POST.get("action") == "make_prediction":
            return make_prediction(self, request, queryset)
        return super().response_action(request, queryset)


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = (
        "model",
        "pretty_history",
    )

    def pretty_history(self, obj):
        if not obj.history:
            return "(no history)"

        formatted = [
            f"<pre>{json.dumps(entry, indent=2)}</pre>" for entry in obj.history
        ]
        return mark_safe("".join(formatted))

    pretty_history.short_description = "Training History"


@admin.register(DataSegment)
class DataSegmentAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "dataset",
        "row_count",
        "color",
        "created_by",
        "created_at",
    )
    list_filter = ("dataset", "created_at")
    search_fields = ("name", "description")


@admin.register(RowSegmentLabel)
class RowSegmentLabelAdmin(admin.ModelAdmin):
    list_display = (
        "segment",
        "row_index",
        "assignment_method",
        "confidence",
        "assigned_by",
        "created_at",
    )
    list_filter = ("segment", "assignment_method", "assigned_by")
    search_fields = ("notes",)
