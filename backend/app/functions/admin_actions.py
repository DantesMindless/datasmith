import pandas as pd
import logging
from django.http import HttpResponse
from django.utils.html import escape
from django.middleware import csrf
from django.contrib import admin
from io import StringIO
from app.models import ModelType, TrainingRun, ModelStatus
from app.models.main import MLModel
from app.functions.celery_tasks import train_cnn_task, train_nn_task, train_sklearn_task
from .prediction import predict_nn, predict_sklearn_model


logger = logging.getLogger(__name__)


@admin.action(description="Train selected ML models")
def train_model(self, request, queryset):
    for obj in queryset:
        try:
            obj.status = ModelStatus.TRAINING
            obj.save()

            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.TRAINING)

            if obj.dataset.is_image_dataset:
                train_cnn_task.delay(obj.id)
            elif obj.model_type == ModelType.NEURAL_NETWORK:
                train_nn_task.delay(obj.id)
            else:
                train_sklearn_task.delay(obj.id)

            self.message_user(
                request, f"Model '{obj.name}' training has started (async via Celery)."
            )

        except Exception as e:
            logger.error(
                f"Failed to queue training task for '{obj.name}': {e}", exc_info=True
            )
            obj.status = ModelStatus.FAILED
            obj.training_log = f"Error queuing training: {e}"
            obj.save()
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
            self.message_user(
                request,
                f"Error: Failed to start training for '{obj.name}': {e}",
                level="error",
            )


@admin.action(description="Upload CSV and Predict")
def make_prediction(self, request, queryset):
    model = queryset.first()
    token = csrf.get_token(request)
    if "apply" in request.POST:
        csv_text = request.POST.get("csv_input")
        selected_model_id = request.POST.get("model_choice")

        try:
            selected_model = MLModel.objects.get(pk=selected_model_id)
            df = pd.read_csv(StringIO(csv_text))

            if selected_model.model_type == ModelType.NEURAL_NETWORK:
                df = predict_nn(selected_model, df)
            else:
                df = predict_sklearn_model(selected_model, df)
            if df is None:
                raise ValueError(
                    "Prediction returned no result — check model compatibility and input features."
                )
            table_html = df.to_html(
                classes="table table-striped", index=False, escape=False
            )

            options_html = "\n".join(
                [
                    f'<option value="{m.pk}" {"selected" if m.pk == selected_model.pk else ""}>{escape(m.name)}</option>'
                    for m in MLModel.objects.filter(status="complete")
                ]
            )

            return HttpResponse(f"""
                <style>
                    .table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
                    .table th, .table td {{ border: 1px solid #ccc; padding: 6px 10px; }}
                    .table th {{ background-color: #f2f2f2; }}
                </style>
                <form method="post">
                    <input type="hidden" name="csrfmiddlewaretoken" value="{token}">
                    <input type="hidden" name="action" value="make_prediction">
                    <input type="hidden" name="_selected_action" value="{model.pk}">
                    <input type="hidden" name="apply" value="1">
                    <label for="model_choice">Choose model for prediction:</label>
                    <select name="model_choice" id="model_choice">
                        {options_html}
                    </select><br><br>
                    <textarea name="csv_input" rows="10" cols="80">{escape(csv_text)}</textarea><br>
                    <button type="submit">Predict Again</button>
                </form>
                <h3>Prediction Results from model: <code>{escape(selected_model.name)}</code></h3>
                {table_html}
            """)

        except Exception as e:
            self.message_user(request, f" Prediction failed: {str(e)}", level="error")
            return None

    options_html = "\n".join(
        [
            f'<option value="{m.pk}" {"selected" if m.pk == model.pk else ""}>{escape(m.name)}</option>'
            for m in MLModel.objects.filter(status="complete")
        ]
    )

    return HttpResponse(f"""
        <form method="post">
            <input type="hidden" name="csrfmiddlewaretoken" value="{token}">
            <input type="hidden" name="action" value="make_prediction">
            <input type="hidden" name="_selected_action" value="{model.pk}">
            <input type="hidden" name="apply" value="1">
            <label for="model_choice">Choose model for prediction:</label>
            <select name="model_choice" id="model_choice">
                {options_html}
            </select><br><br>
            <textarea name="csv_input" rows="10" cols="80" placeholder="Paste CSV test data here"></textarea><br>
            <button type="submit">Predict</button>
        </form>
    """)
