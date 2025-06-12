from django.conf import settings
import pandas as pd
import logging
from django.http import HttpResponse
from django.utils.html import escape
from django.middleware import csrf
from django.contrib import admin
from io import StringIO
from sklearn.model_selection import train_test_split
from app.models import ModelType, TrainingRun, ModelStatus
from app.models.main import MLModel
from .training import train_nn, train_sklearn_model
from .prediction import predict_nn, predict_sklearn_model


logger = logging.getLogger(__name__)


@admin.action(description="Train selected ML models")
def train_model(self, request, queryset):
    for obj in queryset:
        try:
            df = pd.read_csv(obj.dataset.file.path)
            target = obj.target_column
            config = obj.training_config or {}
            features = config.get("features") or df.drop(columns=[target]).columns.tolist()

            if df.isnull().values.any():
                raise ValueError("Dataset contains missing values.")
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found.")
            if not all(df[features].dtypes.apply(lambda dt: dt.kind in "biufc")):
                raise ValueError("Non-numeric values in input features.")

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=obj.test_size, random_state=obj.random_state
            )

            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.TRAINING)
            obj.status = ModelStatus.TRAINING
            obj.save()

            if obj.model_type == ModelType.NEURAL_NETWORK:
                model_path, acc = train_nn(obj, X_train, y_train, X_test, y_test)
            else:
                model_path, acc = train_sklearn_model(obj, X_train, y_train, X_test, y_test)

            obj.model_file.name = model_path.replace(settings.MEDIA_ROOT + "/", "")
            obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
            obj.status = ModelStatus.COMPLETE
            obj.save()

            run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)
            self.message_user(request, f"Model '{obj.name}' trained successfully.")

        except Exception as e:
            error_message = f"Training failed for '{obj.name}': {str(e)}"
            logger.error(error_message, exc_info=True)
            obj.status = ModelStatus.FAILED
            obj.training_log = error_message
            obj.save()
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
            self.message_user(request, error_message, level="error")


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
                raise ValueError("Prediction returned no result — check model compatibility and input features.")
            table_html = df.to_html(classes="table table-striped", index=False, escape=False)

            options_html = "\n".join([
                f'<option value="{m.pk}" {"selected" if m.pk == selected_model.pk else ""}>{escape(m.name)}</option>'
                for m in MLModel.objects.filter(status="complete")
            ])

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

    options_html = "\n".join([
        f'<option value="{m.pk}" {"selected" if m.pk == model.pk else ""}>{escape(m.name)}</option>'
        for m in MLModel.objects.filter(status="complete")
    ])

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
