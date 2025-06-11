from io import StringIO
from django.http import HttpResponse
import os
import pandas as pd
import joblib
from django.contrib import admin
from django.utils.html import format_html, format_html_join, escape
from django.utils.safestring import mark_safe
from django.conf import settings
from django.middleware import csrf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import json
from app.functions.nn import train_neural_network 
from app.models.main import ModelStatus
from .models import Dataset, MLModel, TrainingRun, ModelType

logger = logging.getLogger(__name__)


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
    actions = ["train_model", "make_prediction"]

    @admin.action(description="Train selected ML models")
    def train_model(self, request, queryset):
        for obj in queryset:
            dataset_path = obj.dataset.file.path
            target_col = obj.target_column

            run, _ = TrainingRun.objects.get_or_create(model=obj)
            run.add_entry(status=ModelStatus.TRAINING)
            obj.status = ModelStatus.TRAINING
            obj.save()

            try:
                df = pd.read_csv(dataset_path)

                # CSV validation
                if df.isnull().values.any():
                    raise ValueError("Dataset contains missing values.")

                if target_col not in df.columns:
                    raise ValueError(
                        f"Target column '{target_col}' not found in dataset."
                    )
                config = obj.training_config or {}
                features = config.get("features")
                normalize = config.get("normalize", False)
                if features:
                    X = df[features]
                else:
                    X = df.drop(columns=[target_col])
                y = df[target_col]

                if not all(X.dtypes.apply(lambda dt: dt.kind in "biufc")):
                    raise ValueError("Non-numeric values found in input features.")

                # Split data using config values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=obj.test_size, random_state=obj.random_state
                )

                # Dynamic model selection
                def get_model_instance(algorithm: str, obj: MLModel):
                    if algorithm == ModelType.LOGISTIC_REGRESSION:
                        return LogisticRegression(max_iter=obj.max_iter)
                    elif algorithm == ModelType.RANDOM_FOREST:
                        return RandomForestClassifier(random_state=obj.random_state)
                    elif algorithm == ModelType.SVM:
                        return SVC(max_iter=obj.max_iter)
                    elif algorithm == ModelType.NAIVE_BAYES:
                        return GaussianNB()
                    elif algorithm == ModelType.KNN:
                        return KNeighborsClassifier()
                    elif algorithm == ModelType.GRADIENT_BOOSTING:
                        return GradientBoostingClassifier()
                    else:
                        raise ValueError(f"Unsupported model type: {algorithm}")

                if obj.model_type == ModelType.NEURAL_NETWORK:
                    model_path, acc = train_neural_network(obj, X_train, y_train, X_test, y_test)
                    obj.model_file.name = model_path.replace(settings.MEDIA_ROOT + "/", "")
                else:
                    clf = get_model_instance(obj.model_type, obj)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    # Save scikit-learn model
                    model_dir = os.path.join(settings.MEDIA_ROOT, "trained_models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"{obj.id}.joblib")
                    joblib.dump(clf, model_path)
                    obj.model_file.name = f"trained_models/{obj.id}.joblib"

                    # Save trained model
                    model_dir = os.path.join(settings.MEDIA_ROOT, "trained_models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"{obj.id}.joblib")
                    joblib.dump(clf, model_path)

                    # Update MLModel
                    obj.model_file.name = f"trained_models/{obj.id}.joblib"
                    
                obj.training_log = f"Training complete. Accuracy: {acc:.2f}"
                obj.status = ModelStatus.COMPLETE
                obj.save()

                # Update TrainingRun
                run.add_entry(status=ModelStatus.COMPLETE, accuracy=acc)

                self.message_user(request, f"Model '{obj.name}' trained successfully.")

            except Exception as e:
                error_message = f"Training failed for '{obj.name}': {str(e)}"

                obj.training_log = error_message
                obj.status = ModelStatus.FAILED
                obj.save()

                run.add_entry(status=ModelStatus.FAILED, error=str(e))

                logger.error(error_message, exc_info=True)  # 🔥 logs full traceback
                self.message_user(request, error_message, level="error")

    @admin.action(description="Upload CSV and Predict")
    def make_prediction(self, request, queryset):
        if "apply" in request.POST:
            model = queryset.first()
            csv_text = request.POST.get("csv_input")

            try:
                df = pd.read_csv(StringIO(csv_text))
                model_path = model.model_file.path
                clf = joblib.load(model_path)

                config = model.training_config or {}
                features = config.get("features")
                if not features:
                    features = df.columns.tolist()

                X = df[features]
                preds = clf.predict(X)
                df["prediction"] = preds

                table_html = df.to_html(
                    classes="table table-striped", index=False, escape=False
                )
                token = csrf.get_token(request)
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
                        <textarea name="csv_input" rows="10" cols="80">{escape(csv_text)}</textarea><br>
                        <button type="submit">Predict Again</button>
                    </form>
                    <h3>Prediction Results:</h3>
                    {table_html}
                """)

            except Exception as e:
                self.message_user(
                    request, f"Prediction failed: {str(e)}", level="error"
                )
                return None

        else:
            token = csrf.get_token(request)
            model = queryset.first()
            return HttpResponse(f"""
                <form method="post">
                    <input type="hidden" name="csrfmiddlewaretoken" value="{token}">
                    <input type="hidden" name="action" value="make_prediction">
                    <input type="hidden" name="_selected_action" value="{model.pk}">
                    <input type="hidden" name="apply" value="1">
                    <textarea name="csv_input" rows="10" cols="80" placeholder="Paste CSV test data here"></textarea><br>
                    <button type="submit">Predict</button>
                </form>
            """)

    def response_action(self, request, queryset):
        """
        Override to allow custom non-redirect behavior from make_prediction.
        """
        if request.POST.get("action") == "make_prediction":
            return self.make_prediction(request, queryset)
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

        # Format each entry as an indented JSON block
        formatted = [
            f"<pre>{json.dumps(entry, indent=2)}</pre>"
            for entry in obj.history
        ]
        return mark_safe("".join(formatted))

    pretty_history.short_description = "Training History"
