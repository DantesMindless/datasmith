"""
Test fixtures and mock data generators for ML testing
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from unittest.mock import Mock, MagicMock, patch
from django.core.files.uploadedfile import SimpleUploadedFile
from app.models.choices import DatasetType, DatasetPurpose, DataQuality, ModelType


class MockDataGenerator:
    """Generate mock data for ML tests"""

    @staticmethod
    def create_tabular_classification_data(n_samples=100, n_features=5, n_classes=2):
        """Create mock tabular classification dataset"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y

        return df, feature_cols, 'target'

    @staticmethod
    def create_tabular_regression_data(n_samples=100, n_features=5, target_range=(0, 1000)):
        """Create mock tabular regression dataset with continuous target"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        # Create continuous target variable (e.g., house prices, temperatures)
        y = np.random.uniform(target_range[0], target_range[1], n_samples)
        # Add some correlation with features
        y = y + X[:, 0] * 50 + X[:, 1] * 30

        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y

        return df, feature_cols, 'target'

    @staticmethod
    def create_housing_data(n_samples=500):
        """Create realistic housing price regression data"""
        np.random.seed(42)

        # Generate features
        sqft = np.random.uniform(800, 4000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age = np.random.randint(0, 50, n_samples)
        garage = np.random.randint(0, 3, n_samples)

        # Generate target (price) with some correlation
        price = (sqft * 150 + bedrooms * 20000 + bathrooms * 15000
                 - age * 1000 + garage * 10000 + np.random.randn(n_samples) * 20000)
        price = np.maximum(price, 50000)  # Minimum price

        df = pd.DataFrame({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'garage': garage,
            'price': price
        })

        feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'garage']
        return df, feature_cols, 'price'

    @staticmethod
    def create_multiclass_data(n_samples=150, n_features=5, n_classes=5):
        """Create multi-class classification data"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y

        return df, feature_cols, 'target'

    @staticmethod
    def create_image_data(n_images=10, img_size=(64, 64), n_classes=2):
        """Create mock image data as PIL Images"""
        np.random.seed(42)
        images = []
        labels = []

        for i in range(n_images):
            # Create random image
            img_array = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            images.append(img)
            labels.append(i % n_classes)

        return images, labels

    @staticmethod
    def create_clustering_data(n_samples=100, n_features=5, n_clusters=3):
        """Create mock data for clustering"""
        np.random.seed(42)

        # Create data with natural clusters
        cluster_centers = np.random.randn(n_clusters, n_features) * 10
        X = []
        true_labels = []

        samples_per_cluster = n_samples // n_clusters
        remaining_samples = n_samples % n_clusters

        for i, center in enumerate(cluster_centers):
            n_cluster_samples = samples_per_cluster + (1 if i < remaining_samples else 0)
            cluster_data = center + np.random.randn(n_cluster_samples, n_features)
            X.extend(cluster_data)
            true_labels.extend([i] * n_cluster_samples)

        X = np.array(X)

        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)

        return df, feature_cols, np.array(true_labels)


class MockModelFactory:
    """Factory for creating mock Django model instances"""

    @staticmethod
    def create_mock_dataset(
        dataset_type=DatasetType.TABULAR,
        dataset_purpose=DatasetPurpose.CLASSIFICATION,
        row_count=100,
        column_count=5,
        data_quality=DataQuality.GOOD,
        column_info=None,
        file_name='test_data.csv',
        minio_images_prefix=None
    ):
        """Create a mock Dataset model instance"""
        dataset = Mock()
        dataset.id = 'test-dataset-123'
        dataset.name = 'Test Dataset'
        dataset.dataset_type = dataset_type
        dataset.dataset_purpose = dataset_purpose
        dataset.row_count = row_count
        dataset.column_count = column_count
        dataset.data_quality = data_quality
        dataset.column_info = column_info or {}
        dataset.file = Mock()
        dataset.file.name = file_name
        dataset.file.path = f'/tmp/{file_name}'
        dataset.minio_images_prefix = minio_images_prefix
        dataset.extracted_path = None

        # Mock the file read method
        if dataset_type == DatasetType.TABULAR:
            if dataset_purpose == DatasetPurpose.REGRESSION:
                df, features, target = MockDataGenerator.create_tabular_regression_data(
                    n_samples=row_count,
                    n_features=column_count - 1
                )
            else:
                df, features, target = MockDataGenerator.create_tabular_classification_data(
                    n_samples=row_count,
                    n_features=column_count - 1
                )
            dataset.file.read = Mock(return_value=df.to_csv(index=False).encode())

        return dataset

    @staticmethod
    def create_serialized_sklearn_model(model_type=ModelType.RANDOM_FOREST):
        """Create a real serialized sklearn model for testing"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier

        # Create a simple trained model
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 0, 1, 0])
        y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        model_map = {
            ModelType.RANDOM_FOREST: RandomForestClassifier(n_estimators=10, random_state=42),
            ModelType.LOGISTIC_REGRESSION: LogisticRegression(max_iter=100),
            ModelType.DECISION_TREE: DecisionTreeClassifier(random_state=42),
            ModelType.SVM: SVC(max_iter=100),
            ModelType.NAIVE_BAYES: GaussianNB(),
            ModelType.KNN: KNeighborsClassifier(n_neighbors=2),
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor(n_estimators=10, random_state=42),
        }

        clf = model_map.get(model_type, RandomForestClassifier(n_estimators=10, random_state=42))

        # Train on appropriate target
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST_REGRESSOR]:
            clf.fit(X, y_reg)
        else:
            clf.fit(X, y)

        # Serialize to bytes
        buffer = BytesIO()
        joblib.dump(clf, buffer)
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def create_mock_mlmodel(
        model_type=ModelType.RANDOM_FOREST,
        dataset=None,
        target_column='target',
        training_config=None,
        status='pending',
        with_trained_model=False
    ):
        """Create a mock MLModel model instance"""
        model = Mock()
        model.id = 'test-model-123'
        model.name = 'Test Model'
        model.model_type = model_type
        model.dataset = dataset or MockModelFactory.create_mock_dataset()
        model.target_column = target_column
        model.training_config = training_config or {}
        model.status = status
        model.test_size = 0.2
        model.random_state = 42
        model.max_iter = 100
        model.accuracy = None
        model.training_log = ''

        # Create a mock file object with optional real model data
        mock_file = Mock()
        if with_trained_model:
            model_bytes = MockModelFactory.create_serialized_sklearn_model(model_type)
            mock_file.read = Mock(return_value=model_bytes)
        else:
            mock_file.read = Mock(return_value=b'')
        model.model_file = mock_file

        model.prediction_schema = None
        model.analytics_data = {}

        # Mock save method
        model.save = Mock()

        return model

    @staticmethod
    def create_mock_training_run(model=None):
        """Create a mock TrainingRun model instance"""
        training_run = Mock()
        training_run.model = model or MockModelFactory.create_mock_mlmodel()
        training_run.history = []
        training_run.save = Mock()

        def add_entry(entry):
            training_run.history.append(entry)

        training_run.add_entry = add_entry

        return training_run


class MockMinIOStorage:
    """Mock MinIO storage operations"""

    @staticmethod
    def mock_upload_to_minio(file_content, file_name, **kwargs):
        """Mock MinIO upload - returns fake file path"""
        return f'models/{file_name}'

    @staticmethod
    def mock_download_from_minio(file_path):
        """Mock MinIO download - returns BytesIO"""
        return BytesIO(b'mock_model_data')


class MockTorchData:
    """Mock PyTorch data utilities"""

    @staticmethod
    def create_mock_dataloader(n_samples=10, n_features=5, n_classes=2, batch_size=2):
        """Create a mock PyTorch DataLoader"""
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    @staticmethod
    def create_mock_image_dataloader(n_images=10, img_size=(64, 64, 3), n_classes=2, batch_size=2):
        """Create a mock image DataLoader"""
        # Images shape: (batch, channels, height, width)
        X = torch.randn(n_images, img_size[2], img_size[0], img_size[1])
        y = torch.randint(0, n_classes, (n_images,))

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader


class MockTrainingLogger:
    """Mock TrainingLogger for tests"""

    def __init__(self, model=None):
        self.model = model
        self.logs = []
        self.metrics = {
            'losses': [],
            'accuracies': [],
            'val_accuracies': []
        }

    def start_training(self, config):
        self.logs.append(f"Training started with config: {config}")

    def log_info(self, message):
        self.logs.append(message)

    def log_data_loading(self, info):
        self.logs.append(f"Data loaded: {info}")

    def log_data_split(self, train_size, test_size):
        self.logs.append(f"Data split: train={train_size}, test={test_size}")

    def log_preprocessing(self, steps):
        self.logs.append(f"Preprocessing: {steps}")

    def log_model_architecture(self, architecture):
        self.logs.append(f"Architecture: {architecture}")

    def log_training_start(self, total_epochs=1):
        self.logs.append(f"Training started for {total_epochs} epochs")

    def log_epoch_start(self, epoch, total_epochs):
        self.logs.append(f"Epoch {epoch}/{total_epochs} started")

    def log_epoch_end(self, epoch, total_epochs, loss, acc, val_loss=None, val_acc=None, lr=None, epoch_time=None):
        self.logs.append(f"Epoch {epoch}/{total_epochs} complete: loss={loss}, acc={acc}")
        self.metrics['losses'].append(loss)
        self.metrics['accuracies'].append(acc)
        if val_acc:
            self.metrics['val_accuracies'].append(val_acc)

    def log_batch_progress(self, batch, total_batches, loss, metrics=None):
        pass  # Don't log every batch

    def log_validation_start(self):
        self.logs.append("Validation started")

    def log_evaluation_results(self, results):
        self.logs.append(f"Evaluation: {results}")

    def log_model_save(self, path, size):
        self.logs.append(f"Model saved to {path} ({size} bytes)")

    def log_training_complete(self, accuracy):
        self.logs.append(f"Training complete with accuracy: {accuracy}")

    def log_error(self, error, context=""):
        self.logs.append(f"Error {context}: {error}")

    def _format_duration(self, seconds):
        return f"{seconds:.2f}s"


def create_temp_csv_file(df, filename='test.csv'):
    """Create a temporary CSV file from DataFrame"""
    csv_content = df.to_csv(index=False).encode()
    return SimpleUploadedFile(filename, csv_content, content_type='text/csv')


def create_temp_image_file(img, filename='test.jpg'):
    """Create a temporary image file from PIL Image"""
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return SimpleUploadedFile(filename, buffer.read(), content_type='image/jpeg')
