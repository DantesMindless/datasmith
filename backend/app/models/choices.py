from django.db import models


class ModelStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    TRAINING = "training", "Training"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class ModelType(models.TextChoices):
    # Traditional ML Models
    LOGISTIC_REGRESSION = "logistic_regression", "Logistic Regression"
    DECISION_TREE = "decision_tree", "Decision Tree"
    RANDOM_FOREST = "random_forest", "Random Forest"
    SVM = "svm", "Support Vector Machine"
    NAIVE_BAYES = "naive_bayes", "Naive Bayes"
    KNN = "knn", "k-Nearest Neighbours"

    # Ensemble Models
    GRADIENT_BOOSTING = "gradient_boosting", "Gradient Boosting (sklearn)"
    XGBOOST = "xgboost", "XGBoost"
    LIGHTGBM = "lightgbm", "LightGBM"
    ADABOOST = "adaboost", "AdaBoost"

    # Deep Learning - General
    NEURAL_NETWORK = "neural_network", "Neural Network (PyTorch)"

    # Deep Learning - Computer Vision
    CNN = "cnn", "Convolutional Neural Network (CNN)"
    RESNET = "resnet", "ResNet (Transfer Learning)"
    VGG = "vgg", "VGG (Transfer Learning)"
    EFFICIENTNET = "efficientnet", "EfficientNet (Transfer Learning)"

    # Deep Learning - Sequential Data
    RNN = "rnn", "Recurrent Neural Network (RNN)"
    LSTM = "lstm", "Long Short-Term Memory (LSTM)"
    GRU = "gru", "Gated Recurrent Unit (GRU)"


class ActivationFunction(models.TextChoices):
    RELU = "relu", "ReLU"
    TANH = "tanh", "Tanh"
    SIGMOID = "sigmoid", "Sigmoid"
    LEAKY_RELU = "leaky_relu", "Leaky ReLU"


class DatasetType(models.TextChoices):
    """Dataset type classification"""
    TABULAR = "tabular", "Tabular Data"
    IMAGE = "image", "Image Dataset"
    TEXT = "text", "Text Dataset"
    TIME_SERIES = "time_series", "Time Series"
    MIXED = "mixed", "Mixed Dataset"


class DatasetPurpose(models.TextChoices):
    """Dataset intended use purpose"""
    CLASSIFICATION = "classification", "Classification"
    REGRESSION = "regression", "Regression"
    CLUSTERING = "clustering", "Clustering"
    ANOMALY_DETECTION = "anomaly_detection", "Anomaly Detection"
    RECOMMENDATION = "recommendation", "Recommendation"
    GENERAL = "general", "General Purpose"


class DataQuality(models.TextChoices):
    """Dataset quality assessment"""
    EXCELLENT = "excellent", "Excellent (>95% complete)"
    GOOD = "good", "Good (85-95% complete)"
    FAIR = "fair", "Fair (70-85% complete)"
    POOR = "poor", "Poor (<70% complete)"
    UNKNOWN = "unknown", "Unknown"


class ColumnType(models.TextChoices):
    """Column data types"""
    NUMERIC = "numeric", "Numeric"
    CATEGORICAL = "categorical", "Categorical"
    TEXT = "text", "Text"
    DATETIME = "datetime", "DateTime"
    BOOLEAN = "boolean", "Boolean"
    IMAGE_PATH = "image_path", "Image Path"
    URL = "url", "URL"
    EMAIL = "email", "Email"
    UNKNOWN = "unknown", "Unknown"
