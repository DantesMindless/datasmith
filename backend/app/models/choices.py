from django.db import models


class ModelStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    TRAINING = "training", "Training"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class ModelType(models.TextChoices):
    # ============ CLASSIFICATION MODELS ============
    # Traditional Classification
    LOGISTIC_REGRESSION = "logistic_regression", "Logistic Regression"
    DECISION_TREE = "decision_tree", "Decision Tree Classifier"
    RANDOM_FOREST = "random_forest", "Random Forest Classifier"
    SVM = "svm", "Support Vector Classifier (SVC)"
    NAIVE_BAYES = "naive_bayes", "Naive Bayes"
    KNN = "knn", "K-Nearest Neighbors Classifier"

    # Ensemble Classification
    GRADIENT_BOOSTING = "gradient_boosting", "Gradient Boosting Classifier"
    XGBOOST = "xgboost", "XGBoost Classifier"
    LIGHTGBM = "lightgbm", "LightGBM Classifier"
    ADABOOST = "adaboost", "AdaBoost Classifier"

    # ============ REGRESSION MODELS ============
    # Traditional Regression
    LINEAR_REGRESSION = "linear_regression", "Linear Regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor", "Decision Tree Regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor", "Random Forest Regressor"
    SVR = "svr", "Support Vector Regressor (SVR)"
    KNN_REGRESSOR = "knn_regressor", "K-Nearest Neighbors Regressor"

    # Ensemble Regression
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor", "Gradient Boosting Regressor"
    XGBOOST_REGRESSOR = "xgboost_regressor", "XGBoost Regressor"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor", "LightGBM Regressor"
    ADABOOST_REGRESSOR = "adaboost_regressor", "AdaBoost Regressor"

    # ============ DEEP LEARNING MODELS ============
    NEURAL_NETWORK = "neural_network", "Neural Network (PyTorch)"
    CNN = "cnn", "Convolutional Neural Network (CNN)"

    # ============ CLUSTERING MODELS ============
    KMEANS = "kmeans", "K-Means Clustering"
    DBSCAN = "dbscan", "DBSCAN Clustering"
    HIERARCHICAL = "hierarchical", "Hierarchical Clustering"
    GAUSSIAN_MIXTURE = "gaussian_mixture", "Gaussian Mixture Model"
    MEAN_SHIFT = "mean_shift", "Mean Shift Clustering"


# Helper lists for model categorization
CLASSIFICATION_MODELS = [
    ModelType.LOGISTIC_REGRESSION,
    ModelType.DECISION_TREE,
    ModelType.RANDOM_FOREST,
    ModelType.SVM,
    ModelType.NAIVE_BAYES,
    ModelType.KNN,
    ModelType.GRADIENT_BOOSTING,
    ModelType.XGBOOST,
    ModelType.LIGHTGBM,
    ModelType.ADABOOST,
]

REGRESSION_MODELS = [
    ModelType.LINEAR_REGRESSION,
    ModelType.DECISION_TREE_REGRESSOR,
    ModelType.RANDOM_FOREST_REGRESSOR,
    ModelType.SVR,
    ModelType.KNN_REGRESSOR,
    ModelType.GRADIENT_BOOSTING_REGRESSOR,
    ModelType.XGBOOST_REGRESSOR,
    ModelType.LIGHTGBM_REGRESSOR,
    ModelType.ADABOOST_REGRESSOR,
]

DEEP_LEARNING_MODELS = [
    ModelType.NEURAL_NETWORK,
    ModelType.CNN,
]

CLUSTERING_MODELS = [
    ModelType.KMEANS,
    ModelType.DBSCAN,
    ModelType.HIERARCHICAL,
    ModelType.GAUSSIAN_MIXTURE,
    ModelType.MEAN_SHIFT,
]


class ActivationFunction(models.TextChoices):
    RELU = "relu", "ReLU"
    TANH = "tanh", "Tanh"
    SIGMOID = "sigmoid", "Sigmoid"
    LEAKY_RELU = "leaky_relu", "Leaky ReLU"


class DatasetType(models.TextChoices):
    """Dataset type classification"""
    TABULAR = "tabular", "Tabular Data"
    IMAGE = "image", "Image Dataset"


class DatasetPurpose(models.TextChoices):
    """Dataset intended use purpose"""
    CLASSIFICATION = "classification", "Classification"
    REGRESSION = "regression", "Regression"
    CLUSTERING = "clustering", "Clustering"
    SEGMENTATION = "segmentation", "Segmentation"
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
