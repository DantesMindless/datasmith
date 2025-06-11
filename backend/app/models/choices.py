from django.db import models


class ModelStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    TRAINING = "training", "Training"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class ModelType(models.TextChoices):
    LOGISTIC_REGRESSION = "logistic_regression", "Logistic Regression"
    RANDOM_FOREST = "random_forest", "Random Forest"
    SVM = "svm", "Support Vector Machine"
    NAIVE_BAYES = "naive_bayes", "Naive Bayes"
    KNN = "knn", "k-Nearest Neighbours"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING", "Gradient Boosting"
    NEURAL_NETWORK = "neural_network", "Neural Network (PyTorch)"


class ActivationFunction(models.TextChoices):
    RELU = "relu", "ReLU"
    TANH = "tanh", "Tanh"
    SIGMOID = "sigmoid", "Sigmoid"
    LEAKY_RELU = "leaky_relu", "Leaky ReLU"
