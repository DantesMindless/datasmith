"""
Model-Dataset Compatibility Validation System

This module provides advanced validation and recommendation for
matching machine learning models with appropriate datasets.
"""
from typing import Dict, List, Tuple, Optional
from app.models.choices import ModelType, DatasetType, DatasetPurpose, ColumnType


class ModelCompatibilityValidator:
    """
    Advanced validation system for model-dataset compatibility
    """

    # Define which model types work with which dataset types
    COMPATIBILITY_MATRIX = {
        # Traditional ML - works with tabular data
        ModelType.LOGISTIC_REGRESSION: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION],
            'min_samples': 50,
            'max_features': 1000,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'high',
        },
        ModelType.LINEAR_REGRESSION: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 30,
            'max_features': 1000,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'high',
        },
        ModelType.DECISION_TREE: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 20,
            'max_features': 500,
            'requires_numeric': False,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'high',
        },
        ModelType.RANDOM_FOREST: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 10000,
            'requires_numeric': False,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'medium',
        },
        ModelType.SVM: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 200,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'slow',
            'interpretability': 'low',
        },
        ModelType.NAIVE_BAYES: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION],
            'min_samples': 50,
            'max_features': 5000,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'medium',
        },
        ModelType.KNN: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION, DatasetPurpose.ANOMALY_DETECTION],
            'min_samples': 30,
            'max_features': 50,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'medium',
        },

        # Ensemble models
        ModelType.GRADIENT_BOOSTING: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 5000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'medium',
            'interpretability': 'low',
        },
        ModelType.XGBOOST: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 10000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'fast',
            'interpretability': 'low',
        },
        ModelType.LIGHTGBM: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 50000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'very_fast',
            'interpretability': 'low',
        },
        ModelType.ADABOOST: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 1000,
            'requires_numeric': False,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'medium',
        },

        # ============ REGRESSION MODELS ============
        ModelType.DECISION_TREE_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 20,
            'max_features': 500,
            'requires_numeric': False,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'high',
        },
        ModelType.RANDOM_FOREST_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 10000,
            'requires_numeric': False,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'medium',
        },
        ModelType.SVR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 200,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'slow',
            'interpretability': 'low',
        },
        ModelType.KNN_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 30,
            'max_features': 50,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'medium',
        },
        ModelType.GRADIENT_BOOSTING_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 5000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'medium',
            'interpretability': 'low',
        },
        ModelType.XGBOOST_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 10000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'fast',
            'interpretability': 'low',
        },
        ModelType.LIGHTGBM_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 200,
            'max_features': 50000,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'very_fast',
            'interpretability': 'low',
        },
        ModelType.ADABOOST_REGRESSOR: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.REGRESSION],
            'min_samples': 100,
            'max_features': 1000,
            'requires_numeric': False,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'medium',
        },

        # Deep learning - general
        ModelType.NEURAL_NETWORK: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION, DatasetPurpose.REGRESSION],
            'min_samples': 500,
            'max_features': 10000,
            'requires_numeric': True,
            'complexity': 'high',
            'training_speed': 'medium',
            'interpretability': 'low',
        },

        # Computer vision models
        ModelType.CNN: {
            'compatible_datasets': [DatasetType.IMAGE],
            'compatible_purposes': [DatasetPurpose.CLASSIFICATION],
            'min_samples': 100,
            'max_features': None,
            'requires_numeric': False,
            'complexity': 'high',
            'training_speed': 'slow',
            'interpretability': 'low',
        },

        # ============ CLUSTERING MODELS ============
        ModelType.KMEANS: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLUSTERING],
            'min_samples': 50,
            'max_features': 1000,
            'requires_numeric': True,
            'complexity': 'low',
            'training_speed': 'fast',
            'interpretability': 'high',
        },
        ModelType.DBSCAN: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLUSTERING, DatasetPurpose.ANOMALY_DETECTION],
            'min_samples': 50,
            'max_features': 100,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'fast',
            'interpretability': 'medium',
        },
        ModelType.HIERARCHICAL: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLUSTERING],
            'min_samples': 30,
            'max_features': 500,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'high',
        },
        ModelType.GAUSSIAN_MIXTURE: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLUSTERING, DatasetPurpose.ANOMALY_DETECTION],
            'min_samples': 100,
            'max_features': 500,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'medium',
            'interpretability': 'medium',
        },
        ModelType.MEAN_SHIFT: {
            'compatible_datasets': [DatasetType.TABULAR],
            'compatible_purposes': [DatasetPurpose.CLUSTERING],
            'min_samples': 100,
            'max_features': 50,
            'requires_numeric': True,
            'complexity': 'medium',
            'training_speed': 'slow',
            'interpretability': 'medium',
        },
    }

    @classmethod
    def validate_compatibility(cls, model_type: str, dataset) -> Dict:
        """
        Validate if a model type is compatible with a dataset

        Args:
            model_type: The model type to validate
            dataset: Dataset instance

        Returns:
            Dictionary with validation results:
            {
                'is_compatible': bool,
                'errors': List[str],
                'warnings': List[str],
                'recommendations': List[str]
            }
        """
        result = {
            'is_compatible': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'score': 100  # Compatibility score out of 100
        }

        if model_type not in cls.COMPATIBILITY_MATRIX:
            result['errors'].append(f"Unknown model type: {model_type}")
            result['is_compatible'] = False
            result['score'] = 0
            return result

        model_config = cls.COMPATIBILITY_MATRIX[model_type]

        # Check dataset type compatibility
        if dataset.dataset_type not in model_config['compatible_datasets']:
            result['errors'].append(
                f"{model_type} requires {' or '.join(model_config['compatible_datasets'])} data, "
                f"but dataset is {dataset.dataset_type}"
            )
            result['is_compatible'] = False
            result['score'] -= 50

        # Check dataset purpose compatibility
        if (dataset.dataset_purpose != DatasetPurpose.GENERAL and
            dataset.dataset_purpose not in model_config['compatible_purposes']):
            result['errors'].append(
                f"{model_type} is designed for {' or '.join(model_config['compatible_purposes'])}, "
                f"but dataset is for {dataset.dataset_purpose}"
            )
            result['is_compatible'] = False
            result['score'] -= 30

        # Check minimum samples
        if dataset.row_count and dataset.row_count < model_config['min_samples']:
            result['warnings'].append(
                f"{model_type} typically requires at least {model_config['min_samples']} samples, "
                f"but dataset has only {dataset.row_count}. Performance may be poor."
            )
            result['score'] -= 15

        # Check maximum features (for tabular data)
        if (dataset.dataset_type == DatasetType.TABULAR and
            model_config['max_features'] and
            dataset.column_count and
            dataset.column_count > model_config['max_features']):
            result['warnings'].append(
                f"{model_type} may struggle with {dataset.column_count} features. "
                f"Consider feature selection or dimensionality reduction."
            )
            result['score'] -= 10

        # Check for numeric requirement
        if (model_config['requires_numeric'] and
            dataset.dataset_type == DatasetType.TABULAR and
            dataset.column_info):
            non_numeric_cols = [
                col for col, info in dataset.column_info.items()
                if info.get('type') not in [ColumnType.NUMERIC, ColumnType.BOOLEAN]
            ]
            if non_numeric_cols:
                result['warnings'].append(
                    f"{model_type} works best with numeric features. "
                    f"You may need to encode these categorical columns: {', '.join(non_numeric_cols[:5])}"
                )
                result['score'] -= 5

        # Determine if this is an image model
        is_image_model = DatasetType.IMAGE in model_config['compatible_datasets']
        is_tabular_model = DatasetType.TABULAR in model_config['compatible_datasets']

        # Add context-appropriate recommendations
        if is_image_model:
            # Image model specific recommendations
            if dataset.row_count:
                num_classes = len(dataset.column_info) if dataset.column_info else 0
                # For image datasets, row_count typically represents number of images
                images_per_class = dataset.row_count / max(num_classes, 1) if num_classes > 0 else dataset.row_count

                if dataset.row_count < 100:
                    result['recommendations'].append(
                        f"With only {dataset.row_count} images, consider using data augmentation "
                        f"to increase effective training set size."
                    )
                elif dataset.row_count < 500 and model_type == ModelType.CNN:
                    result['recommendations'].append(
                        f"For {dataset.row_count} images, consider using data augmentation to increase "
                        f"effective training set size and prevent overfitting."
                    )

                # GPU recommendation for image models
                if dataset.row_count > 500:
                    result['recommendations'].append(
                        f"Enable GPU acceleration (CUDA) for significantly faster training on {dataset.row_count} images."
                    )

        elif is_tabular_model:
            # Tabular model specific recommendations
            if model_config['complexity'] in ['high', 'very_high'] and dataset.row_count and dataset.row_count < 1000:
                result['recommendations'].append(
                    f"{model_type} is a complex model. With only {dataset.row_count} samples, "
                    f"simpler models (Decision Tree, Logistic Regression) might generalize better."
                )

            if model_config['training_speed'] == 'slow' and dataset.row_count and dataset.row_count > 10000:
                # For image models, suggest GPU acceleration instead of different algorithms
                if model_type.lower() in ['cnn', 'convolutional']:
                    result['recommendations'].append(
                        f"{model_type} may take a long time to train on {dataset.row_count} samples. "
                        f"Consider enabling GPU acceleration for much faster training."
                    )
                else:
                    result['recommendations'].append(
                        f"{model_type} may take a long time to train on {dataset.row_count} samples. "
                        f"Consider using faster alternatives like LightGBM or XGBoost."
                    )

            # Neural network specific recommendations for tabular data
            if model_type == ModelType.NEURAL_NETWORK:
                if dataset.row_count and dataset.row_count < 1000:
                    result['recommendations'].append(
                        f"Neural networks typically need more data to generalize well. "
                        f"With {dataset.row_count} samples, ensemble methods (Random Forest, XGBoost) "
                        f"often perform better."
                    )
                elif dataset.row_count and dataset.row_count > 10000:
                    result['recommendations'].append(
                        f"Good dataset size for neural networks. Consider enabling GPU acceleration "
                        f"for faster training."
                    )

        return result

    @classmethod
    def get_recommended_models(cls, dataset, top_n: int = 5) -> List[Tuple[str, int, Dict]]:
        """
        Get recommended models for a dataset, ranked by compatibility score

        Args:
            dataset: Dataset instance
            top_n: Number of top recommendations to return

        Returns:
            List of tuples (model_type, score, validation_result)
        """
        recommendations = []

        for model_type in cls.COMPATIBILITY_MATRIX.keys():
            validation = cls.validate_compatibility(model_type, dataset)
            if validation['is_compatible'] or validation['score'] > 50:
                recommendations.append((
                    model_type,
                    validation['score'],
                    validation
                ))

        # Sort by score (descending) and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]

    @classmethod
    def get_model_info(cls, model_type: str) -> Optional[Dict]:
        """Get detailed information about a model type"""
        return cls.COMPATIBILITY_MATRIX.get(model_type)

    @classmethod
    def get_compatible_models_for_dataset_type(cls, dataset_type: str) -> List[str]:
        """Get all models compatible with a specific dataset type"""
        compatible = []
        for model_type, config in cls.COMPATIBILITY_MATRIX.items():
            if dataset_type in config['compatible_datasets']:
                compatible.append(model_type)
        return compatible
