"""
Advanced Training Logger
Provides detailed, real-time logging for ML model training with progress tracking,
metrics monitoring, and user-friendly status updates.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from django.core.cache import cache
from app.utils.websocket_logger import WebSocketLogger


class TrainingLogger:
    """
    Advanced logger for ML model training with detailed progress tracking.
    Logs are stored in the model's training_log field and cached for real-time access.
    """

    def __init__(self, model_obj, logger_name: str = __name__):
        """
        Initialize the training logger.

        Args:
            model_obj: The MLModel instance being trained
            logger_name: Name for the Python logger
        """
        self.model = model_obj
        self.logger = logging.getLogger(logger_name)
        self.logs = []
        self.start_time = None
        self.phase_start_time = None
        self.cache_key = f"training_logs_{model_obj.id}"

        # Initialize WebSocket logger for real-time streaming
        self.ws_logger = WebSocketLogger(model_obj.id)

        # Initialize metrics tracking
        self.metrics = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'best_accuracy': 0.0,
            'best_epoch': 0
        }

    def _friendly_model_name(self, model_type: str) -> str:
        """Convert technical model type to user-friendly name."""
        friendly_names = {
            'decision_tree': 'Decision Tree',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression',
            'svm': 'Support Vector Machine',
            'knn': 'K-Nearest Neighbors',
            'naive_bayes': 'Naive Bayes',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'adaboost': 'AdaBoost',
            'neural_network': 'Neural Network',
            'cnn': 'Convolutional Neural Network (Image Recognition)',
        }
        return friendly_names.get(model_type.lower(), model_type.title())

    def _friendly_column_name(self, column_name: str) -> str:
        """Convert column name to user-friendly format."""
        # Replace underscores with spaces and capitalize
        return column_name.replace('_', ' ').title()

    def _format_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _add_log(self, message: str, level: str = "INFO", data: Optional[Dict] = None):
        """
        Add a log entry.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, SUCCESS, DEBUG)
            data: Additional structured data
        """
        log_entry = {
            'timestamp': self._format_timestamp(),
            'level': level,
            'message': message,
            'data': data or {}
        }

        self.logs.append(log_entry)

        # Log to Python logger
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[Model {self.model.id}] {message}")

        # Update cache for real-time access
        cache.set(self.cache_key, self.logs, timeout=3600)  # 1 hour

        # Send via WebSocket for real-time streaming
        self.ws_logger.log(level, message, data)

        # Update model's training_log field
        self._update_model_log()

    def _update_model_log(self):
        """Update the model's training_log field with formatted logs."""
        formatted_log = self._format_logs_for_display()
        self.model.training_log = formatted_log
        self.model.save(update_fields=['training_log'])

    def _format_logs_for_display(self) -> str:
        """Format logs as a readable text string."""
        lines = []
        for entry in self.logs[-100:]:  # Keep last 100 entries
            timestamp = entry['timestamp']
            level = entry['level']
            message = entry['message']

            # Add emoji for visual clarity
            emoji = {
                'INFO': 'ℹ️',
                'SUCCESS': '✅',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'DEBUG': '🔍',
                'PROGRESS': '⏳'
            }.get(level, '📝')

            lines.append(f"{timestamp} {emoji} [{level}] {message}")

            # Add structured data if present
            if entry['data']:
                for key, value in entry['data'].items():
                    lines.append(f"    {key}: {value}")

        return '\n'.join(lines)

    def start_training(self, config: Dict[str, Any]):
        """
        Log the start of training.

        Args:
            config: Training configuration dictionary
        """
        self.start_time = time.time()
        self._add_log(f"🚀 Training Started for {self.model.name}", "INFO")
        self._add_log(f"Using {self._friendly_model_name(self.model.model_type)} algorithm", "INFO")
        self._add_log(f"Learning from dataset: {self.model.dataset.name}", "INFO")
        if self.model.target_column:
            self._add_log(f"Predicting: {self._friendly_column_name(self.model.target_column)}", "INFO")

    def log_data_loading(self, data_info: Dict[str, Any]):
        """
        Log data loading information.

        Args:
            data_info: Dictionary with data statistics
        """
        self.phase_start_time = time.time()
        self._add_log("\n📊 Preparing Your Data...", "INFO")

        for key, value in data_info.items():
            friendly_key = self._friendly_data_key(key)
            self._add_log(f"  • {friendly_key}: {value}", "INFO")

    def _friendly_data_key(self, key: str) -> str:
        """Convert data info keys to user-friendly format."""
        key_map = {
            'total_rows': 'Total examples',
            'total_samples': 'Total examples',
            'total_features': 'Number of features',
            'feature_count': 'Number of features',
            'missing_values': 'Missing data points',
            'dataset_size': 'Dataset size',
            'num_classes': 'Number of categories',
            'class_distribution': 'Category distribution',
        }
        return key_map.get(key, self._friendly_column_name(key))

    def log_data_split(self, train_size: int, val_size: int, test_size: Optional[int] = None):
        """Log data split information."""
        total = train_size + val_size + (test_size or 0)
        train_pct = (train_size / total) * 100
        val_pct = (val_size / total) * 100

        self._add_log("\n✂️ Dividing Data for Training:", "INFO")
        self._add_log(f"  • Learning from {train_size} examples ({train_pct:.1f}%)", "INFO")
        self._add_log(f"  • Testing accuracy on {val_size} examples ({val_pct:.1f}%)", "INFO")

        if test_size:
            test_pct = (test_size / total) * 100
            self._add_log(f"  • Final testing on {test_size} examples ({test_pct:.1f}%)", "INFO")

    def log_preprocessing(self, steps: List[str]):
        """
        Log preprocessing steps.

        Args:
            steps: List of preprocessing step descriptions
        """
        self._add_log("\n🔧 Preparing Data for Learning:", "INFO")
        for i, step in enumerate(steps, 1):
            friendly_step = self._friendly_preprocessing_step(step)
            self._add_log(f"  {i}. {friendly_step}", "INFO")

    def _friendly_preprocessing_step(self, step: str) -> str:
        """Convert preprocessing step to user-friendly format."""
        replacements = {
            'StandardScaler': 'Normalizing values to same scale',
            'MinMaxScaler': 'Scaling values to 0-1 range',
            'LabelEncoder': 'Converting categories to numbers',
            'OneHotEncoder': 'Converting categories to binary format',
            'Imputer': 'Filling in missing values',
            'PCA': 'Reducing number of features',
            'train_test_split': 'Splitting data for training and testing',
        }
        for tech, friendly in replacements.items():
            if tech in step:
                return friendly
        return step

    def log_model_architecture(self, architecture: Dict[str, Any]):
        """
        Log model architecture details.

        Args:
            architecture: Dictionary describing model architecture
        """
        self._add_log("\n🏗️ Building the Model:", "INFO")

        for key, value in architecture.items():
            friendly_key = self._friendly_arch_key(key)
            if isinstance(value, (list, dict)):
                self._add_log(f"  • {friendly_key}:", "INFO")
                if isinstance(value, list):
                    for item in value:
                        self._add_log(f"      - {item}", "INFO")
                else:
                    for k, v in value.items():
                        self._add_log(f"      - {k}: {v}", "INFO")
            else:
                self._add_log(f"  • {friendly_key}: {value}", "INFO")

    def _friendly_arch_key(self, key: str) -> str:
        """Convert architecture keys to user-friendly format."""
        key_map = {
            'algorithm': 'Algorithm',
            'n_estimators': 'Number of trees',
            'max_depth': 'Maximum depth',
            'learning_rate': 'Learning rate',
            'hidden_layers': 'Network layers',
            'activation': 'Activation function',
            'optimizer': 'Optimization method',
            'loss': 'Loss function',
            'batch_size': 'Batch size',
            'epochs': 'Training cycles',
            'dropout': 'Dropout rate',
            'kernel_size': 'Kernel size',
            'filters': 'Number of filters',
            'pooling': 'Pooling type',
        }
        return key_map.get(key, self._friendly_column_name(key))

    def log_training_start(self, total_epochs: int):
        """Log the start of the training phase."""
        self._add_log(f"\n🎯 Starting to Learn ({total_epochs} training rounds)", "INFO")
        self.phase_start_time = time.time()

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self._add_log(f"\n📈 Training Round {epoch} of {total_epochs}", "PROGRESS")

    def log_batch_progress(self, batch: int, total_batches: int, loss: float, metrics: Optional[Dict] = None):
        """
        Log progress within an epoch (for mini-batch training).

        Args:
            batch: Current batch number
            total_batches: Total number of batches
            loss: Current batch loss
            metrics: Additional metrics dictionary
        """
        progress_pct = (batch / total_batches) * 100
        progress_bar = self._create_progress_bar(batch, total_batches, width=30)

        msg = f"  Batch {batch}/{total_batches} {progress_bar} {progress_pct:.1f}% | Loss: {loss:.4f}"

        if metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            msg += f" | {metric_str}"

        # Only log every 10% progress or last batch to avoid spam
        if batch % max(1, total_batches // 10) == 0 or batch == total_batches:
            self._add_log(msg, "DEBUG")

    def log_epoch_end(self, epoch: int, total_epochs: int, train_loss: float,
                     train_acc: Optional[float] = None, val_loss: Optional[float] = None,
                     val_acc: Optional[float] = None, lr: Optional[float] = None,
                     epoch_time: Optional[float] = None):
        """
        Log the end of an epoch with comprehensive metrics.

        Args:
            epoch: Current epoch number
            total_epochs: Total epochs
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
            epoch_time: Time taken for this epoch
        """
        # Track metrics
        self.metrics['epochs'].append(epoch)
        self.metrics['losses'].append(train_loss)
        if train_acc is not None:
            self.metrics['accuracies'].append(train_acc)
        if val_loss is not None:
            self.metrics['val_losses'].append(val_loss)
        if val_acc is not None:
            self.metrics['val_accuracies'].append(val_acc)
        if lr is not None:
            self.metrics['learning_rates'].append(lr)

        # Check for best accuracy
        if val_acc and val_acc > self.metrics['best_accuracy']:
            self.metrics['best_accuracy'] = val_acc
            self.metrics['best_epoch'] = epoch
            is_best = True
        else:
            is_best = False

        # Format log message
        msg = f"\n  ✓ Round {epoch} of {total_epochs} Complete:"
        self._add_log(msg, "SUCCESS" if is_best else "INFO")

        data = {}
        self._add_log(f"      Error rate: {train_loss:.4f}", "INFO")
        data['train_loss'] = f"{train_loss:.4f}"

        if train_acc is not None:
            self._add_log(f"      Training accuracy: {train_acc*100:.1f}%", "INFO")
            data['train_accuracy'] = f"{train_acc*100:.2f}%"

        if val_loss is not None:
            self._add_log(f"      Validation error: {val_loss:.4f}", "INFO")
            data['val_loss'] = f"{val_loss:.4f}"

        if val_acc is not None:
            self._add_log(f"      Validation accuracy: {val_acc*100:.1f}%", "INFO")
            data['val_accuracy'] = f"{val_acc*100:.2f}%"

        if lr is not None:
            self._add_log(f"      Learning speed: {lr:.6f}", "INFO")
            data['learning_rate'] = f"{lr:.6f}"

        if epoch_time:
            self._add_log(f"      Time taken: {self._format_duration(epoch_time)}", "INFO")
            data['epoch_time'] = self._format_duration(epoch_time)

        if is_best:
            self._add_log("      🌟 Best result so far!", "SUCCESS")

        # Estimate remaining time
        if epoch < total_epochs and epoch_time:
            remaining_epochs = total_epochs - epoch
            estimated_time = remaining_epochs * epoch_time
            self._add_log(f"      ⏱️ Estimated time left: {self._format_duration(estimated_time)}", "INFO")

    def log_validation_start(self):
        """Log start of validation phase."""
        self._add_log("\n🔍 Checking Model Performance...", "PROGRESS")

    def log_evaluation_results(self, metrics: Dict[str, Any]):
        """
        Log final evaluation results.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        self._add_log("\n📊 Final Performance Results", "SUCCESS")

        for key, value in metrics.items():
            friendly_key = self._friendly_metric_key(key)
            if isinstance(value, float):
                if 'accuracy' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                    self._add_log(f"  • {friendly_key}: {value*100:.1f}%", "INFO")
                else:
                    self._add_log(f"  • {friendly_key}: {value:.4f}", "INFO")
            else:
                self._add_log(f"  • {friendly_key}: {value}", "INFO")

    def _friendly_metric_key(self, key: str) -> str:
        """Convert metric keys to user-friendly format."""
        key_map = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'train_loss': 'Training error',
            'val_loss': 'Validation error',
            'test_loss': 'Test error',
            'train_accuracy': 'Training accuracy',
            'val_accuracy': 'Validation accuracy',
            'test_accuracy': 'Test accuracy',
        }
        return key_map.get(key, self._friendly_column_name(key))

    def log_model_save(self, model_path: str, model_size: Optional[int] = None):
        """
        Log model saving.

        Args:
            model_path: Path where model is saved
            model_size: Size of saved model in bytes
        """
        self._add_log("\n💾 Saving Your Model...", "INFO")
        self._add_log(f"  • Saved to: {model_path}", "INFO")

        if model_size:
            size_mb = model_size / (1024 * 1024)
            self._add_log(f"  • Model size: {size_mb:.2f} MB", "INFO")

    def log_training_complete(self, final_accuracy: Optional[float] = None):
        """
        Log training completion.

        Args:
            final_accuracy: Final accuracy achieved
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        self._add_log("\n✅ Training Complete!", "SUCCESS")

        if final_accuracy is not None:
            self._add_log(f"🎯 Final Accuracy: {final_accuracy*100:.1f}%", "SUCCESS")

        if self.metrics['best_accuracy'] > 0:
            self._add_log(f"🌟 Best Result: {self.metrics['best_accuracy']*100:.1f}% (achieved in round {self.metrics['best_epoch']})", "SUCCESS")

        self._add_log(f"⏱️ Total Time: {self._format_duration(elapsed_time)}", "INFO")

        # Training summary
        if self.metrics['epochs']:
            self._add_log("\n📈 Summary:", "INFO")
            self._add_log(f"  • Training rounds completed: {len(self.metrics['epochs'])}", "INFO")
            self._add_log(f"  • Final error rate: {self.metrics['losses'][-1]:.4f}", "INFO")

            if self.metrics['val_accuracies']:
                avg_val_acc = sum(self.metrics['val_accuracies']) / len(self.metrics['val_accuracies'])
                self._add_log(f"  • Average accuracy: {avg_val_acc*100:.1f}%", "INFO")

        # Send completion notification via WebSocket
        self.ws_logger.complete(
            status='complete',
            accuracy=final_accuracy,
            message='Training completed successfully'
        )

    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error during training.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        error_message = f"ERROR {context}: {str(error)}"
        self._add_log(f"\n❌ {error_message}", "ERROR", {
            'error_type': type(error).__name__,
            'error_message': str(error)
        })

        # Send error notification via WebSocket
        self.ws_logger.send_error(error_message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self._add_log(f"⚠️ {message}", "WARNING")

    def log_info(self, message: str):
        """Log an info message."""
        self._add_log(message, "INFO")

    def _create_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """
        Create a text-based progress bar.

        Args:
            current: Current progress value
            total: Total value
            width: Width of the progress bar in characters

        Returns:
            Formatted progress bar string
        """
        filled = int(width * current / total)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def get_logs_json(self) -> str:
        """Get logs as JSON string."""
        return json.dumps({
            'logs': self.logs,
            'metrics': self.metrics,
            'model_id': str(self.model.id),
            'model_name': self.model.name
        }, indent=2)

    def get_cached_logs(self) -> List[Dict]:
        """Get logs from cache."""
        return cache.get(self.cache_key, [])

    def clear_cache(self):
        """Clear cached logs."""
        cache.delete(self.cache_key)
