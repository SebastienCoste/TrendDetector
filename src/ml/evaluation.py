import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Configurable evaluation metrics for both classification and regression models"""
    
    def __init__(self, model_type: str, metrics_config: Dict[str, Any]):
        self.model_type = model_type.lower()
        self.metrics_config = metrics_config
        
        if self.model_type == "classification":
            self.available_metrics = metrics_config.get('classification_metrics', ['accuracy', 'precision', 'recall', 'f1'])
            self.primary_metric = metrics_config.get('primary_classification_metric', 'accuracy')
        elif self.model_type == "regression":
            self.available_metrics = metrics_config.get('regression_metrics', ['mae', 'rmse', 'r2'])
            self.primary_metric = metrics_config.get('primary_regression_metric', 'mae')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        logger.info(f"ModelEvaluator initialized for {self.model_type} with metrics: {self.available_metrics}")

    def evaluate_classification(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Evaluate classification model performance"""
        results = {}
        
        if 'accuracy' in self.available_metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.available_metrics:
            results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            
        if 'recall' in self.available_metrics:
            results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
        if 'f1' in self.available_metrics:
            results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
        return results

    def evaluate_regression(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Evaluate regression model performance"""
        results = {}
        
        if 'mae' in self.available_metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
            
        if 'rmse' in self.available_metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
        if 'r2' in self.available_metrics:
            results['r2'] = r2_score(y_true, y_pred)
            
        return results

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
        """Unified evaluation interface"""
        if self.model_type == "classification":
            return self.evaluate_classification(y_true, y_pred)
        else:  # regression
            return self.evaluate_regression(y_true, y_pred)

    def get_primary_metric_value(self, evaluation_results: Dict[str, float]) -> float:
        """Get the primary metric value from evaluation results"""
        if self.primary_metric in evaluation_results:
            return evaluation_results[self.primary_metric]
        else:
            logger.warning(f"Primary metric '{self.primary_metric}' not found in results")
            return 0.0

    def format_results(self, evaluation_results: Dict[str, float]) -> str:
        """Format evaluation results for display"""
        if not evaluation_results:
            return "No evaluation results available"
            
        formatted_lines = []
        for metric, value in evaluation_results.items():
            marker = " *" if metric == self.primary_metric else ""
            formatted_lines.append(f"{metric.upper()}: {value:.4f}{marker}")
            
        return "\n".join(formatted_lines)

    def is_better_score(self, new_score: float, old_score: Optional[float]) -> bool:
        """Determine if new score is better than old score based on metric type"""
        if old_score is None:
            return True
            
        # For classification metrics (accuracy, precision, recall, f1), higher is better
        # For MAE and RMSE, lower is better
        # For R2, higher is better
        if self.primary_metric in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
            return new_score > old_score
        elif self.primary_metric in ['mae', 'rmse']:
            return new_score < old_score
        else:
            # Default to higher is better
            return new_score > old_score