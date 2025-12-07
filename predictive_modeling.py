"""
Predictive Modeling Module
Builds ML models from cleaned data for predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from enum import Enum


class ProblemType(Enum):
    """Types of ML problems"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"


class ModelType(Enum):
    """Available model types"""
    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    
    # Regression
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    
    # Clustering
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


@dataclass
class ModelConfig:
    """Configuration for ML model"""
    problem_type: ProblemType
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    random_state: int = 42
    hyperparameters: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'problem_type': self.problem_type.value,
            'model_type': self.model_type.value,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'hyperparameters': self.hyperparameters or {}
        }


@dataclass
class ModelResults:
    """Results from model training/evaluation"""
    model_type: str
    problem_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    predictions_sample: Optional[List[Any]]
    training_time: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


class DataPreprocessor:
    """Prepares cleaned data for ML"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame, target_column: str,
                        feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for ML
        Returns: X, y, feature_names
        """
        df = df.copy()
        
        # Validate target column exists
        if target_column not in df.columns:
            # Try to find similar column (case-insensitive, strip spaces)
            target_column_clean = target_column.strip().lower()
            for col in df.columns:
                if col.strip().lower() == target_column_clean:
                    target_column = col
                    break
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
        
        # If no features specified, use all columns except target
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            # Validate feature columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Simple label encoding for now
            unique_vals = X[col].unique()
            encoding = {val: idx for idx, val in enumerate(unique_vals)}
            self.encoders[col] = encoding
            X[col] = X[col].map(encoding)
        
        # Handle missing values in features
        X = X.fillna(X.mean(numeric_only=True))
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            unique_vals = y.unique()
            encoding = {val: idx for idx, val in enumerate(unique_vals)}
            self.encoders[target_column] = encoding
            y = y.map(encoding)
        
        self.feature_names = list(X.columns)
        
        return X.values, y.values, self.feature_names
    
    def inverse_transform_target(self, y_encoded: np.ndarray, target_column: str) -> np.ndarray:
        """Convert encoded target back to original values"""
        if target_column in self.encoders:
            inverse_encoding = {v: k for k, v in self.encoders[target_column].items()}
            return np.array([inverse_encoding.get(val, val) for val in y_encoded])
        return y_encoded


class ModelBuilder:
    """Builds and trains ML models"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.config = None
        self.results = None
    
    def auto_detect_problem_type(self, df: pd.DataFrame, target_column: str) -> ProblemType:
        """Automatically detect the type of ML problem"""
        # Handle column names with spaces or special characters
        if target_column not in df.columns:
            # Try to find similar column (case-insensitive, strip spaces)
            target_column_clean = target_column.strip().lower()
            for col in df.columns:
                if col.strip().lower() == target_column_clean:
                    target_column = col
                    break
            
            # If still not found, raise clear error
            if target_column not in df.columns:
                raise ValueError(f"Column '{target_column}' not found. Available columns: {list(df.columns)}")
        
        target = df[target_column]
        
        # Check number of unique values
        n_unique = target.nunique()
        n_samples = len(target)
        
        # If target is numeric and has many unique values, likely regression
        if pd.api.types.is_numeric_dtype(target):
            if n_unique > 20 or n_unique / n_samples > 0.05:
                return ProblemType.REGRESSION
            else:
                return ProblemType.CLASSIFICATION
        else:
            # Categorical target
            return ProblemType.CLASSIFICATION
    
    def suggest_model(self, problem_type: ProblemType, n_samples: int, n_features: int) -> ModelType:
        """Suggest best model based on problem and data characteristics"""
        if problem_type == ProblemType.CLASSIFICATION:
            if n_samples < 1000:
                return ModelType.LOGISTIC_REGRESSION
            elif n_samples < 10000:
                return ModelType.DECISION_TREE_CLASSIFIER
            else:
                return ModelType.RANDOM_FOREST_CLASSIFIER
        
        elif problem_type == ProblemType.REGRESSION:
            if n_samples < 1000:
                return ModelType.LINEAR_REGRESSION
            elif n_samples < 10000:
                return ModelType.DECISION_TREE_REGRESSOR
            else:
                return ModelType.RANDOM_FOREST_REGRESSOR
        
        elif problem_type == ProblemType.CLUSTERING:
            return ModelType.KMEANS
        
        return ModelType.RANDOM_FOREST_CLASSIFIER
    
    def build_model(self, model_type: ModelType, hyperparameters: Dict = None):
        """Build model instance (placeholder - would use sklearn in production)"""
        # Note: In a real implementation, you would import sklearn here
        # For now, we'll create a mock model structure
        
        self.model = {
            'type': model_type.value,
            'hyperparameters': hyperparameters or {},
            'fitted': False
        }
        
        return self.model
    
    def train(self, df: pd.DataFrame, config: ModelConfig) -> ModelResults:
        """Train the model"""
        import time
        start_time = time.time()
        
        # Prepare data
        X, y, feature_names = self.preprocessor.prepare_features(
            df, config.target_column, config.feature_columns
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Build and train model
        model = self._create_sklearn_model(config.model_type, config.hyperparameters)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, config.problem_type)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(feature_names, np.abs(model.coef_).flatten()))
        
        # Get sample predictions
        predictions_sample = self._get_prediction_samples(y_test, y_pred, config.target_column)
        
        training_time = time.time() - start_time
        
        # Store model and config
        self.model = model
        self.config = config
        
        # Create results
        self.results = ModelResults(
            model_type=config.model_type.value,
            problem_type=config.problem_type.value,
            metrics=metrics,
            feature_importance=feature_importance,
            predictions_sample=predictions_sample,
            training_time=training_time,
            timestamp=datetime.now().isoformat()
        )
        
        return self.results
    
    def _create_sklearn_model(self, model_type: ModelType, hyperparameters: Dict = None):
        """Create sklearn model instance"""
        params = hyperparameters or {}
        
        if model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        
        elif model_type == ModelType.DECISION_TREE_CLASSIFIER:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**params)
        
        elif model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        
        elif model_type == ModelType.GRADIENT_BOOSTING_CLASSIFIER:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params)
        
        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        
        elif model_type == ModelType.RIDGE_REGRESSION:
            from sklearn.linear_model import Ridge
            return Ridge(**params)
        
        elif model_type == ModelType.DECISION_TREE_REGRESSOR:
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(**params)
        
        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        
        elif model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
        
        elif model_type == ModelType.KMEANS:
            from sklearn.cluster import KMeans
            return KMeans(**params)
        
        elif model_type == ModelType.DBSCAN:
            from sklearn.cluster import DBSCAN
            return DBSCAN(**params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: ProblemType) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""
        metrics = {}
        
        if problem_type == ProblemType.CLASSIFICATION:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            try:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            except Exception as e:
                metrics['error'] = str(e)
        
        elif problem_type == ProblemType.REGRESSION:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def _get_prediction_samples(self, y_true, y_pred, target_column: str, n_samples: int = 10):
        """Get sample predictions for display"""
        # Inverse transform if needed
        y_true_decoded = self.preprocessor.inverse_transform_target(y_true, target_column)
        y_pred_decoded = self.preprocessor.inverse_transform_target(y_pred, target_column)
        
        samples = []
        for i in range(min(n_samples, len(y_true))):
            samples.append({
                'actual': str(y_true_decoded[i]),
                'predicted': str(y_pred_decoded[i]),
                'correct': y_true_decoded[i] == y_pred_decoded[i]
            })
        
        return samples
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X, _, _ = self.preprocessor.prepare_features(
            new_data, 
            self.config.target_column,
            self.config.feature_columns
        )
        
        # Predict
        predictions = self.model.predict(X)
        
        # Inverse transform if needed
        predictions = self.preprocessor.inverse_transform_target(
            predictions, 
            self.config.target_column
        )
        
        return predictions
    
    def export_model(self) -> bytes:
        """Export trained model"""
        import pickle
        return pickle.dumps({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'results': self.results
        })
    
    def load_model(self, model_bytes: bytes):
        """Load trained model"""
        import pickle
        data = pickle.loads(model_bytes)
        self.model = data['model']
        self.preprocessor = data['preprocessor']
        self.config = data['config']
        self.results = data['results']


class PredictionPipeline:
    """Orchestrates the prediction workflow"""
    
    def __init__(self, cleaned_data: pd.DataFrame):
        self.data = cleaned_data
        self.model_builder = ModelBuilder()
        self.trained_models = {}
    
    def analyze_prediction_readiness(self) -> Dict[str, Any]:
        """Check if data is ready for ML"""
        analysis = {
            'ready': True,
            'issues': [],
            'recommendations': [],
            'potential_targets': [],
            'feature_count': 0
        }
        
        # Check data size
        if len(self.data) < 30:
            analysis['ready'] = False
            analysis['issues'].append("Insufficient data (need at least 30 rows)")
        elif len(self.data) < 100:
            analysis['recommendations'].append("More data would improve model accuracy (current: {})".format(len(self.data)))
        
        # Check for numeric columns (potential targets)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis['feature_count'] = len(self.data.columns)
        analysis['numeric_columns'] = numeric_cols
        analysis['categorical_columns'] = categorical_cols
        
        # Suggest potential target columns
        for col in self.data.columns:
            n_unique = self.data[col].nunique()
            if 2 <= n_unique <= 20:  # Good for classification
                analysis['potential_targets'].append({
                    'column': col,
                    'type': 'classification',
                    'n_classes': n_unique
                })
            elif pd.api.types.is_numeric_dtype(self.data[col]) and n_unique > 20:
                analysis['potential_targets'].append({
                    'column': col,
                    'type': 'regression',
                    'range': f"{self.data[col].min():.2f} - {self.data[col].max():.2f}"
                })
        
        # Check for missing values
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).max()
        if missing_pct > 0:
            analysis['recommendations'].append(f"Some columns still have missing values ({missing_pct:.1f}% max)")
        
        return analysis
    
    def train_model(self, target_column: str, feature_columns: List[str] = None,
                   model_type: ModelType = None, hyperparameters: Dict = None) -> ModelResults:
        """Train a prediction model"""
        
        # Auto-detect problem type
        problem_type = self.model_builder.auto_detect_problem_type(self.data, target_column)
        
        # Auto-select model if not specified
        if model_type is None:
            model_type = self.model_builder.suggest_model(
                problem_type, 
                len(self.data), 
                len(feature_columns) if feature_columns else len(self.data.columns) - 1
            )
        
        # Create config
        config = ModelConfig(
            problem_type=problem_type,
            model_type=model_type,
            target_column=target_column,
            feature_columns=feature_columns or [col for col in self.data.columns if col != target_column],
            hyperparameters=hyperparameters
        )
        
        # Train model
        results = self.model_builder.train(self.data, config)
        
        # Store trained model
        model_key = f"{target_column}_{model_type.value}"
        self.trained_models[model_key] = {
            'model_builder': self.model_builder,
            'config': config,
            'results': results
        }
        
        return results
    
    def get_model_recommendations(self, target_column: str) -> List[Dict[str, Any]]:
        """Get recommended models for target"""
        problem_type = self.model_builder.auto_detect_problem_type(self.data, target_column)
        
        recommendations = []
        
        if problem_type == ProblemType.CLASSIFICATION:
            recommendations = [
                {
                    'model': ModelType.LOGISTIC_REGRESSION,
                    'pros': ['Fast', 'Interpretable', 'Works well for binary classification'],
                    'cons': ['Assumes linear relationships'],
                    'best_for': 'Small to medium datasets, binary classification'
                },
                {
                    'model': ModelType.RANDOM_FOREST_CLASSIFIER,
                    'pros': ['Handles non-linear relationships', 'Feature importance', 'Robust'],
                    'cons': ['Slower', 'Less interpretable'],
                    'best_for': 'Medium to large datasets, complex patterns'
                },
                {
                    'model': ModelType.GRADIENT_BOOSTING_CLASSIFIER,
                    'pros': ['High accuracy', 'Handles complex patterns'],
                    'cons': ['Slow to train', 'Requires tuning'],
                    'best_for': 'When maximum accuracy is needed'
                }
            ]
        
        elif problem_type == ProblemType.REGRESSION:
            recommendations = [
                {
                    'model': ModelType.LINEAR_REGRESSION,
                    'pros': ['Simple', 'Fast', 'Interpretable'],
                    'cons': ['Assumes linear relationships'],
                    'best_for': 'Linear relationships, baseline model'
                },
                {
                    'model': ModelType.RANDOM_FOREST_REGRESSOR,
                    'pros': ['Handles non-linearity', 'Feature importance', 'Robust'],
                    'cons': ['Slower', 'Can overfit'],
                    'best_for': 'Complex patterns, medium to large data'
                },
                {
                    'model': ModelType.GRADIENT_BOOSTING_REGRESSOR,
                    'pros': ['High accuracy', 'Excellent performance'],
                    'cons': ['Slow', 'Requires tuning'],
                    'best_for': 'Maximum accuracy needed'
                }
            ]
        
        return recommendations


class ModelComparison:
    """Compare multiple models"""
    
    @staticmethod
    def compare_models(results_list: List[ModelResults]) -> pd.DataFrame:
        """Compare multiple model results"""
        comparison_data = []
        
        for result in results_list:
            row = {
                'Model': result.model_type,
                'Training Time (s)': result.training_time
            }
            row.update(result.metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def recommend_best_model(results_list: List[ModelResults], 
                            metric: str = 'accuracy') -> ModelResults:
        """Recommend best model based on metric"""
        if not results_list:
            return None
        
        # Find model with best metric
        best_model = max(results_list, key=lambda x: x.metrics.get(metric, 0))
        return best_model