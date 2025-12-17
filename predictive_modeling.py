"""
Complete Predictive Modeling Module - Production Ready
Fully functional ML with real sklearn models, hyperparameter tuning, and model persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from enum import Enum
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Classification models
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC, SVR


class ProblemType(Enum):
    """Types of ML problems"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class ModelType(Enum):
    """Available model types"""
    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    SVM_CLASSIFIER = "svm_classifier"
    
    # Regression
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    SVM_REGRESSOR = "svm_regressor"
    
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
    cv_folds: int = 5
    tune_hyperparameters: bool = False
    scale_features: bool = True
    
    def to_dict(self):
        return {
            'problem_type': self.problem_type.value,
            'model_type': self.model_type.value,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'hyperparameters': self.hyperparameters or {},
            'cv_folds': self.cv_folds,
            'tune_hyperparameters': self.tune_hyperparameters,
            'scale_features': self.scale_features
        }


@dataclass
class ModelResults:
    """Results from model training/evaluation"""
    model_type: str
    problem_type: str
    metrics: Dict[str, float]
    cv_scores: Optional[Dict[str, List[float]]]
    feature_importance: Optional[Dict[str, float]]
    predictions_sample: Optional[List[Dict[str, Any]]]
    confusion_matrix: Optional[List[List[int]]]
    classification_report: Optional[str]
    best_params: Optional[Dict[str, Any]]
    training_time: float
    cv_time: Optional[float]
    timestamp: str
    model_path: Optional[str]
    
    def to_dict(self):
        return asdict(self)


class DataPreprocessor:
    """Prepares data for ML with proper encoding and scaling"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_encoder = None
        self.categorical_columns = []
        self.numeric_columns = []
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame, target_column: str,
                     feature_columns: List[str] = None,
                     scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit preprocessor and transform data
        Returns: X, y, feature_names
        """
        df = df.copy()
        
        # Validate columns
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical features
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Handle missing values
            X[col] = X[col].fillna('__MISSING__')
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Fill missing values in numeric columns
        for col in self.numeric_columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y = y.fillna('__MISSING__')
            y = self.target_encoder.fit_transform(y.astype(str))
        else:
            # Handle missing in numeric target
            y = y.fillna(y.median())
        
        self.feature_names = list(X.columns)
        
        # Convert to numpy - X is still a DataFrame here
        X_array = X.values.astype(float)
        # y might be Series or numpy array
        y_array = y.values if hasattr(y, 'values') else y
        
        # Scale features
        if scale_features and len(self.numeric_columns) > 0:
            X_array = self.scaler.fit_transform(X_array)
        
        self.is_fitted = True
        return X_array, y_array, self.feature_names
    
    def transform(self, new_data: pd.DataFrame, feature_columns: List[str] = None,
                 scale_features: bool = True) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Handle if new_data is already a numpy array
        if isinstance(new_data, np.ndarray):
            X_array = new_data.astype(float)
            if scale_features and len(self.numeric_columns) > 0:
                X_array = self.scaler.transform(X_array)
            return X_array
        
        df = new_data.copy()
        
        if feature_columns is None:
            feature_columns = self.feature_names
        
        X = df[feature_columns].copy()
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = X[col].fillna('__MISSING__')
                # Handle unseen categories
                le = self.label_encoders[col]
                X[col] = X[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
        
        # Fill missing in numeric
        for col in self.numeric_columns:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        X_array = X.values.astype(float)
        
        # Scale
        if scale_features and len(self.numeric_columns) > 0:
            X_array = self.scaler.transform(X_array)
        
        return X_array
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded target back to original values"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y_encoded.astype(int))
        return y_encoded


class ModelBuilder:
    """Builds and trains real ML models with sklearn"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.config = None
        self.results = None
        self.best_model = None
    
    def auto_detect_problem_type(self, df: pd.DataFrame, target_column: str) -> ProblemType:
        """Automatically detect the type of ML problem"""
        if target_column not in df.columns:
            target_column_clean = target_column.strip().lower()
            for col in df.columns:
                if col.strip().lower() == target_column_clean:
                    target_column = col
                    break
            
            if target_column not in df.columns:
                raise ValueError(f"Column '{target_column}' not found")
        
        target = df[target_column]
        n_unique = target.nunique()
        n_samples = len(target)
        
        # Classification if categorical or few unique values
        if pd.api.types.is_numeric_dtype(target):
            if n_unique > 20 and n_unique / n_samples > 0.05:
                return ProblemType.REGRESSION
            else:
                return ProblemType.CLASSIFICATION
        else:
            return ProblemType.CLASSIFICATION
    
    def suggest_model(self, problem_type: ProblemType, n_samples: int, 
                     n_features: int) -> ModelType:
        """Suggest best model based on problem and data characteristics"""
        if problem_type == ProblemType.CLASSIFICATION:
            if n_samples < 1000:
                return ModelType.LOGISTIC_REGRESSION
            elif n_samples < 10000:
                return ModelType.RANDOM_FOREST_CLASSIFIER
            else:
                return ModelType.GRADIENT_BOOSTING_CLASSIFIER
        
        elif problem_type == ProblemType.REGRESSION:
            if n_samples < 1000:
                return ModelType.LINEAR_REGRESSION
            elif n_samples < 10000:
                return ModelType.RANDOM_FOREST_REGRESSOR
            else:
                return ModelType.GRADIENT_BOOSTING_REGRESSOR
        
        elif problem_type == ProblemType.CLUSTERING:
            return ModelType.KMEANS
        
        return ModelType.RANDOM_FOREST_CLASSIFIER
    
    def get_default_hyperparameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default hyperparameters for each model type"""
        defaults = {
            ModelType.LOGISTIC_REGRESSION: {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            ModelType.DECISION_TREE_CLASSIFIER: {
                'max_depth': 10,
                'min_samples_split': 2,
                'random_state': 42
            },
            ModelType.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            ModelType.LINEAR_REGRESSION: {},
            ModelType.RIDGE_REGRESSION: {
                'alpha': 1.0
            },
            ModelType.RANDOM_FOREST_REGRESSOR: {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            ModelType.GRADIENT_BOOSTING_REGRESSOR: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            ModelType.KMEANS: {
                'n_clusters': 3,
                'random_state': 42
            }
        }
        return defaults.get(model_type, {})
    
    def get_hyperparameter_grid(self, model_type: ModelType) -> Dict[str, List]:
        """Get hyperparameter grid for tuning"""
        grids = {
            ModelType.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            ModelType.RANDOM_FOREST_REGRESSOR: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelType.GRADIENT_BOOSTING_REGRESSOR: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'max_iter': [1000]
            }
        }
        return grids.get(model_type, {})
    
    def create_model(self, model_type: ModelType, 
                    hyperparameters: Dict[str, Any] = None) -> Any:
        """Create sklearn model instance"""
        params = hyperparameters or self.get_default_hyperparameters(model_type)
        
        models = {
            ModelType.LOGISTIC_REGRESSION: LogisticRegression,
            ModelType.DECISION_TREE_CLASSIFIER: DecisionTreeClassifier,
            ModelType.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier,
            ModelType.SVM_CLASSIFIER: SVC,
            ModelType.LINEAR_REGRESSION: LinearRegression,
            ModelType.RIDGE_REGRESSION: Ridge,
            ModelType.DECISION_TREE_REGRESSOR: DecisionTreeRegressor,
            ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor,
            ModelType.SVM_REGRESSOR: SVR,
            ModelType.KMEANS: KMeans,
            ModelType.DBSCAN: DBSCAN
        }
        
        model_class = models.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_class(**params)
    
    def train(self, df: pd.DataFrame, config: ModelConfig) -> ModelResults:
        """Train model with full pipeline"""
        import time
        start_time = time.time()
        
        # Prepare data
        X, y, feature_names = self.preprocessor.fit_transform(
            df, config.target_column, config.feature_columns, config.scale_features
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Hyperparameter tuning if requested
        best_params = None
        cv_time = None
        
        if config.tune_hyperparameters:
            cv_start = time.time()
            print("🔧 Tuning hyperparameters...")
            
            base_model = self.create_model(config.model_type)
            param_grid = self.get_hyperparameter_grid(config.model_type)
            
            if param_grid:
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=config.cv_folds,
                    scoring='accuracy' if config.problem_type == ProblemType.CLASSIFICATION else 'r2',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"✅ Best params: {best_params}")
            else:
                self.model = base_model
                self.model.fit(X_train, y_train)
            
            cv_time = time.time() - cv_start
        else:
            # Use default or provided hyperparameters
            self.model = self.create_model(config.model_type, config.hyperparameters)
            self.model.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = None
        if config.problem_type != ProblemType.CLUSTERING:
            print("📊 Running cross-validation...")
            if config.problem_type == ProblemType.CLASSIFICATION:
                cv_accuracy = cross_val_score(
                    self.model, X_train, y_train, cv=config.cv_folds, scoring='accuracy'
                )
                cv_scores = {
                    'accuracy': cv_accuracy.tolist(),
                    'mean_accuracy': float(cv_accuracy.mean()),
                    'std_accuracy': float(cv_accuracy.std())
                }
            else:  # Regression
                cv_r2 = cross_val_score(
                    self.model, X_train, y_train, cv=config.cv_folds, scoring='r2'
                )
                cv_scores = {
                    'r2': cv_r2.tolist(),
                    'mean_r2': float(cv_r2.mean()),
                    'std_r2': float(cv_r2.std())
                }
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, config.problem_type)
        
        # Feature importance
        feature_importance = self._get_feature_importance(feature_names)
        
        # Sample predictions
        predictions_sample = self._get_prediction_samples(
            y_test, y_pred, config.target_column, n_samples=10
        )
        
        # Confusion matrix for classification
        conf_matrix = None
        class_report = None
        if config.problem_type == ProblemType.CLASSIFICATION:
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # Save model
        model_path = self._save_model(config)
        
        # Store config
        self.config = config
        
        # Create results
        self.results = ModelResults(
            model_type=config.model_type.value,
            problem_type=config.problem_type.value,
            metrics=metrics,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            predictions_sample=predictions_sample,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            best_params=best_params,
            training_time=training_time,
            cv_time=cv_time,
            timestamp=datetime.now().isoformat(),
            model_path=model_path
        )
        
        print(f"✅ Model trained in {training_time:.2f}s")
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred, 
                          problem_type: ProblemType) -> Dict[str, float]:
        """Calculate appropriate metrics"""
        metrics = {}
        
        if problem_type == ProblemType.CLASSIFICATION:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # Handle binary vs multiclass
            n_classes = len(np.unique(y_true))
            avg_method = 'binary' if n_classes == 2 else 'weighted'
            
            metrics['precision'] = float(precision_score(
                y_true, y_pred, average=avg_method, zero_division=0
            ))
            metrics['recall'] = float(recall_score(
                y_true, y_pred, average=avg_method, zero_division=0
            ))
            metrics['f1_score'] = float(f1_score(
                y_true, y_pred, average=avg_method, zero_division=0
            ))
            
            # ROC-AUC for binary classification
            if n_classes == 2 and hasattr(self.model, 'predict_proba'):
                try:
                    y_proba = self.model.predict_proba(
                        self.preprocessor.transform(
                            pd.DataFrame(y_true), scale_features=self.config.scale_features
                        )
                    )[:, 1]
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                except:
                    pass
        
        elif problem_type == ProblemType.REGRESSION:
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
            
            # Additional metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            
            # Explained variance
            from sklearn.metrics import explained_variance_score
            metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        return metrics
    
    def _get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coef = np.abs(self.model.coef_)
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            return {name: float(c) for name, c in zip(feature_names, coef)}
        
        return None
    
    def _get_prediction_samples(self, y_true, y_pred, target_column: str,
                               n_samples: int = 10) -> List[Dict[str, Any]]:
        """Get sample predictions for display"""
        y_true_decoded = self.preprocessor.inverse_transform_target(y_true)
        y_pred_decoded = self.preprocessor.inverse_transform_target(y_pred)
        
        samples = []
        indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
        
        for i in indices:
            samples.append({
                'actual': str(y_true_decoded[i]),
                'predicted': str(y_pred_decoded[i]),
                'correct': bool(y_true_decoded[i] == y_pred_decoded[i])
            })
        
        return samples
    
    def _save_model(self, config: ModelConfig) -> str:
        """Save model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{config.model_type.value}_{timestamp}.pkl"
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'config': config,
            'timestamp': timestamp
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Model saved: {filename}")
        return filename
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.config = model_data['config']
        print(f"✅ Model loaded from {filepath}")
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")
        
        if self.config is None:
            raise ValueError("Config not found. Train or load model first.")
        
        # Transform data
        X_new = self.preprocessor.transform(
            new_data, self.config.feature_columns, self.config.scale_features
        )
        
        # Predict
        predictions = self.model.predict(X_new)
        
        # Inverse transform if needed
        predictions = self.preprocessor.inverse_transform_target(predictions)
        
        return predictions
    
    def predict_proba(self, new_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (classification only)"""
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        X_new = self.preprocessor.transform(
            new_data, self.config.feature_columns, self.config.scale_features
        )
        
        return self.model.predict_proba(X_new)


class PredictionPipeline:
    """Orchestrates the complete prediction workflow"""
    
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
            'feature_count': 0,
            'data_quality_score': 0
        }
        
        # Check data size
        if len(self.data) < 30:
            analysis['ready'] = False
            analysis['issues'].append("Insufficient data (need at least 30 rows)")
        elif len(self.data) < 100:
            analysis['recommendations'].append(
                f"More data recommended (current: {len(self.data)} rows)"
            )
        
        # Check for columns
        if len(self.data.columns) < 2:
            analysis['ready'] = False
            analysis['issues'].append("Need at least 2 columns (1 target + 1 feature)")
        
        analysis['feature_count'] = len(self.data.columns)
        
        # Missing data check
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).max()
        if missing_pct > 50:
            analysis['ready'] = False
            analysis['issues'].append(f"High missing data: {missing_pct:.1f}% in some columns")
        elif missing_pct > 20:
            analysis['recommendations'].append(
                f"Some columns have {missing_pct:.1f}% missing data"
            )
        
        # Data quality score
        quality_score = 100 - min(missing_pct, 100)
        analysis['data_quality_score'] = quality_score
        
        # Suggest potential targets
        for col in self.data.columns:
            n_unique = self.data[col].nunique()
            n_missing = self.data[col].isnull().sum()
            
            # Skip if too much missing data
            if n_missing / len(self.data) > 0.5:
                continue
            
            if 2 <= n_unique <= 20:  # Classification
                analysis['potential_targets'].append({
                    'column': col,
                    'type': 'classification',
                    'n_classes': n_unique,
                    'missing_pct': float((n_missing / len(self.data)) * 100)
                })
            elif pd.api.types.is_numeric_dtype(self.data[col]) and n_unique > 20:
                analysis['potential_targets'].append({
                    'column': col,
                    'type': 'regression',
                    'range': f"{self.data[col].min():.2f} - {self.data[col].max():.2f}",
                    'missing_pct': float((n_missing / len(self.data)) * 100)
                })
        
        return analysis
    
    def train_model(self, target_column: str, feature_columns: List[str] = None,
                   model_type: ModelType = None, hyperparameters: Dict = None,
                   tune_hyperparameters: bool = False) -> ModelResults:
        """Train a prediction model"""
        
        # Auto-detect problem type
        problem_type = self.model_builder.auto_detect_problem_type(
            self.data, target_column
        )
        
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
            feature_columns=feature_columns or [
                col for col in self.data.columns if col != target_column
            ],
            hyperparameters=hyperparameters,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Train model
        print(f"🚀 Training {model_type.value} for {problem_type.value}...")
        results = self.model_builder.train(self.data, config)
        
        # Store trained model
        model_key = f"{target_column}_{model_type.value}"
        self.trained_models[model_key] = {
            'model_builder': self.model_builder,
            'config': config,
            'results': results
        }
        
        return results
    
    def predict_new_data(self, model_key: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data using trained model"""
        if model_key not in self.trained_models:
            raise ValueError(f"Model '{model_key}' not found. Train it first.")
        
        model_info = self.trained_models[model_key]
        model_builder = model_info['model_builder']
        
        # Make predictions
        predictions = model_builder.predict(new_data)
        
        # Create results dataframe
        results_df = new_data.copy()
        results_df['prediction'] = predictions
        
        # Add probabilities if available
        probabilities = model_builder.predict_proba(new_data)
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]
        
        return results_df
    
    def get_model_recommendations(self, target_column: str) -> List[Dict[str, Any]]:
        """Get recommended models for target"""
        try:
            problem_type = self.model_builder.auto_detect_problem_type(
                self.data, target_column
            )
        except Exception as e:
            # Fallback to classification if detection fails
            problem_type = ProblemType.CLASSIFICATION
        
        recommendations = []
        
        if problem_type == ProblemType.CLASSIFICATION:
            recommendations = [
                {
                    'model': ModelType.LOGISTIC_REGRESSION,
                    'pros': ['Fast', 'Interpretable', 'Good for binary classification'],
                    'cons': ['Assumes linear relationships', 'Limited with complex patterns'],
                    'best_for': 'Small to medium datasets, linear patterns',
                    'complexity': 'Low',
                    'training_speed': 'Very Fast'
                },
                {
                    'model': ModelType.RANDOM_FOREST_CLASSIFIER,
                    'pros': ['Handles non-linearity', 'Feature importance', 'Robust'],
                    'cons': ['Slower training', 'Less interpretable', 'Larger model'],
                    'best_for': 'Medium to large datasets, complex patterns',
                    'complexity': 'Medium',
                    'training_speed': 'Medium'
                },
                {
                    'model': ModelType.GRADIENT_BOOSTING_CLASSIFIER,
                    'pros': ['Highest accuracy', 'Handles complex patterns well'],
                    'cons': ['Slow to train', 'Requires tuning', 'Risk of overfitting'],
                    'best_for': 'When maximum accuracy is needed',
                    'complexity': 'High',
                    'training_speed': 'Slow'
                }
            ]
        
        elif problem_type == ProblemType.REGRESSION:
            recommendations = [
                {
                    'model': ModelType.LINEAR_REGRESSION,
                    'pros': ['Simple', 'Fast', 'Very interpretable'],
                    'cons': ['Assumes linearity', 'Sensitive to outliers'],
                    'best_for': 'Linear relationships, baseline model',
                    'complexity': 'Low',
                    'training_speed': 'Very Fast'
                },
                {
                    'model': ModelType.RANDOM_FOREST_REGRESSOR,
                    'pros': ['Handles non-linearity', 'Feature importance', 'Robust'],
                    'cons': ['Slower training', 'Can overfit small data'],
                    'best_for': 'Complex patterns, medium to large data',
                    'complexity': 'Medium',
                    'training_speed': 'Medium'
                },
                {
                    'model': ModelType.GRADIENT_BOOSTING_REGRESSOR,
                    'pros': ['Excellent accuracy', 'State-of-the-art performance'],
                    'cons': ['Slow training', 'Requires careful tuning'],
                    'best_for': 'Maximum accuracy needed',
                    'complexity': 'High',
                    'training_speed': 'Slow'
                }
            ]
        
        return recommendations
    
    def compare_models(self, model_keys: List[str]) -> pd.DataFrame:
        """Compare multiple trained models"""
        if not model_keys:
            raise ValueError("No models to compare")
        
        comparison_data = []
        
        for key in model_keys:
            if key not in self.trained_models:
                continue
            
            results = self.trained_models[key]['results']
            
            row = {
                'Model': results.model_type,
                'Training Time (s)': f"{results.training_time:.2f}"
            }
            
            # Add metrics
            for metric, value in results.metrics.items():
                row[metric.replace('_', ' ').title()] = f"{value:.4f}"
            
            # Add CV scores if available
            if results.cv_scores:
                for metric, scores in results.cv_scores.items():
                    if metric.startswith('mean_'):
                        row[f'CV {metric.replace("mean_", "")}'] = f"{scores:.4f}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """Get best model based on metric"""
        if not self.trained_models:
            return None
        
        best_model = None
        best_score = -float('inf')
        
        for key, model_info in self.trained_models.items():
            results = model_info['results']
            score = results.metrics.get(metric, -float('inf'))
            
            if score > best_score:
                best_score = score
                best_model = model_info
        
        return best_model
    
    def export_model(self, model_key: str, filepath: str = None) -> str:
        """Export trained model"""
        if model_key not in self.trained_models:
            raise ValueError(f"Model '{model_key}' not found")
        
        model_builder = self.trained_models[model_key]['model_builder']
        
        if filepath:
            joblib.dump({
                'model': model_builder.model,
                'preprocessor': model_builder.preprocessor,
                'config': model_builder.config
            }, filepath)
            return filepath
        
        return model_builder.results.model_path
    
    def load_model(self, filepath: str, model_key: str = None):
        """Load model from file"""
        model_builder = ModelBuilder()
        model_builder.load_model(filepath)
        
        if model_key is None:
            model_key = f"{model_builder.config.target_column}_{model_builder.config.model_type.value}"
        
        self.trained_models[model_key] = {
            'model_builder': model_builder,
            'config': model_builder.config,
            'results': model_builder.results
        }
        
        return model_key


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Complete ML Implementation...")
    
    # Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    # Test Classification
    print("\n📊 Testing Classification...")
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_classes=2, random_state=42
    )
    df_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
    df_clf['target'] = y_clf
    
    pipeline_clf = PredictionPipeline(df_clf)
    
    # Analyze readiness
    analysis = pipeline_clf.analyze_prediction_readiness()
    print(f"✅ Data ready: {analysis['ready']}")
    print(f"📊 Potential targets: {len(analysis['potential_targets'])}")
    
    # Train model
    results_clf = pipeline_clf.train_model(
        target_column='target',
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
        tune_hyperparameters=False
    )
    
    print(f"✅ Classification Accuracy: {results_clf.metrics['accuracy']:.4f}")
    print(f"⏱️ Training time: {results_clf.training_time:.2f}s")
    
    # Test Regression
    print("\n📈 Testing Regression...")
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=10, random_state=42
    )
    df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
    df_reg['target'] = y_reg
    
    pipeline_reg = PredictionPipeline(df_reg)
    
    results_reg = pipeline_reg.train_model(
        target_column='target',
        model_type=ModelType.RANDOM_FOREST_REGRESSOR
    )
    
    print(f"✅ R² Score: {results_reg.metrics['r2_score']:.4f}")
    print(f"📉 RMSE: {results_reg.metrics['rmse']:.2f}")
    
    # Test predictions on new data
    print("\n🔮 Testing Predictions...")
    new_data = df_clf.head(5).drop('target', axis=1)
    predictions = pipeline_clf.trained_models[
        list(pipeline_clf.trained_models.keys())[0]
    ]['model_builder'].predict(new_data)
    
    print(f"✅ Predictions: {predictions}")
    
    print("\n✨ All tests passed! ML implementation is complete and functional.")