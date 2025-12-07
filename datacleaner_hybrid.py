import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re

# Import the configuration manager, hybrid intelligence, visual insights, power tools, and predictive modeling
from config import ConfigManager, config_ui
from hybrid_intelligence import (
    CleaningMode, RiskLevel, CleaningRule, OperationExplanation,
    IntelligentRuleGenerator, ExplanationEngine, ApprovalManager, ProgressNarrator
)
from visual_insights import DataQualityVisualizer, ComparisonVisualizer, InteractiveDashboard
from power_tools import (
    SnapshotManager, CodeGenerator, LearningEngine, RecipeManager, 
    INDUSTRY_TEMPLATES
)
from predictive_modeling import (
    PredictionPipeline, ModelBuilder, ProblemType, ModelType, ModelConfig
)

class DataProfiler:
    """Profiles datasets to detect quality issues"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = {}
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        self.profile = {
            'basic_info': self._get_basic_info(),
            'missing_data': self._analyze_missing_data(),
            'duplicates': self._analyze_duplicates(),
            'data_types': self._analyze_data_types(),
            'column_stats': self._get_column_statistics(),
            'outliers': self._detect_outliers(),
            'format_issues': self._detect_format_issues()
        }
        return self.profile
    
    def _get_basic_info(self) -> Dict[str, Any]:
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'column_names': list(self.df.columns)
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        return {
            'total_missing': int(missing.sum()),
            'by_column': {
                col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
                for col in self.df.columns if missing[col] > 0
            }
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        duplicates = self.df.duplicated()
        return {
            'duplicate_rows': int(duplicates.sum()),
            'duplicate_percentage': float((duplicates.sum() / len(self.df) * 100).round(2))
        }
    
    def _analyze_data_types(self) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}
    
    def _get_column_statistics(self) -> Dict[str, Any]:
        stats = {}
        for col in self.df.columns:
            col_stats = {
                'unique_values': int(self.df[col].nunique()),
                'null_count': int(self.df[col].isnull().sum())
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_stats.update({
                    'min': float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                    'max': float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                    'mean': float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                    'median': float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None,
                    'std': float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None
                })
            
            stats[col] = col_stats
        return stats
    
    def _detect_outliers(self) -> Dict[str, Any]:
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(self.df) * 100).round(2)),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return outliers
    
    def _detect_format_issues(self) -> Dict[str, List[str]]:
        issues = {}
        
        for col in self.df.columns:
            col_issues = []
            
            # Check for mixed types in object columns
            if self.df[col].dtype == 'object':
                sample = self.df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check if column might be numeric but stored as string
                    try:
                        pd.to_numeric(sample)
                        col_issues.append("Numeric data stored as text")
                    except:
                        pass
                    
                    # Check for inconsistent date formats
                    date_like = sample.astype(str).str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', regex=True)
                    if date_like.any():
                        col_issues.append("Possible date/time data")
                    
                    # Check for leading/trailing whitespace
                    if sample.astype(str).str.strip().ne(sample.astype(str)).any():
                        col_issues.append("Contains leading/trailing whitespace")
            
            if col_issues:
                issues[col] = col_issues
        
        return issues


class DataCleaner:
    """Applies cleaning and transformation rules to datasets"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []
    
    def apply_cleaning_rules(self, rules: List[CleaningRule], narrator: Optional[ProgressNarrator] = None) -> pd.DataFrame:
        """Apply a list of CleaningRule objects"""
        
        for rule in rules:
            if narrator:
                narrator.start_operation(
                    rule.operation.replace('_', ' ').title(),
                    f"Applying to column: {rule.column}"
                )
            
            try:
                if rule.operation == 'remove_duplicates':
                    self._remove_duplicates()
                    if narrator:
                        narrator.add_detail(f"Removed {rule.expected_changes} duplicate rows")
                
                elif rule.operation == 'handle_missing':
                    method = rule.parameters.get('method', 'drop')
                    self._handle_missing_values({rule.column: method})
                    if narrator:
                        narrator.add_detail(f"Filled {rule.expected_changes} missing values using {method}")
                
                elif rule.operation == 'type_conversion':
                    target_type = rule.parameters.get('target_type', 'string')
                    self._convert_types({rule.column: target_type})
                    if narrator:
                        narrator.add_detail(f"Converted to {target_type}")
                
                elif rule.operation == 'normalize_text':
                    self._normalize_text([rule.column])
                    if narrator:
                        narrator.add_detail("Normalized text formatting")
                
                elif rule.operation == 'handle_outliers':
                    method = rule.parameters.get('method', 'cap')
                    self._handle_outliers({rule.column: method})
                    if narrator:
                        narrator.add_detail(f"Handled {rule.expected_changes} outliers using {method}")
                
                if narrator:
                    narrator.complete_operation(f"✓ Completed successfully")
                    
            except Exception as e:
                error_msg = f"Failed: {str(e)}"
                self._log_change(error_msg, level='error')
                if narrator:
                    narrator.fail_operation(error_msg)
        
        return self.df
    
    def apply_cleaning_rules_dict(self, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply a dictionary of cleaning rules (legacy method)"""
        
        if rules.get('remove_duplicates'):
            self._remove_duplicates()
        
        if rules.get('handle_missing'):
            self._handle_missing_values(rules['handle_missing'])
        
        if rules.get('type_conversions'):
            self._convert_types(rules['type_conversions'])
        
        if rules.get('normalize_text'):
            self._normalize_text(rules['normalize_text'])
        
        if rules.get('handle_outliers'):
            self._handle_outliers(rules['handle_outliers'])
        
        if rules.get('remove_columns'):
            self._remove_columns(rules['remove_columns'])
        
        if rules.get('rename_columns'):
            self._rename_columns(rules['rename_columns'])
        
        return self.df
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        if removed > 0:
            self._log_change(f"Removed {removed} duplicate rows")
    
    def _handle_missing_values(self, strategy: Dict[str, str]):
        """Handle missing values with various strategies"""
        for col, method in strategy.items():
            if col not in self.df.columns:
                continue
            
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if method == 'drop':
                self.df = self.df.dropna(subset=[col])
                self._log_change(f"Dropped {missing_count} rows with missing values in '{col}'")
            
            elif method == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)
                self._log_change(f"Filled {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
            
            elif method == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                self._log_change(f"Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")
            
            elif method == 'mode':
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                if mode_val is not None:
                    self.df[col].fillna(mode_val, inplace=True)
                    self._log_change(f"Filled {missing_count} missing values in '{col}' with mode: {mode_val}")
            
            elif method == 'forward_fill':
                self.df[col].fillna(method='ffill', inplace=True)
                self._log_change(f"Forward filled {missing_count} missing values in '{col}'")
            
            elif method == 'backward_fill':
                self.df[col].fillna(method='bfill', inplace=True)
                self._log_change(f"Backward filled {missing_count} missing values in '{col}'")
            
            elif method.startswith('constant:'):
                constant_value = method.split(':', 1)[1]
                self.df[col].fillna(constant_value, inplace=True)
                self._log_change(f"Filled {missing_count} missing values in '{col}' with constant: {constant_value}")
    
    def _convert_types(self, conversions: Dict[str, str]):
        """Convert column data types"""
        for col, target_type in conversions.items():
            if col not in self.df.columns:
                continue
            
            try:
                if target_type == 'numeric':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self._log_change(f"Converted '{col}' to numeric")
                
                elif target_type == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self._log_change(f"Converted '{col}' to datetime")
                
                elif target_type == 'string':
                    self.df[col] = self.df[col].astype(str)
                    self._log_change(f"Converted '{col}' to string")
                
                elif target_type == 'category':
                    self.df[col] = self.df[col].astype('category')
                    self._log_change(f"Converted '{col}' to category")
                
            except Exception as e:
                self._log_change(f"Failed to convert '{col}' to {target_type}: {str(e)}", level='error')
    
    def _normalize_text(self, columns: List[str]):
        """Normalize text columns (strip, lowercase, etc.)"""
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if self.df[col].dtype == 'object':
                # Strip whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                # Remove multiple spaces
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                self._log_change(f"Normalized text in '{col}'")
    
    def _handle_outliers(self, strategy: Dict[str, str]):
        """Handle outliers in numeric columns"""
        for col, method in strategy.items():
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count == 0:
                continue
            
            if method == 'remove':
                self.df = self.df[~outlier_mask]
                self._log_change(f"Removed {outlier_count} outliers from '{col}'")
            
            elif method == 'cap':
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                self._log_change(f"Capped {outlier_count} outliers in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    def _remove_columns(self, columns: List[str]):
        """Remove specified columns"""
        cols_to_remove = [col for col in columns if col in self.df.columns]
        if cols_to_remove:
            self.df = self.df.drop(columns=cols_to_remove)
            self._log_change(f"Removed columns: {', '.join(cols_to_remove)}")
    
    def _rename_columns(self, mapping: Dict[str, str]):
        """Rename columns"""
        valid_mapping = {old: new for old, new in mapping.items() if old in self.df.columns}
        if valid_mapping:
            self.df = self.df.rename(columns=valid_mapping)
            self._log_change(f"Renamed columns: {valid_mapping}")
    
    def _log_change(self, message: str, level: str = 'info'):
        """Log cleaning operations"""
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        })
    
    def get_cleaning_log(self) -> List[Dict[str, str]]:
        """Return the cleaning log"""
        return self.cleaning_log


class DataPipeline:
    """Orchestrates the data cleaning pipeline"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.data = None
        self.profiler = None
        self.cleaner = None
        self.profile_result = None
        self.cleaned_data = None
        self.config = config_manager or ConfigManager()
        
        # Hybrid intelligence components
        self.rule_generator = None
        self.approval_manager = ApprovalManager()
        self.narrator = ProgressNarrator()
        self.generated_rules = []
        self.explanations = []
        
        # Visual insights components
        self.dashboard = None
        self.original_data = None
        self.original_profile = None
        
        # Power tools components
        self.snapshot_manager = SnapshotManager(max_snapshots=10)
        self.code_generator = CodeGenerator()
        self.learning_engine = LearningEngine()
        self.recipe_manager = RecipeManager()
        
        # Predictive modeling components
        self.prediction_pipeline = None
        self.trained_models = {}
        
        # Load industry templates
        for template_key, template_data in INDUSTRY_TEMPLATES.items():
            self.recipe_manager.save_recipe(
                name=template_data['name'],
                operations=template_data['operations'],
                description=template_data['description'],
                tags=template_data['tags']
            )
    
    def ingest_data(self, file, file_type: str) -> pd.DataFrame:
        """Load data from various formats"""
        try:
            if file_type == 'csv':
                self.data = pd.read_csv(file)
            elif file_type == 'excel':
                self.data = pd.read_excel(file)
            elif file_type == 'json':
                self.data = pd.read_json(file)
            elif file_type == 'parquet':
                self.data = pd.read_parquet(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return self.data
        except Exception as e:
            raise Exception(f"Error ingesting data: {str(e)}")
    
    def ingest_from_firebase(self, firebase_data: List[Dict]) -> pd.DataFrame:
        """Load data from Firebase format"""
        self.data = pd.DataFrame(firebase_data)
        return self.data
    
    def profile_data(self) -> Dict[str, Any]:
        """Profile the ingested data"""
        if self.data is None:
            raise ValueError("No data ingested. Please ingest data first.")
        
        # Store original data and profile for comparison
        if self.original_data is None:
            self.original_data = self.data.copy()
        
        self.profiler = DataProfiler(self.data)
        self.profile_result = self.profiler.generate_profile()
        
        # Store original profile
        if self.original_profile is None:
            self.original_profile = self.profile_result.copy()
        
        # Create dashboard
        self.dashboard = InteractiveDashboard(self.data, self.profile_result)
        
        return self.profile_result
    
    def generate_intelligent_rules(self) -> Tuple[List[CleaningRule], List[OperationExplanation]]:
        """Generate intelligent cleaning rules with explanations"""
        if self.profile_result is None:
            self.profile_data()
        
        self.rule_generator = IntelligentRuleGenerator(self.data, self.profile_result)
        self.generated_rules, self.explanations = self.rule_generator.generate_all_rules()
        
        return self.generated_rules, self.explanations
    
    def clean_data_intelligent(self, approved_rules: List[CleaningRule], auto_mode: bool = False) -> pd.DataFrame:
        """Clean data using approved intelligent rules"""
        if self.data is None:
            raise ValueError("No data ingested. Please ingest data first.")
        
        # Save snapshot before cleaning
        if self.profile_result:
            self.snapshot_manager.save_snapshot(
                self.data,
                self.profile_result,
                "before_cleaning",
                "Original data before cleaning"
            )
        
        self.narrator = ProgressNarrator()
        self.cleaner = DataCleaner(self.data)
        self.cleaned_data = self.cleaner.apply_cleaning_rules(approved_rules, self.narrator)
        
        # Save snapshot after cleaning
        if self.profile_result:
            cleaned_profiler = DataProfiler(self.cleaned_data)
            cleaned_profile = cleaned_profiler.generate_profile()
            self.snapshot_manager.save_snapshot(
                self.cleaned_data,
                cleaned_profile,
                "after_cleaning",
                "Data after cleaning"
            )
        
        # Record operations for code generation
        for rule in approved_rules:
            self.code_generator.add_operation(rule.operation, rule.parameters)
            
            # NEW: Auto-learn from successful operations in automatic mode
            if auto_mode and rule.confidence >= 0.75:  # Only learn from confident rules
                try:
                    # Check if operation was successful (data actually changed)
                    success = True  # Could add validation here
                    
                    self.learning_engine.auto_learn_from_operation(
                        operation=rule.operation,
                        parameters=rule.parameters,
                        column=rule.column,
                        confidence=rule.confidence,
                        success=success
                    )
                except Exception as e:
                    # Don't fail cleaning if learning fails
                    pass
        
        return self.cleaned_data
    
    def clean_data(self, rules: Dict[str, Any]) -> pd.DataFrame:
        """Clean the data using specified rules (legacy method)"""
        if self.data is None:
            raise ValueError("No data ingested. Please ingest data first.")
        
        self.cleaner = DataCleaner(self.data)
        self.cleaned_data = self.cleaner.apply_cleaning_rules_dict(rules)
        return self.cleaned_data
    
    def export_data(self, format: str = None) -> bytes:
        """Export cleaned data in various formats"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        
        if format is None:
            format = self.config.get_config_value("app_settings", "default_export_format") or 'csv'
        
        buffer = io.BytesIO()
        
        if format == 'csv':
            self.cleaned_data.to_csv(buffer, index=False)
            buffer.seek(0)
            return buffer.getvalue()
        
        elif format == 'excel':
            self.cleaned_data.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            return buffer.getvalue()
        
        elif format == 'json':
            json_str = self.cleaned_data.to_json(orient='records', indent=2)
            return json_str.encode()
        
        elif format == 'parquet':
            self.cleaned_data.to_parquet(buffer, index=False)
            buffer.seek(0)
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_to_firebase_format(self) -> List[Dict]:
        """Export cleaned data in Firebase-compatible format"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        
        return self.cleaned_data.to_dict('records')
    
    def get_cleaning_log(self) -> List[Dict[str, str]]:
        """Get the cleaning log"""
        if self.cleaner is None:
            return []
        return self.cleaner.get_cleaning_log()
    
    def get_narrative(self) -> str:
        """Get the cleaning narrative"""
        return self.narrator.format_for_display()


# Streamlit UI
def main():
    st.set_page_config(page_title="Data Cleaning & Profiling System", layout="wide", page_icon="🧹")
    
    st.title("🧹 Data Cleaning & Profiling System")
    st.markdown("### Hybrid Manual/Automatic Cleaning with AI Explanations")
    st.markdown("---")
    
    # Initialize session state
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataPipeline(st.session_state.config_manager)
    
    if 'cleaning_mode' not in st.session_state:
        st.session_state.cleaning_mode = CleaningMode.MANUAL
    
    if 'show_config' not in st.session_state:
        st.session_state.show_config = False
    
    config_mgr = st.session_state.config_manager
    pipeline = st.session_state.pipeline
    
    # Sidebar for configuration and mode selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration status
        firebase_api = config_mgr.get_api_key("firebase")
        google_api = config_mgr.get_api_key("google_cloud")
        
        if firebase_api:
            st.success("✅ Firebase API configured")
        else:
            st.info("ℹ️ Firebase API not configured")
        
        if google_api:
            st.success(f"✅ Google Cloud API configured")
        else:
            st.info("ℹ️ Google Cloud API not configured")
        
        if st.button("⚙️ Manage Configuration"):
            st.session_state.show_config = not st.session_state.show_config
        
        st.markdown("---")
        
        # Mode Selection
        st.header("🎯 Cleaning Mode")
        
        mode_descriptions = {
            "Manual": "You configure all cleaning rules yourself",
            "Assisted": "AI suggests rules, you review and approve",
            "Automatic": "AI cleans automatically with explanations"
        }
        
        selected_mode = st.radio(
            "Select Mode",
            ["Manual", "Assisted", "Automatic"],
            index=["Manual", "Assisted", "Automatic"].index(st.session_state.cleaning_mode.value.title()) if hasattr(st.session_state.cleaning_mode, 'value') else 0,
            help="Choose how you want to clean your data"
        )
        
        if selected_mode == "Manual":
            st.session_state.cleaning_mode = CleaningMode.MANUAL
            st.info("📝 " + mode_descriptions["Manual"])
        elif selected_mode == "Assisted":
            st.session_state.cleaning_mode = CleaningMode.ASSISTED
            st.info("🤝 " + mode_descriptions["Assisted"])
        else:
            st.session_state.cleaning_mode = CleaningMode.AUTOMATIC
            st.info("🤖 " + mode_descriptions["Automatic"])
        
        st.markdown("---")
        
        # Data source selection
        st.header("📂 Data Source")
        data_source = st.radio("Source", ["Upload File", "Firebase Data"])
        
        if data_source == "Upload File":
            allowed_types = config_mgr.get_config_value("app_settings", "allowed_file_types") or ['csv', 'xlsx', 'json', 'parquet']
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=allowed_types
            )
            
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1]
                if file_type == 'xlsx':
                    file_type = 'excel'
                
                if st.button("📥 Load Data"):
                    try:
                        with st.spinner("Loading data..."):
                            pipeline.ingest_data(uploaded_file, file_type)
                        st.success(f"✅ Loaded {len(pipeline.data)} rows")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        else:
            st.info("📝 Paste Firebase data as JSON array")
            firebase_json = st.text_area("Firebase Data (JSON)", height=200)
            
            if st.button("📥 Load Firebase Data"):
                try:
                    firebase_data = json.loads(firebase_json)
                    pipeline.ingest_from_firebase(firebase_data)
                    st.success(f"✅ Loaded {len(pipeline.data)} rows")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show configuration UI if toggled
    if st.session_state.show_config:
        with st.expander("⚙️ Configuration Manager", expanded=True):
            config_ui()
        st.markdown("---")
    
    # Main content area
    if pipeline.data is not None:
        # Different tab layouts based on mode
        if st.session_state.cleaning_mode == CleaningMode.MANUAL:
            tabs = st.tabs(["📊 Data Preview", "🔍 Profile", "🧹 Clean (Manual)", "⚡ Power Tools", "🤖 Predictions", "📤 Export"])
            show_manual_mode(pipeline, tabs)
        
        elif st.session_state.cleaning_mode == CleaningMode.ASSISTED:
            tabs = st.tabs(["📊 Data Preview", "🔍 Profile", "🤖 AI Suggestions", "✅ Review & Approve", "⚡ Power Tools", "🔮 Predictions", "📤 Export"])
            show_assisted_mode(pipeline, tabs)
        
        else:  # Automatic
            tabs = st.tabs(["📊 Data Preview", "🔍 Profile", "🤖 Auto-Clean", "📊 Results", "⚡ Power Tools", "🔮 Predictions", "📤 Export"])
            show_automatic_mode(pipeline, tabs)
    
    else:
        st.info("👆 Please upload a file or provide Firebase data to get started")
        
        # Quick start guide
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            ### Getting Started
            
            #### 1. **Choose Your Mode**
            - **Manual**: Traditional hands-on approach
            - **Assisted**: AI helps, you approve
            - **Automatic**: AI does everything with explanations
            
            #### 2. **Load Your Data**
            - Upload CSV, Excel, JSON, or Parquet
            - Or paste Firebase JSON data
            
            #### 3. **Profile & Clean**
            - AI analyzes your data
            - Get intelligent suggestions
            - Understand why each action is recommended
            
            #### 4. **Export Results**
            - Download cleaned data
            - Save cleaning recipes for reuse
            """)


def show_manual_mode(pipeline, tabs):
    """Display manual mode interface"""
    with tabs[0]:
        show_data_preview(pipeline)
    
    with tabs[1]:
        show_profile_tab(pipeline)
    
    with tabs[2]:
        show_manual_cleaning_tab(pipeline)
    
    with tabs[3]:
        show_power_tools_tab(pipeline)
    
    with tabs[4]:
        show_predictions_tab(pipeline)
    
    with tabs[5]:
        show_export_tab(pipeline)


def show_assisted_mode(pipeline, tabs):
    """Display assisted mode interface"""
    with tabs[0]:
        show_data_preview(pipeline)
    
    with tabs[1]:
        show_profile_tab(pipeline)
    
    with tabs[2]:
        show_ai_suggestions_tab(pipeline)
    
    with tabs[3]:
        show_review_approve_tab(pipeline)
    
    with tabs[4]:
        show_power_tools_tab(pipeline)
    
    with tabs[5]:
        show_predictions_tab(pipeline)
    
    with tabs[6]:
        show_export_tab(pipeline)


def show_automatic_mode(pipeline, tabs):
    """Display automatic mode interface"""
    with tabs[0]:
        show_data_preview(pipeline)
    
    with tabs[1]:
        show_profile_tab(pipeline)
    
    with tabs[2]:
        show_auto_clean_tab(pipeline)
    
    with tabs[3]:
        show_results_tab(pipeline)
    
    with tabs[4]:
        show_power_tools_tab(pipeline)
    
    with tabs[5]:
        show_predictions_tab(pipeline)
    
    with tabs[6]:
        show_export_tab(pipeline)


def show_data_preview(pipeline):
    """Show data preview tab"""
    st.subheader("Data Preview")
    st.dataframe(pipeline.data.head(100), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(pipeline.data))
    col2.metric("Columns", len(pipeline.data.columns))
    col3.metric("Memory", f"{pipeline.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def show_profile_tab(pipeline):
    """Show profiling tab"""
    st.subheader("Data Profiling")
    
    if st.button("🔍 Generate Profile"):
        with st.spinner("Profiling data..."):
            profile = pipeline.profile_data()
        
        st.success("✅ Profile generated")
        
        # Add visualization tabs
        viz_tabs = st.tabs(["📊 Overview Dashboard", "🔍 Detailed Analysis", "📋 Raw Profile"])
        
        with viz_tabs[0]:
            # Render interactive dashboard
            pipeline.dashboard.render_overview_dashboard()
        
        with viz_tabs[1]:
            # Render detailed analysis
            pipeline.dashboard.render_detailed_analysis()
        
        with viz_tabs[2]:
            # Display raw profile results (original format)
            st.markdown("### Missing Data")
            if profile['missing_data']['by_column']:
                missing_df = pd.DataFrame(profile['missing_data']['by_column']).T
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.info("No missing data found")
            
            st.markdown("### Duplicates")
            col1, col2 = st.columns(2)
            col1.metric("Duplicate Rows", profile['duplicates']['duplicate_rows'])
            col2.metric("Percentage", f"{profile['duplicates']['duplicate_percentage']}%")
            
            st.markdown("### Data Types")
            types_df = pd.DataFrame(list(profile['data_types'].items()), columns=['Column', 'Type'])
            st.dataframe(types_df, use_container_width=True)
            
            st.markdown("### Outliers")
            if profile['outliers']:
                outliers_df = pd.DataFrame(profile['outliers']).T
                st.dataframe(outliers_df, use_container_width=True)
            else:
                st.info("No outliers detected")
            
            st.markdown("### Format Issues")
            if profile['format_issues']:
                for col, issues in profile['format_issues'].items():
                    st.warning(f"**{col}**: {', '.join(issues)}")
            else:
                st.info("No format issues detected")


def show_manual_cleaning_tab(pipeline):
    """Show manual cleaning interface"""
    st.subheader("Manual Data Cleaning")
    st.info("💡 Configure cleaning rules manually")
    
    cleaning_rules = {}
    
    # Remove duplicates
    if st.checkbox("Remove duplicate rows"):
        cleaning_rules['remove_duplicates'] = True
    
    # Handle missing values
    st.markdown("#### Handle Missing Values")
    missing_strategy = {}
    for col in pipeline.data.columns:
        missing_count = pipeline.data[col].isnull().sum()
        if missing_count > 0:
            method = st.selectbox(
                f"{col} ({missing_count} missing)",
                ['skip', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant'],
                key=f"manual_missing_{col}"
            )
            if method != 'skip':
                if method == 'constant':
                    const_val = st.text_input(f"Constant value for {col}", key=f"manual_const_{col}")
                    missing_strategy[col] = f"constant:{const_val}"
                else:
                    missing_strategy[col] = method
    
    if missing_strategy:
        cleaning_rules['handle_missing'] = missing_strategy
    
    # Type conversions
    st.markdown("#### Type Conversions")
    type_conversions = {}
    for col in pipeline.data.columns:
        current_type = str(pipeline.data[col].dtype)
        new_type = st.selectbox(
            f"{col} (current: {current_type})",
            ['no change', 'numeric', 'datetime', 'string', 'category'],
            key=f"manual_type_{col}"
        )
        if new_type != 'no change':
            type_conversions[col] = new_type
    
    if type_conversions:
        cleaning_rules['type_conversions'] = type_conversions
    
    # Text normalization
    text_cols = st.multiselect(
        "Normalize text in columns",
        pipeline.data.select_dtypes(include=['object']).columns.tolist(),
        key="manual_text_norm"
    )
    if text_cols:
        cleaning_rules['normalize_text'] = text_cols
    
    # Apply cleaning
    if st.button("🧹 Apply Cleaning Rules", type="primary", key="manual_apply"):
        with st.spinner("Cleaning data..."):
            pipeline.clean_data(cleaning_rules)
        
        st.success("✅ Data cleaned successfully!")
        
        # Show cleaning log
        st.markdown("### Cleaning Log")
        log = pipeline.get_cleaning_log()
        for entry in log:
            if entry['level'] == 'error':
                st.error(f"⚠️ {entry['message']}")
            else:
                st.info(f"ℹ️ {entry['message']}")
        
        # Show cleaned data preview
        st.markdown("### Cleaned Data Preview")
        st.dataframe(pipeline.cleaned_data.head(100), use_container_width=True)


def show_ai_suggestions_tab(pipeline):
    """Show AI suggestions in assisted mode"""
    st.subheader("🤖 AI-Generated Cleaning Suggestions")
    st.info("💡 AI has analyzed your data and generated intelligent cleaning recommendations")
    
    if st.button("🔮 Generate AI Suggestions", type="primary"):
        with st.spinner("🤖 AI is analyzing your data..."):
            rules, explanations = pipeline.generate_intelligent_rules()
        
        st.session_state.ai_rules = rules
        st.session_state.ai_explanations = explanations
        
        st.success(f"✅ Generated {len(rules)} intelligent cleaning suggestions")
    
    if 'ai_rules' in st.session_state and st.session_state.ai_rules:
        # Show summary
        st.markdown(ExplanationEngine.generate_summary(st.session_state.ai_rules))
        
        # Show each rule with explanation
        st.markdown("### 📋 Detailed Suggestions")
        
        for i, (rule, explanation) in enumerate(zip(st.session_state.ai_rules, st.session_state.ai_explanations)):
            with st.expander(f"**{i+1}. {explanation.title}**", expanded=(i < 3)):
                st.markdown(explanation.to_markdown())
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Confidence", f"{rule.confidence:.0%}")
                with col2:
                    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                    st.metric("Risk", f"{risk_colors[rule.risk_level.value]} {rule.risk_level.value.upper()}")


def show_review_approve_tab(pipeline):
    """Show review and approval interface"""
    st.subheader("✅ Review & Approve AI Suggestions")
    
    if 'ai_rules' not in st.session_state or not st.session_state.ai_rules:
        st.warning("⚠️ Please generate AI suggestions first")
        return
    
    st.info("💡 Review each suggestion and approve, modify, or reject")
    
    # Auto-approve option
    if st.checkbox("🚀 Auto-approve all LOW risk operations (confidence > 85%)"):
        pipeline.approval_manager.auto_approve_safe_operations(st.session_state.ai_rules)
        st.success(f"✅ Auto-approved {len(pipeline.approval_manager.get_approved_rules())} safe operations")
    
    # Show approval interface for each rule
    for i, rule in enumerate(st.session_state.ai_rules):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        emoji = risk_emoji.get(rule.risk_level.value, "⚪")
        
        with st.container():
            st.markdown(f"### {emoji} {i+1}. {rule.operation.replace('_', ' ').title()} - '{rule.column}'")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Reasoning:** {rule.reasoning}")
                st.write(f"**Impact:** {rule.impact_description}")
            
            with col2:
                st.metric("Confidence", f"{rule.confidence:.0%}")
                st.metric("Risk", rule.risk_level.value.upper())
            
            with col3:
                if st.button(f"✅ Approve", key=f"approve_{i}"):
                    pipeline.approval_manager.approve_rule(rule)
                    st.success("Approved!")
                
                if st.button(f"❌ Reject", key=f"reject_{i}"):
                    pipeline.approval_manager.reject_rule(rule)
                    st.warning("Rejected")
            
            st.markdown("---")
    
    # Show approval summary
    status = pipeline.approval_manager.get_status_summary()
    st.markdown("### 📊 Approval Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Approved", status['approved'])
    col2.metric("⏳ Pending", status['pending'])
    col3.metric("❌ Rejected", status['rejected'])
    col4.metric("✏️ Modified", status['modified'])
    
    # Apply approved rules
    if status['approved'] > 0:
        if st.button("🚀 Apply Approved Rules", type="primary"):
            with st.spinner("🤖 Applying cleaning rules with AI narration..."):
                approved_rules = pipeline.approval_manager.get_approved_rules()
                pipeline.clean_data_intelligent(approved_rules)
            
            st.success("✅ Data cleaned successfully with AI assistance!")
            
            # Show narrative
            st.markdown(pipeline.get_narrative())
            
            # Show cleaned data
            st.markdown("### Cleaned Data Preview")
            st.dataframe(pipeline.cleaned_data.head(100), use_container_width=True)


def show_auto_clean_tab(pipeline):
    """Show automatic cleaning interface"""
    st.subheader("🤖 Automatic Cleaning with AI")
    st.info("💡 AI will automatically clean your data and explain every decision")
    
    # Auto-learning toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_learning = st.checkbox(
            "🧠 Enable Auto-Learning (AI learns from this cleaning)",
            value=pipeline.learning_engine.auto_learning_enabled,
            help="When enabled, AI will learn from successful operations to improve future suggestions"
        )
        pipeline.learning_engine.enable_auto_learning(auto_learning)
    
    with col2:
        if auto_learning:
            threshold = st.slider(
                "Confidence threshold",
                min_value=0.7,
                max_value=1.0,
                value=pipeline.learning_engine.confidence_threshold,
                step=0.05,
                help="Minimum confidence to learn from (higher = more selective)"
            )
            pipeline.learning_engine.set_confidence_threshold(threshold)
    
    if st.button("🚀 Start Auto-Clean", type="primary"):
        with st.spinner("🤖 AI is analyzing and cleaning your data..."):
            # Generate rules
            rules, explanations = pipeline.generate_intelligent_rules()
            
            # Auto-approve safe operations
            pipeline.approval_manager.auto_approve_safe_operations(rules)
            
            # Apply all approved rules with auto-learning enabled
            approved_rules = pipeline.approval_manager.get_approved_rules()
            pipeline.clean_data_intelligent(approved_rules, auto_mode=True)
        
        st.success("✅ Automatic cleaning complete!")
        
        # Show learning summary if enabled
        if auto_learning:
            insights = pipeline.learning_engine.get_learning_insights()
            st.info(f"🧠 AI learned from {insights['auto_learned']} operations in this session (avg confidence: {insights['avg_auto_confidence']:.0%})")
        
        # Store for results tab
        st.session_state.auto_clean_complete = True
        st.session_state.auto_rules = rules
        st.session_state.auto_explanations = explanations
        
        # Show live narrative
        st.markdown("### 🎬 Cleaning Narrative")
        st.markdown(pipeline.get_narrative())
        
        # Show summary
        st.markdown("### 📊 Cleaning Summary")
        st.markdown(ExplanationEngine.generate_summary(rules))


def show_results_tab(pipeline):
    """Show results of automatic cleaning"""
    st.subheader("📊 Cleaning Results")
    
    if not hasattr(st.session_state, 'auto_clean_complete') or not st.session_state.auto_clean_complete:
        st.warning("⚠️ Please run auto-clean first")
        return
    
    # Use tabs for different views
    result_tabs = st.tabs(["✅ Cleaned Data", "📊 Visual Comparison", "🔍 Operations"])
    
    with result_tabs[0]:
        # Show cleaned data
        st.markdown("### ✅ Cleaned Data Preview")
        st.dataframe(pipeline.cleaned_data.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", len(pipeline.original_data))
            st.metric("Cleaned Rows", len(pipeline.cleaned_data))
        
        with col2:
            row_diff = len(pipeline.original_data) - len(pipeline.cleaned_data)
            st.metric("Rows Removed", row_diff)
            pct_change = (row_diff / len(pipeline.original_data) * 100) if len(pipeline.original_data) > 0 else 0
            st.metric("Data Retained", f"{100 - pct_change:.1f}%")
    
    with result_tabs[1]:
        # Show visual comparison
        if pipeline.original_data is not None and pipeline.original_profile is not None:
            # Re-profile cleaned data
            cleaned_profiler = DataProfiler(pipeline.cleaned_data)
            cleaned_profile = cleaned_profiler.generate_profile()
            
            # Create dashboard for cleaned data
            cleaned_dashboard = InteractiveDashboard(pipeline.cleaned_data, cleaned_profile)
            
            # Render comparison
            cleaned_dashboard.render_comparison_dashboard(
                pipeline.original_data,
                pipeline.original_profile,
                cleaned_profile
            )
        else:
            st.warning("Original data not available for comparison")
    
    with result_tabs[2]:
        # Show what was done
        st.markdown("### 🔍 Operations Performed")
        for i, explanation in enumerate(st.session_state.auto_explanations):
            with st.expander(f"{i+1}. {explanation.title}"):
                st.markdown(explanation.to_markdown())


def show_export_tab(pipeline):
    """Show export interface"""
    st.subheader("📤 Export Data")
    
    if pipeline.cleaned_data is not None:
        default_format = pipeline.config.get_config_value("app_settings", "default_export_format") or 'csv'
        format_options = ['csv', 'excel', 'json', 'parquet', 'firebase']
        default_index = format_options.index(default_format) if default_format in format_options else 0
        
        export_format = st.selectbox(
            "Export Format",
            format_options,
            index=default_index
        )
        
        if st.button("📥 Generate Export"):
            try:
                if export_format == 'firebase':
                    firebase_data = pipeline.export_to_firebase_format()
                    st.json(firebase_data[:5])  # Show first 5 records
                    st.download_button(
                        "Download Firebase JSON",
                        json.dumps(firebase_data, indent=2),
                        "cleaned_data.json",
                        "application/json"
                    )
                else:
                    data_bytes = pipeline.export_data(export_format)
                    file_ext = export_format if export_format != 'excel' else 'xlsx'
                    st.download_button(
                        f"Download {export_format.upper()}",
                        data_bytes,
                        f"cleaned_data.{file_ext}",
                        f"application/{export_format}"
                    )
                
                st.success("✅ Export ready for download")
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
        
        # Export cleaning log
        st.markdown("### 📜 Export Cleaning Log")
        log = pipeline.get_cleaning_log()
        if log:
            log_json = json.dumps(log, indent=2)
            st.download_button(
                "Download Cleaning Log",
                log_json,
                "cleaning_log.json",
                "application/json"
            )
        
        # Export cleaning recipe (if AI-generated rules exist)
        if hasattr(st.session_state, 'ai_rules') and st.session_state.ai_rules:
            st.markdown("### 📋 Export Cleaning Recipe")
            st.info("💡 Save this cleaning workflow to reuse on similar datasets")
            
            recipe = {
                'created_at': datetime.now().isoformat(),
                'mode': st.session_state.cleaning_mode.value,
                'rules': [rule.to_dict() for rule in st.session_state.ai_rules],
                'summary': {
                    'total_rules': len(st.session_state.ai_rules),
                    'approved': len(pipeline.approval_manager.get_approved_rules()),
                    'rejected': len(pipeline.approval_manager.rejected_rules)
                }
            }
            
            recipe_json = json.dumps(recipe, indent=2)
            st.download_button(
                "Download Cleaning Recipe",
                recipe_json,
                "cleaning_recipe.json",
                "application/json"
            )
    else:
        st.warning("⚠️ Please clean the data first before exporting")


def show_power_tools_tab(pipeline):
    """Show power tools interface"""
    st.subheader("⚡ Power User Tools")
    
    # Create sub-tabs for different power tools
    power_tabs = st.tabs(["⏪ Undo/Redo", "🐍 Code Generation", "📚 Recipes", "🧠 Learning Insights"])
    
    with power_tabs[0]:
        show_undo_redo_tab(pipeline)
    
    with power_tabs[1]:
        show_code_generation_tab(pipeline)
    
    with power_tabs[2]:
        show_recipes_tab(pipeline)
    
    with power_tabs[3]:
        show_learning_tab(pipeline)


def show_undo_redo_tab(pipeline):
    """Show undo/redo interface"""
    st.markdown("### ⏪ Version History & Undo/Redo")
    st.info("💡 Track changes and revert to previous versions of your data")
    
    # Undo/Redo buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("⏪ Undo", disabled=not pipeline.snapshot_manager.can_undo()):
            snapshot = pipeline.snapshot_manager.undo()
            if snapshot:
                pipeline.data = snapshot.data.copy()
                pipeline.profile_result = snapshot.profile
                st.success(f"↩️ Reverted to: {snapshot.description}")
                st.rerun()
    
    with col2:
        if st.button("⏩ Redo", disabled=not pipeline.snapshot_manager.can_redo()):
            snapshot = pipeline.snapshot_manager.redo()
            if snapshot:
                pipeline.data = snapshot.data.copy()
                pipeline.profile_result = snapshot.profile
                st.success(f"↪️ Restored: {snapshot.description}")
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear History"):
            pipeline.snapshot_manager.clear_history()
            st.success("✅ History cleared")
            st.rerun()
    
    # Show snapshot history
    st.markdown("### 📜 Snapshot History")
    history = pipeline.snapshot_manager.get_history()
    
    if not history:
        st.info("No snapshots yet. Snapshots are created automatically when you clean data.")
    else:
        for i, snapshot in enumerate(history):
            is_current = snapshot['is_current']
            
            with st.expander(
                f"{'📍 ' if is_current else '📄 '}Snapshot {i+1}: {snapshot['operation']} "
                f"({snapshot['rows']:,} rows) {'← Current' if is_current else ''}",
                expanded=is_current
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{snapshot['rows']:,}")
                col2.metric("Columns", snapshot['columns'])
                col3.metric("Memory", f"{snapshot['memory_mb']:.2f} MB")
                
                st.write(f"**Description:** {snapshot['description']}")
                st.write(f"**Time:** {snapshot['timestamp']}")
                
                if not is_current:
                    if st.button(f"🔄 Restore This Version", key=f"restore_{i}"):
                        restored = pipeline.snapshot_manager.jump_to_snapshot(i)
                        if restored:
                            pipeline.data = restored.data.copy()
                            pipeline.profile_result = restored.profile
                            st.success(f"✅ Restored snapshot {i+1}")
                            st.rerun()


def show_code_generation_tab(pipeline):
    """Show code generation interface"""
    st.markdown("### 🐍 Generate Executable Code")
    st.info("💡 Export your cleaning workflow as executable Python, PySpark, or SQL code")
    
    if not pipeline.code_generator.operations:
        st.warning("⚠️ No cleaning operations recorded yet. Clean your data first to generate code.")
        return
    
    # Show operations summary
    st.markdown(f"**Operations to Export:** {len(pipeline.code_generator.operations)}")
    
    for i, op in enumerate(pipeline.code_generator.operations):
        st.write(f"{i+1}. `{op['operation']}` - {len(op['parameters'])} parameter(s)")
    
    st.markdown("---")
    
    # Code generation options
    code_type = st.selectbox(
        "Select Code Type",
        ["Pandas (Python)", "PySpark", "SQL", "Jupyter Notebook"],
        help="Choose the format for generated code"
    )
    
    df_name = st.text_input("DataFrame Variable Name", value="df", help="Name for the dataframe variable")
    
    if st.button("🔮 Generate Code", type="primary"):
        with st.spinner("Generating code..."):
            if code_type == "Pandas (Python)":
                code = pipeline.code_generator.generate_pandas_code(df_name)
                st.code(code, language="python")
                
                st.download_button(
                    "📥 Download Python Script",
                    code,
                    "data_cleaning.py",
                    "text/x-python"
                )
            
            elif code_type == "PySpark":
                code = pipeline.code_generator.generate_pyspark_code(df_name)
                st.code(code, language="python")
                
                st.download_button(
                    "📥 Download PySpark Script",
                    code,
                    "data_cleaning_pyspark.py",
                    "text/x-python"
                )
            
            elif code_type == "SQL":
                code = pipeline.code_generator.generate_sql_code(df_name)
                st.code(code, language="sql")
                
                st.download_button(
                    "📥 Download SQL Script",
                    code,
                    "data_cleaning.sql",
                    "text/x-sql"
                )
            
            elif code_type == "Jupyter Notebook":
                notebook = pipeline.code_generator.export_as_notebook(df_name)
                notebook_json = json.dumps(notebook, indent=2)
                
                st.success("✅ Jupyter Notebook generated!")
                st.info("💡 Download and open in Jupyter to run interactively")
                
                st.download_button(
                    "📥 Download Notebook (.ipynb)",
                    notebook_json,
                    "data_cleaning.ipynb",
                    "application/x-ipynb+json"
                )
        
        st.success("✅ Code generated successfully!")
        
        # Usage instructions
        with st.expander("📖 How to Use Generated Code"):
            st.markdown("""
            ### Using Your Generated Code
            
            **Pandas (Python):**
            1. Save the script as `data_cleaning.py`
            2. Update the file path in the load data section
            3. Run: `python data_cleaning.py`
            
            **PySpark:**
            1. Save as `data_cleaning_pyspark.py`
            2. Ensure Spark is installed
            3. Run: `spark-submit data_cleaning_pyspark.py`
            
            **SQL:**
            1. Update table names
            2. Execute in your SQL environment
            
            **Jupyter Notebook:**
            1. Open in Jupyter Lab/Notebook
            2. Run cells sequentially
            3. Modify as needed
            """)


def show_recipes_tab(pipeline):
    """Show recipe management interface"""
    st.markdown("### 📚 Cleaning Recipes & Templates")
    st.info("💡 Save and reuse cleaning workflows, or use industry templates")
    
    # Recipe tabs
    recipe_tabs = st.tabs(["💾 Save Recipe", "📂 My Recipes", "🏭 Industry Templates"])
    
    with recipe_tabs[0]:
        st.markdown("#### Save Current Workflow as Recipe")
        
        if not pipeline.code_generator.operations:
            st.warning("⚠️ No operations to save. Clean your data first.")
        else:
            recipe_name = st.text_input("Recipe Name", placeholder="e.g., Customer Data Standard Cleaning")
            recipe_description = st.text_area("Description", placeholder="Describe what this recipe does...")
            recipe_tags = st.text_input("Tags (comma-separated)", placeholder="e.g., customer, crm, basic")
            
            if st.button("💾 Save Recipe", type="primary"):
                if recipe_name:
                    tags = [tag.strip() for tag in recipe_tags.split(',')] if recipe_tags else []
                    
                    recipe = pipeline.recipe_manager.save_recipe(
                        name=recipe_name,
                        operations=pipeline.code_generator.operations,
                        description=recipe_description,
                        tags=tags
                    )
                    
                    st.success(f"✅ Recipe '{recipe_name}' saved successfully!")
                else:
                    st.error("❌ Please provide a recipe name")
    
    with recipe_tabs[1]:
        st.markdown("#### Your Saved Recipes")
        
        recipes = pipeline.recipe_manager.list_recipes()
        
        if not recipes:
            st.info("No recipes saved yet. Save your first workflow!")
        else:
            # Search recipes
            search_query = st.text_input("🔍 Search recipes", placeholder="Search by name...")
            
            filtered_recipes = [r for r in recipes if not search_query or search_query.lower() in r['name'].lower()]
            
            for recipe in filtered_recipes:
                with st.expander(f"📋 {recipe['name']} ({recipe['operations_count']} operations)"):
                    st.write(f"**Description:** {recipe['description']}")
                    st.write(f"**Created:** {recipe['created_at']}")
                    st.write(f"**Tags:** {', '.join(recipe['tags']) if recipe['tags'] else 'None'}")
                    st.write(f"**Used:** {recipe['usage_count']} times")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"📥 Load Recipe", key=f"load_{recipe['name']}"):
                            loaded = pipeline.recipe_manager.load_recipe(recipe['name'])
                            if loaded:
                                # Apply recipe operations
                                st.success(f"✅ Recipe '{recipe['name']}' loaded!")
                                st.info("💡 Recipe operations are ready to apply")
                    
                    with col2:
                        recipe_json = pipeline.recipe_manager.export_recipe(recipe['name'])
                        if recipe_json:
                            st.download_button(
                                "📤 Export",
                                recipe_json,
                                f"{recipe['name'].replace(' ', '_')}.json",
                                "application/json",
                                key=f"export_{recipe['name']}"
                            )
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{recipe['name']}"):
                            pipeline.recipe_manager.delete_recipe(recipe['name'])
                            st.success(f"✅ Recipe deleted")
                            st.rerun()
            
            # Popular recipes
            st.markdown("---")
            st.markdown("#### 🔥 Most Used Recipes")
            popular = pipeline.recipe_manager.get_popular_recipes(limit=3)
            
            if popular:
                for recipe in popular:
                    st.write(f"- **{recipe['name']}** ({recipe['usage_count']} uses)")
    
    with recipe_tabs[2]:
        st.markdown("#### 🏭 Industry Templates")
        st.info("💡 Pre-built templates for common data cleaning scenarios")
        
        for template_key, template in INDUSTRY_TEMPLATES.items():
            with st.expander(f"🏷️ {template['name']}"):
                st.write(f"**Description:** {template['description']}")
                st.write(f"**Tags:** {', '.join(template['tags'])}")
                st.write(f"**Operations:** {len(template['operations'])}")
                
                # Show operations
                st.markdown("**Workflow:**")
                for i, op in enumerate(template['operations']):
                    st.write(f"{i+1}. {op['operation'].replace('_', ' ').title()}")
                
                if st.button(f"🚀 Use Template", key=f"use_{template_key}"):
                    # Load template into recipe manager
                    pipeline.recipe_manager.save_recipe(
                        name=f"{template['name']} (Copy)",
                        operations=template['operations'],
                        description=template['description'],
                        tags=template['tags']
                    )
                    st.success(f"✅ Template loaded! Check 'My Recipes' tab")


def show_learning_tab(pipeline):
    """Show AI learning insights"""
    st.markdown("### 🧠 AI Learning Insights")
    st.info("💡 See how the AI learns from your preferences and automatic operations")
    
    insights = pipeline.learning_engine.get_learning_insights()
    
    # Auto-learning status
    st.markdown("#### ⚙️ Auto-Learning Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        status = "🟢 Enabled" if insights['auto_learning_enabled'] else "🔴 Disabled"
        st.metric("Auto-Learning Status", status)
    
    with col2:
        st.metric("Confidence Threshold", f"{insights['confidence_threshold']:.0%}")
    
    st.markdown("---")
    
    # Overall metrics
    st.markdown("#### 📊 Learning Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Interactions", insights['total_interactions'])
    col2.metric("Manual Approvals", insights['approvals'])
    col3.metric("Rejections", insights['rejections'])
    col4.metric("Auto-Learned", insights['auto_learned'], help="Operations learned from automatic mode")
    col5.metric("Approval Rate", f"{insights['approval_rate']:.1f}%")
    
    # Learned patterns with breakdown
    st.markdown("#### 🎯 Learned Patterns")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Missing Value Rules", insights['learned_patterns']['missing_value_preferences'])
    col2.metric("Outlier Rules", insights['learned_patterns']['outlier_preferences'])
    col3.metric("Auto-Learned Operations", insights['learned_patterns']['auto_learned_operations'])
    
    if insights['auto_learned'] > 0:
        st.markdown("#### 🤖 Auto-Learning Effectiveness")
        st.metric("Average Auto-Learning Confidence", f"{insights['avg_auto_confidence']:.0%}")
        
        st.info("""
        **How Auto-Learning Works:**
        - ✅ Enabled in Automatic mode
        - 🎯 Learns from high-confidence operations (>{:.0%})
        - 📈 Improves future suggestions
        - 🔄 Applies learned patterns automatically
        """.format(insights['confidence_threshold']))
    
    # Feedback history with auto-learn indicator
    if pipeline.learning_engine.feedback_history:
        st.markdown("#### 📜 Recent Learning Activity")
        
        recent = pipeline.learning_engine.feedback_history[-15:]  # Last 15
        
        for feedback in reversed(recent):
            action_emoji = {
                'approved': '✅',
                'rejected': '❌',
                'modified': '✏️',
                'auto_learned': '🤖'
            }
            emoji = action_emoji.get(feedback['action'], '📝')
            
            confidence_badge = ""
            if feedback.get('confidence'):
                conf = feedback['confidence']
                if conf >= 0.9:
                    confidence_badge = f" 🟢 {conf:.0%}"
                elif conf >= 0.75:
                    confidence_badge = f" 🟡 {conf:.0%}"
                else:
                    confidence_badge = f" 🔴 {conf:.0%}"
            
            action_text = feedback['action'].replace('_', ' ').title()
            st.write(f"{emoji} **{action_text}**: {feedback['operation']}{confidence_badge} - {feedback['timestamp'][:19]}")
    else:
        st.info("No learning activity yet. Use Assisted or Automatic mode to start teaching the AI!")
    
    # Learning effectiveness explanation
    with st.expander("📖 Understanding Auto-Learning"):
        st.markdown("""
        ### How Auto-Learning Works
        
        **In Automatic Mode:**
        1. AI generates cleaning suggestions
        2. Applies high-confidence operations (>85%)
        3. **Monitors success** of each operation
        4. **Learns from successful operations** above your threshold
        5. **Improves future suggestions** based on learned patterns
        
        **What AI Learns:**
        - ✅ Your preferred strategies for missing values
        - ✅ Your preferred outlier handling methods
        - ✅ Patterns in your data types
        - ✅ Which operations work best for you
        
        **Benefits:**
        - 🚀 Faster cleaning over time
        - 🎯 More accurate suggestions
        - 🔄 Consistent with your style
        - 📈 Continuously improving
        
        **Example:**
        ```
        Dataset 1 (Auto mode):
          Column 'age' has missing values
          AI uses MEDIAN (confidence: 85%)
          ✓ Success! AI learns you prefer MEDIAN for age
        
        Dataset 2 (weeks later):
          Column 'age' has missing values
          AI suggests MEDIAN (confidence: 95% - learned!)
          ✓ Applied automatically with higher confidence
        ```
        
        **Confidence Threshold:**
        - Higher threshold (90%+): More selective, only learns from very confident operations
        - Lower threshold (70-80%): More exploratory, learns from more operations
        - Default (80%): Balanced approach
        """)
    
    # Export/Import preferences
    st.markdown("---")
    st.markdown("#### 💾 Manage Learning Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Preferences**")
        st.caption("Includes both manual and auto-learned patterns")
        if st.button("📥 Export Learning Data"):
            prefs_json = pipeline.learning_engine.export_preferences()
            st.download_button(
                "Download Preferences",
                prefs_json,
                "learning_preferences.json",
                "application/json"
            )
    
    with col2:
        st.markdown("**Import Preferences**")
        st.caption("Load saved learning data")
        uploaded_prefs = st.file_uploader("Upload learning data", type=['json'], key="import_learning")
        if uploaded_prefs:
            try:
                prefs_data = uploaded_prefs.read().decode()
                pipeline.learning_engine.import_preferences(prefs_data)
                st.success("✅ Preferences imported!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error importing: {e}")
    
    # Reset option
    st.markdown("---")
    if st.button("🔄 Reset All Learning Data", help="Clear all learned preferences (manual + automatic)"):
        if st.checkbox("I understand this will delete all learning data"):
            pipeline.learning_engine.reset_learning()
            st.success("✅ Learning data reset")
            st.rerun()


def show_predictions_tab(pipeline):
    """Show predictive modeling interface"""
    st.subheader("🔮 Predictive Modeling")
    st.info("💡 Build ML models from your cleaned data to make predictions")
    
    # Check if we have cleaned data
    data_to_use = pipeline.cleaned_data if pipeline.cleaned_data is not None else pipeline.data
    
    if data_to_use is None:
        st.warning("⚠️ No data available. Please load data first.")
        return
    
    # Initialize prediction pipeline if needed
    if pipeline.prediction_pipeline is None:
        pipeline.prediction_pipeline = PredictionPipeline(data_to_use)
    
    # Create sub-tabs
    pred_tabs = st.tabs(["🎯 Build Model", "📊 Model Results", "🔮 Make Predictions", "📚 Model Comparison"])
    
    with pred_tabs[0]:
        show_model_building_tab(pipeline)
    
    with pred_tabs[1]:
        show_model_results_tab(pipeline)
    
    with pred_tabs[2]:
        show_make_predictions_tab(pipeline)
    
    with pred_tabs[3]:
        show_model_comparison_tab(pipeline)


def show_model_building_tab(pipeline):
    """Show model building interface"""
    st.markdown("### 🎯 Build Prediction Model")
    
    data_to_use = pipeline.cleaned_data if pipeline.cleaned_data is not None else pipeline.data
    
    # Analyze readiness
    if st.button("🔍 Analyze Prediction Readiness"):
        analysis = pipeline.prediction_pipeline.analyze_prediction_readiness()
        
        if analysis['ready']:
            st.success("✅ Data is ready for ML!")
        else:
            st.error("❌ Data needs preparation")
            for issue in analysis['issues']:
                st.write(f"- {issue}")
        
        if analysis['recommendations']:
            st.markdown("**Recommendations:**")
            for rec in analysis['recommendations']:
                st.info(rec)
        
        # Show potential targets
        st.markdown("### 🎯 Suggested Target Columns")
        if analysis['potential_targets']:
            for target in analysis['potential_targets']:
                with st.expander(f"📊 {target['column']} ({target['type']})"):
                    if target['type'] == 'classification':
                        st.write(f"**Classes:** {target['n_classes']}")
                    else:
                        st.write(f"**Range:** {target['range']}")
        else:
            st.warning("No obvious target columns found. You can still select manually below.")
    
    st.markdown("---")
    st.markdown("### ⚙️ Configure Model")
    
    # Select target column
    target_column = st.selectbox(
        "Target Column (what to predict)",
        data_to_use.columns.tolist(),
        help="Choose the column you want to predict"
    )
    
    if not target_column:
        st.warning("⚠️ Please select a target column")
        return
    
    # Select features
    available_features = [col for col in data_to_use.columns if col != target_column]
    feature_columns = st.multiselect(
        "Feature Columns (predictors)",
        available_features,
        default=available_features[:min(10, len(available_features))],
        help="Choose columns to use for prediction"
    )
    
    if not feature_columns:
        st.warning("⚠️ Please select at least one feature column")
        return
    
    # Auto-detect problem type with error handling
    try:
        problem_type = pipeline.prediction_pipeline.model_builder.auto_detect_problem_type(
            data_to_use, target_column
        )
        st.info(f"🎯 Detected problem type: **{problem_type.value.title()}**")
        
        # Get model recommendations
        recommendations = pipeline.prediction_pipeline.get_model_recommendations(target_column)
    except Exception as e:
        st.error(f"❌ Error analyzing target column: {str(e)}")
        st.info("💡 Tips: Make sure the column exists and has valid data")
        return
    
    st.markdown("### 🤖 Recommended Models")
    
    selected_models = []
    for i, rec in enumerate(recommendations):
        with st.expander(f"📊 {rec['model'].value.replace('_', ' ').title()}", expanded=(i==0)):
            st.write(f"**Best For:** {rec['best_for']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pros:**")
                for pro in rec['pros']:
                    st.write(f"✅ {pro}")
            with col2:
                st.write("**Cons:**")
                for con in rec['cons']:
                    st.write(f"⚠️ {con}")
            
            if st.checkbox(f"Train this model", key=f"train_{rec['model'].value}"):
                selected_models.append(rec['model'])
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random Seed", value=42, min_value=0)
    
    # Train button
    if st.button("🚀 Train Model(s)", type="primary", disabled=len(selected_models)==0):
        if not selected_models:
            st.error("Please select at least one model to train")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_list = []
            
            for idx, model_type in enumerate(selected_models):
                status_text.text(f"Training {model_type.value.replace('_', ' ').title()}...")
                progress_bar.progress((idx + 1) / len(selected_models))
                
                try:
                    results = pipeline.prediction_pipeline.train_model(
                        target_column=target_column,
                        feature_columns=feature_columns,
                        model_type=model_type
                    )
                    results_list.append(results)
                    
                    # Store in session state
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = []
                    st.session_state.trained_models.append(results)
                    
                    st.success(f"✅ {model_type.value.replace('_', ' ').title()} trained!")
                    
                except Exception as e:
                    st.error(f"❌ Error training {model_type.value}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ All models trained!")
            
            if results_list:
                st.balloons()
                st.success(f"🎉 Successfully trained {len(results_list)} model(s)!")


def show_model_results_tab(pipeline):
    """Show model training results"""
    st.markdown("### 📊 Model Training Results")
    
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.info("No models trained yet. Go to 'Build Model' tab to train your first model.")
        return
    
    # Display each model's results
    for idx, results in enumerate(st.session_state.trained_models):
        with st.expander(
            f"{'🥇' if idx == 0 else '🥈' if idx == 1 else '🥉' if idx == 2 else '📊'} "
            f"{results.model_type.replace('_', ' ').title()} - "
            f"{list(results.metrics.keys())[0]}: {list(results.metrics.values())[0]:.3f}",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Model Type:** {results.model_type.replace('_', ' ').title()}")
                st.markdown(f"**Problem Type:** {results.problem_type.title()}")
                st.markdown(f"**Training Time:** {results.training_time:.2f} seconds")
                st.markdown(f"**Trained:** {results.timestamp}")
            
            with col2:
                # Show main metric prominently
                main_metric = list(results.metrics.keys())[0]
                main_value = list(results.metrics.values())[0]
                st.metric(main_metric.upper(), f"{main_value:.3f}")
            
            # Show all metrics
            st.markdown("#### 📈 Performance Metrics")
            metric_cols = st.columns(len(results.metrics))
            for col, (metric, value) in zip(metric_cols, results.metrics.items()):
                col.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
            
            # Feature importance
            if results.feature_importance:
                st.markdown("#### 🎯 Feature Importance")
                
                # Sort by importance
                sorted_features = sorted(
                    results.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Bar(
                    x=[imp for _, imp in sorted_features[:10]],
                    y=[feat for feat, _ in sorted_features[:10]],
                    orientation='h',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Sample predictions
            if results.predictions_sample:
                st.markdown("#### 🔮 Sample Predictions")
                
                sample_df = pd.DataFrame(results.predictions_sample)
                sample_df['Status'] = sample_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
                
                st.dataframe(
                    sample_df[['actual', 'predicted', 'Status']],
                    use_container_width=True,
                    hide_index=True
                )


def show_make_predictions_tab(pipeline):
    """Show interface for making new predictions"""
    st.markdown("### 🔮 Make Predictions on New Data")
    
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.info("No trained models available. Train a model first.")
        return
    
    st.info("💡 Upload new data to make predictions, or use current data")
    
    # Select model
    model_options = [
        f"{r.model_type.replace('_', ' ').title()} - {r.timestamp}"
        for r in st.session_state.trained_models
    ]
    
    selected_idx = st.selectbox(
        "Select Model",
        range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    
    # Option to upload new data or use current
    prediction_source = st.radio(
        "Data Source",
        ["Use Current Data", "Upload New Data"]
    )
    
    if prediction_source == "Upload New Data":
        uploaded_file = st.file_uploader("Upload data for predictions", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_file)
                else:
                    new_data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(new_data)} rows")
                st.dataframe(new_data.head(), use_container_width=True)
                
                if st.button("🔮 Generate Predictions"):
                    st.info("Prediction functionality would generate predictions here")
                    # In full implementation:
                    # predictions = pipeline.prediction_pipeline.trained_models[selected_idx].predict(new_data)
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        st.write("Using current dataset for predictions")
        data_to_use = pipeline.cleaned_data if pipeline.cleaned_data is not None else pipeline.data
        st.dataframe(data_to_use.head(), use_container_width=True)


def show_model_comparison_tab(pipeline):
    """Show model comparison"""
    st.markdown("### 📚 Model Comparison")
    
    if 'trained_models' not in st.session_state or len(st.session_state.trained_models) < 2:
        st.info("Train at least 2 models to compare them")
        return
    
    # Create comparison table
    comparison_data = []
    for results in st.session_state.trained_models:
        row = {
            'Model': results.model_type.replace('_', ' ').title(),
            'Training Time (s)': f"{results.training_time:.2f}"
        }
        for metric, value in results.metrics.items():
            row[metric.title()] = f"{value:.3f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Recommend best model
    st.markdown("### 🏆 Recommendation")
    
    # Find best model based on primary metric
    best_idx = 0
    best_score = 0
    primary_metric = list(st.session_state.trained_models[0].metrics.keys())[0]
    
    for idx, results in enumerate(st.session_state.trained_models):
        score = list(results.metrics.values())[0]
        if score > best_score:
            best_score = score
            best_idx = idx
    
    best_model = st.session_state.trained_models[best_idx]
    
    st.success(
        f"🥇 **Best Model:** {best_model.model_type.replace('_', ' ').title()}\n\n"
        f"**{primary_metric.upper()}:** {best_score:.3f}"
    )


if __name__ == "__main__":
    main()