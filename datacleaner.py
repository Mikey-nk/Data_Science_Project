import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

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
    
    def apply_cleaning_rules(self, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply a dictionary of cleaning rules"""
        
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
    
    def __init__(self):
        self.data = None
        self.profiler = None
        self.cleaner = None
        self.profile_result = None
        self.cleaned_data = None
    
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
        
        self.profiler = DataProfiler(self.data)
        self.profile_result = self.profiler.generate_profile()
        return self.profile_result
    
    def clean_data(self, rules: Dict[str, Any]) -> pd.DataFrame:
        """Clean the data using specified rules"""
        if self.data is None:
            raise ValueError("No data ingested. Please ingest data first.")
        
        self.cleaner = DataCleaner(self.data)
        self.cleaned_data = self.cleaner.apply_cleaning_rules(rules)
        return self.cleaned_data
    
    def export_data(self, format: str = 'csv') -> bytes:
        """Export cleaned data in various formats"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        
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


# Streamlit UI
def main():
    st.set_page_config(page_title="Data Cleaning & Profiling System", layout="wide")
    
    st.title("🧹 Data Cleaning & Profiling System")
    st.markdown("---")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataPipeline()
    
    pipeline = st.session_state.pipeline
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_source = st.radio("Data Source", ["Upload File", "Firebase Data"])
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'json', 'parquet']
            )
            
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1]
                if file_type == 'xlsx':
                    file_type = 'excel'
                
                if st.button("Load Data"):
                    try:
                        with st.spinner("Loading data..."):
                            pipeline.ingest_data(uploaded_file, file_type)
                        st.success(f"✅ Loaded {len(pipeline.data)} rows")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        else:
            st.info("📝 Paste Firebase data as JSON array")
            firebase_json = st.text_area("Firebase Data (JSON)", height=200)
            
            if st.button("Load Firebase Data"):
                try:
                    firebase_data = json.loads(firebase_json)
                    pipeline.ingest_from_firebase(firebase_data)
                    st.success(f"✅ Loaded {len(pipeline.data)} rows")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Main content area
    if pipeline.data is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "🔍 Profile", "🧹 Clean", "📤 Export"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(pipeline.data.head(100), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(pipeline.data))
            col2.metric("Columns", len(pipeline.data.columns))
            col3.metric("Memory", f"{pipeline.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with tab2:
            st.subheader("Data Profiling")
            
            if st.button("Generate Profile"):
                with st.spinner("Profiling data..."):
                    profile = pipeline.profile_data()
                
                st.success("✅ Profile generated")
                
                # Display profile results
                st.markdown("### Missing Data")
                if profile['missing_data']['by_column']:
                    missing_df = pd.DataFrame(profile['missing_data']['by_column']).T
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.info("No missing data found")
                
                st.markdown("### Duplicates")
                st.write(f"Duplicate rows: {profile['duplicates']['duplicate_rows']} ({profile['duplicates']['duplicate_percentage']}%)")
                
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
        
        with tab3:
            st.subheader("Data Cleaning Rules")
            
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
                        key=f"missing_{col}"
                    )
                    if method != 'skip':
                        if method == 'constant':
                            const_val = st.text_input(f"Constant value for {col}", key=f"const_{col}")
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
                    key=f"type_{col}"
                )
                if new_type != 'no change':
                    type_conversions[col] = new_type
            
            if type_conversions:
                cleaning_rules['type_conversions'] = type_conversions
            
            # Text normalization
            text_cols = st.multiselect(
                "Normalize text in columns",
                pipeline.data.select_dtypes(include=['object']).columns.tolist()
            )
            if text_cols:
                cleaning_rules['normalize_text'] = text_cols
            
            # Apply cleaning
            if st.button("🧹 Apply Cleaning Rules", type="primary"):
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
        
        with tab4:
            st.subheader("Export Data")
            
            if pipeline.cleaned_data is not None:
                export_format = st.selectbox(
                    "Export Format",
                    ['csv', 'excel', 'json', 'parquet', 'firebase']
                )
                
                if st.button("Generate Export"):
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
                st.markdown("### Export Cleaning Log")
                log = pipeline.get_cleaning_log()
                if log:
                    log_json = json.dumps(log, indent=2)
                    st.download_button(
                        "Download Cleaning Log",
                        log_json,
                        "cleaning_log.json",
                        "application/json"
                    )
            else:
                st.warning("⚠️ Please clean the data first before exporting")
    
    else:
        st.info("👆 Please upload a file or provide Firebase data to get started")


if __name__ == "__main__":
    main()