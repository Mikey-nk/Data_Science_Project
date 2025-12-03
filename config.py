import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
import streamlit as st

class ConfigManager:
    """Manages API keys and configuration settings securely"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration structure"""
        return {
            "apis": {
                "firebase": {
                    "api_key": "",
                    "project_id": "",
                    "database_url": "",
                    "storage_bucket": ""
                },
                "google_cloud": {
                    "api_key": "",
                    "project_id": ""
                },
                "openai": {
                    "api_key": "",
                    "model": "gpt-4"
                },
                "aws": {
                    "access_key_id": "",
                    "secret_access_key": "",
                    "region": "us-east-1"
                }
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "datacleaner_db",
                "user": "",
                "password": ""
            },
            "app_settings": {
                "max_file_size_mb": 200,
                "allowed_file_types": ["csv", "xlsx", "json", "parquet"],
                "default_export_format": "csv",
                "enable_logging": True,
                "log_level": "INFO"
            },
            "email": {
                "smtp_server": "",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": ""
            },
            "data_processing": {
                "chunk_size": 10000,
                "max_workers": 4,
                "cache_enabled": True,
                "cache_dir": "./cache"
            }
        }
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        return self.config.get("apis", {}).get(service, {}).get("api_key")
    
    def set_api_key(self, service: str, api_key: str) -> bool:
        """Set API key for a specific service"""
        if "apis" not in self.config:
            self.config["apis"] = {}
        if service not in self.config["apis"]:
            self.config["apis"][service] = {}
        
        self.config["apis"][service]["api_key"] = api_key
        return self.save_config()
    
    def get_config_value(self, *keys) -> Optional[Any]:
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    
    def set_config_value(self, value: Any, *keys) -> bool:
        """Set nested configuration value"""
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        return self.save_config()
    
    def get_firebase_config(self) -> Dict[str, str]:
        """Get Firebase configuration"""
        return self.config.get("apis", {}).get("firebase", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.get("database", {})
    
    def get_app_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        return self.config.get("app_settings", {})
    
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return missing/invalid items"""
        issues = {
            "missing": [],
            "invalid": []
        }
        
        # Check required API keys
        required_apis = ["firebase", "google_cloud"]
        for api in required_apis:
            if not self.get_api_key(api):
                issues["missing"].append(f"API key for {api}")
        
        # Validate email settings if provided
        email_config = self.config.get("email", {})
        if email_config.get("smtp_server") and not email_config.get("sender_email"):
            issues["invalid"].append("SMTP server configured but sender email missing")
        
        return issues
    
    def export_config(self, filepath: str, exclude_sensitive: bool = True) -> bool:
        """Export configuration to a file"""
        try:
            export_data = self.config.copy()
            
            if exclude_sensitive:
                # Remove sensitive information
                if "apis" in export_data:
                    for service in export_data["apis"]:
                        if "api_key" in export_data["apis"][service]:
                            export_data["apis"][service]["api_key"] = "***HIDDEN***"
                        if "secret_access_key" in export_data["apis"][service]:
                            export_data["apis"][service]["secret_access_key"] = "***HIDDEN***"
                
                if "database" in export_data and "password" in export_data["database"]:
                    export_data["database"]["password"] = "***HIDDEN***"
                
                if "email" in export_data and "sender_password" in export_data["email"]:
                    export_data["email"]["sender_password"] = "***HIDDEN***"
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error exporting config: {e}")
            return False
    
    def import_config(self, filepath: str, merge: bool = True) -> bool:
        """Import configuration from a file"""
        try:
            with open(filepath, 'r') as f:
                imported_config = json.load(f)
            
            if merge:
                # Merge with existing config
                self._merge_configs(self.config, imported_config)
            else:
                # Replace entire config
                self.config = imported_config
            
            return self.save_config()
        except Exception as e:
            print(f"Error importing config: {e}")
            return False
    
    def _merge_configs(self, base: dict, update: dict):
        """Recursively merge two configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value


# Streamlit UI for Configuration Management
def config_ui():
    """Streamlit interface for managing configuration"""
    st.subheader("⚙️ Configuration Manager")
    
    # Initialize config manager in session state
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    config_mgr = st.session_state.config_manager
    
    tabs = st.tabs(["🔑 API Keys", "🗄️ Database", "📧 Email", "⚙️ App Settings", "📤 Import/Export"])
    
    # API Keys Tab
    with tabs[0]:
        st.markdown("#### Firebase Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            firebase_api_key = st.text_input(
                "Firebase API Key",
                value=config_mgr.get_config_value("apis", "firebase", "api_key") or "",
                type="password"
            )
            firebase_project_id = st.text_input(
                "Project ID",
                value=config_mgr.get_config_value("apis", "firebase", "project_id") or ""
            )
        
        with col2:
            firebase_db_url = st.text_input(
                "Database URL",
                value=config_mgr.get_config_value("apis", "firebase", "database_url") or ""
            )
            firebase_storage = st.text_input(
                "Storage Bucket",
                value=config_mgr.get_config_value("apis", "firebase", "storage_bucket") or ""
            )
        
        if st.button("Save Firebase Config"):
            config_mgr.set_config_value(firebase_api_key, "apis", "firebase", "api_key")
            config_mgr.set_config_value(firebase_project_id, "apis", "firebase", "project_id")
            config_mgr.set_config_value(firebase_db_url, "apis", "firebase", "database_url")
            config_mgr.set_config_value(firebase_storage, "apis", "firebase", "storage_bucket")
            st.success("✅ Firebase configuration saved!")
        
        st.markdown("---")
        st.markdown("#### Google Cloud Configuration")
        
        google_api_key = st.text_input(
            "Google Cloud API Key",
            value=config_mgr.get_config_value("apis", "google_cloud", "api_key") or "",
            type="password"
        )
        
        google_project_id = st.text_input(
            "Google Cloud Project ID",
            value=config_mgr.get_config_value("apis", "google_cloud", "project_id") or ""
        )
        
        if st.button("Save Google Cloud Config"):
            config_mgr.set_config_value(google_api_key, "apis", "google_cloud", "api_key")
            config_mgr.set_config_value(google_project_id, "apis", "google_cloud", "project_id")
            st.success("✅ Google Cloud configuration saved!")
        
        st.markdown("---")
        st.markdown("#### Other API Keys")
        
        openai_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=config_mgr.get_config_value("apis", "openai", "api_key") or "",
            type="password"
        )
        
        if st.button("Save OpenAI Config"):
            config_mgr.set_config_value(openai_key, "apis", "openai", "api_key")
            st.success("✅ OpenAI configuration saved!")
    
    # Database Tab
    with tabs[1]:
        st.markdown("#### Database Connection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input("Host", value=config_mgr.get_config_value("database", "host") or "localhost")
            db_port = st.number_input("Port", value=config_mgr.get_config_value("database", "port") or 5432)
            db_name = st.text_input("Database Name", value=config_mgr.get_config_value("database", "name") or "")
        
        with col2:
            db_user = st.text_input("Username", value=config_mgr.get_config_value("database", "user") or "")
            db_password = st.text_input("Password", value=config_mgr.get_config_value("database", "password") or "", type="password")
        
        if st.button("Save Database Config"):
            config_mgr.set_config_value(db_host, "database", "host")
            config_mgr.set_config_value(db_port, "database", "port")
            config_mgr.set_config_value(db_name, "database", "name")
            config_mgr.set_config_value(db_user, "database", "user")
            config_mgr.set_config_value(db_password, "database", "password")
            st.success("✅ Database configuration saved!")
    
    # Email Tab
    with tabs[2]:
        st.markdown("#### Email Configuration (For Notifications)")
        
        smtp_server = st.text_input("SMTP Server", value=config_mgr.get_config_value("email", "smtp_server") or "")
        smtp_port = st.number_input("SMTP Port", value=config_mgr.get_config_value("email", "smtp_port") or 587)
        sender_email = st.text_input("Sender Email", value=config_mgr.get_config_value("email", "sender_email") or "")
        sender_password = st.text_input("Email Password", value=config_mgr.get_config_value("email", "sender_password") or "", type="password")
        
        if st.button("Save Email Config"):
            config_mgr.set_config_value(smtp_server, "email", "smtp_server")
            config_mgr.set_config_value(smtp_port, "email", "smtp_port")
            config_mgr.set_config_value(sender_email, "email", "sender_email")
            config_mgr.set_config_value(sender_password, "email", "sender_password")
            st.success("✅ Email configuration saved!")
    
    # App Settings Tab
    with tabs[3]:
        st.markdown("#### Application Settings")
        
        max_file_size = st.number_input(
            "Max File Size (MB)",
            value=config_mgr.get_config_value("app_settings", "max_file_size_mb") or 200,
            min_value=1,
            max_value=1000
        )
        
        default_export = st.selectbox(
            "Default Export Format",
            ["csv", "excel", "json", "parquet"],
            index=["csv", "excel", "json", "parquet"].index(
                config_mgr.get_config_value("app_settings", "default_export_format") or "csv"
            )
        )
        
        enable_logging = st.checkbox(
            "Enable Logging",
            value=config_mgr.get_config_value("app_settings", "enable_logging") or True
        )
        
        chunk_size = st.number_input(
            "Data Processing Chunk Size",
            value=config_mgr.get_config_value("data_processing", "chunk_size") or 10000,
            min_value=1000,
            max_value=100000
        )
        
        if st.button("Save App Settings"):
            config_mgr.set_config_value(max_file_size, "app_settings", "max_file_size_mb")
            config_mgr.set_config_value(default_export, "app_settings", "default_export_format")
            config_mgr.set_config_value(enable_logging, "app_settings", "enable_logging")
            config_mgr.set_config_value(chunk_size, "data_processing", "chunk_size")
            st.success("✅ App settings saved!")
    
    # Import/Export Tab
    with tabs[4]:
        st.markdown("#### Export Configuration")
        
        exclude_sensitive = st.checkbox("Exclude sensitive data (API keys, passwords)", value=True)
        
        if st.button("Export Configuration"):
            export_path = "config_export.json"
            if config_mgr.export_config(export_path, exclude_sensitive):
                with open(export_path, 'r') as f:
                    config_data = f.read()
                
                st.download_button(
                    "Download Configuration",
                    config_data,
                    "config_export.json",
                    "application/json"
                )
                st.success("✅ Configuration exported!")
        
        st.markdown("---")
        st.markdown("#### Import Configuration")
        
        uploaded_config = st.file_uploader("Upload Configuration File", type=['json'])
        merge_config = st.checkbox("Merge with existing configuration", value=True)
        
        if uploaded_config and st.button("Import Configuration"):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_config.getvalue())
                tmp_path = tmp_file.name
            
            if config_mgr.import_config(tmp_path, merge_config):
                st.success("✅ Configuration imported successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to import configuration")
            
            os.unlink(tmp_path)
        
        st.markdown("---")
        st.markdown("#### Validation")
        
        if st.button("Validate Configuration"):
            issues = config_mgr.validate_config()
            
            if issues["missing"]:
                st.warning("⚠️ Missing configuration items:")
                for item in issues["missing"]:
                    st.write(f"- {item}")
            
            if issues["invalid"]:
                st.error("❌ Invalid configuration items:")
                for item in issues["invalid"]:
                    st.write(f"- {item}")
            
            if not issues["missing"] and not issues["invalid"]:
                st.success("✅ Configuration is valid!")


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Configuration Manager", layout="wide")
    config_ui()