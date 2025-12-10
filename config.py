import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
import streamlit as st
from firebase_config_integration import (
    parse_firebase_js_config,
    FirebaseConfigAdapter,
    init_storage_from_config
)

class ConfigManager:
    """Manages API keys and configuration settings securely with AI/ML model support"""
    
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
                # AI/ML Model APIs
                "openai": {
                    "api_key": "",
                    "model": "gpt-4",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "enabled": False
                },
                "anthropic": {
                    "api_key": "",
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "enabled": False
                },
                "huggingface": {
                    "api_key": "",
                    "model": "meta-llama/Llama-2-7b-chat-hf",
                    "enabled": False
                },
                "cohere": {
                    "api_key": "",
                    "model": "command",
                    "enabled": False
                },
                "azure_openai": {
                    "api_key": "",
                    "endpoint": "",
                    "deployment_name": "",
                    "api_version": "2023-05-15",
                    "enabled": False
                },
                "aws": {
                    "access_key_id": "",
                    "secret_access_key": "",
                    "region": "us-east-1",
                    "bedrock_model": "anthropic.claude-v2"
                },
                "google_ai": {
                    "api_key": "",
                    "model": "gemini-pro",
                    "enabled": False
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
            },
            "ai_features": {
                "enabled": True,
                "preferred_provider": "openai",  # openai, anthropic, huggingface, etc.
                "fallback_providers": ["anthropic", "google_ai"],
                "use_local_models": False,
                "local_model_path": "./models",
                "enable_embeddings": True,
                "embedding_model": "text-embedding-ada-002",
                "enable_explanations": True,
                "explanation_detail_level": "medium"  # low, medium, high
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
    
    def get_ai_config(self, provider: str = None) -> Dict[str, Any]:
        """Get AI provider configuration (for ML model integration)"""
        if provider is None:
            provider = self.config.get("ai_features", {}).get("preferred_provider", "openai")
        
        return self.config.get("apis", {}).get(provider, {})
    
    def get_active_ai_provider(self) -> Optional[str]:
        """Get the currently active AI provider"""
        ai_features = self.config.get("ai_features", {})
        
        if not ai_features.get("enabled", False):
            return None
        
        preferred = ai_features.get("preferred_provider", "openai")
        
        # Check if preferred provider is enabled and has API key
        provider_config = self.config.get("apis", {}).get(preferred, {})
        if provider_config.get("enabled", False) and provider_config.get("api_key"):
            return preferred
        
        # Try fallback providers
        for fallback in ai_features.get("fallback_providers", []):
            fallback_config = self.config.get("apis", {}).get(fallback, {})
            if fallback_config.get("enabled", False) and fallback_config.get("api_key"):
                return fallback
        
        return None
    
    def is_ai_enabled(self) -> bool:
        """Check if AI features are enabled and configured"""
        return (
            self.config.get("ai_features", {}).get("enabled", False) and
            self.get_active_ai_provider() is not None
        )
    
    def get_ai_model_params(self, provider: str = None) -> Dict[str, Any]:
        """Get AI model parameters for the specified provider"""
        if provider is None:
            provider = self.get_active_ai_provider()
        
        if provider is None:
            return {}
        
        provider_config = self.config.get("apis", {}).get(provider, {})
        
        # Extract relevant parameters
        params = {
            "api_key": provider_config.get("api_key", ""),
            "model": provider_config.get("model", ""),
        }
        
        # Add provider-specific parameters
        if provider == "openai":
            params.update({
                "max_tokens": provider_config.get("max_tokens", 2000),
                "temperature": provider_config.get("temperature", 0.7)
            })
        elif provider == "anthropic":
            params.update({
                "max_tokens": provider_config.get("max_tokens", 4096)
            })
        elif provider == "azure_openai":
            params.update({
                "endpoint": provider_config.get("endpoint", ""),
                "deployment_name": provider_config.get("deployment_name", ""),
                "api_version": provider_config.get("api_version", "")
            })
        elif provider == "aws":
            params.update({
                "access_key_id": provider_config.get("access_key_id", ""),
                "secret_access_key": provider_config.get("secret_access_key", ""),
                "region": provider_config.get("region", "us-east-1"),
                "bedrock_model": provider_config.get("bedrock_model", "")
            })
        
        return params
    
    def enable_ai_provider(self, provider: str, enabled: bool = True) -> bool:
        """Enable or disable an AI provider"""
        if provider not in self.config.get("apis", {}):
            return False
        
        self.config["apis"][provider]["enabled"] = enabled
        return self.save_config()
    
    def set_preferred_ai_provider(self, provider: str) -> bool:
        """Set the preferred AI provider"""
        if "ai_features" not in self.config:
            self.config["ai_features"] = {}
        
        self.config["ai_features"]["preferred_provider"] = provider
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
            "invalid": [],
            "warnings": []
        }
        
        # Check required API keys
        required_apis = ["firebase", "google_cloud"]
        for api in required_apis:
            if not self.get_api_key(api):
                issues["missing"].append(f"API key for {api}")
        
        # Check AI configuration
        if self.config.get("ai_features", {}).get("enabled", False):
            active_provider = self.get_active_ai_provider()
            if not active_provider:
                issues["warnings"].append(
                    "AI features enabled but no provider configured with valid API key"
                )
        
        # Validate email settings if provided
        email_config = self.config.get("email", {})
        if email_config.get("smtp_server") and not email_config.get("sender_email"):
            issues["invalid"].append("SMTP server configured but sender email missing")
        
        return issues
    
    def get_ai_status_summary(self) -> Dict[str, Any]:
        """Get summary of AI configuration status"""
        summary = {
            "ai_enabled": self.is_ai_enabled(),
            "active_provider": self.get_active_ai_provider(),
            "configured_providers": [],
            "enabled_providers": [],
            "use_local_models": self.config.get("ai_features", {}).get("use_local_models", False)
        }
        
        for provider, config in self.config.get("apis", {}).items():
            if provider in ["openai", "anthropic", "huggingface", "cohere", 
                           "azure_openai", "google_ai"]:
                if config.get("api_key"):
                    summary["configured_providers"].append(provider)
                if config.get("enabled", False):
                    summary["enabled_providers"].append(provider)
        
        return summary
    
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
    
    tabs = st.tabs([
        "🔑 API Keys", 
        "🤖 AI Models", 
        "🗄️ Database", 
        "📧 Email", 
        "⚙️ App Settings", 
        "📤 Import/Export"
    ])
    
    # API Keys Tab
    with tabs[0]:
        st.markdown("#### Firebase Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            firebase_api_key = st.text_input(
                "Firebase API Key",
                value=config_mgr.get_config_value("apis", "firebase", "api_key") or "",
                type="password",
                key="firebase_api_key"
            )
            firebase_project_id = st.text_input(
                "Project ID",
                value=config_mgr.get_config_value("apis", "firebase", "project_id") or "",
                key="firebase_project_id"
            )
        
        with col2:
            firebase_db_url = st.text_input(
                "Database URL",
                value=config_mgr.get_config_value("apis", "firebase", "database_url") or "",
                key="firebase_db_url"
            )
            firebase_storage = st.text_input(
                "Storage Bucket",
                value=config_mgr.get_config_value("apis", "firebase", "storage_bucket") or "",
                key="firebase_storage"
            )
        
        if st.button("Save Firebase Config", key="save_firebase"):
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
            type="password",
            key="google_api_key"
        )
        
        google_project_id = st.text_input(
            "Google Cloud Project ID",
            value=config_mgr.get_config_value("apis", "google_cloud", "project_id") or "",
            key="google_project_id"
        )
        
        if st.button("Save Google Cloud Config", key="save_google"):
            config_mgr.set_config_value(google_api_key, "apis", "google_cloud", "api_key")
            config_mgr.set_config_value(google_project_id, "apis", "google_cloud", "project_id")
            st.success("✅ Google Cloud configuration saved!")
    
    # AI Models Tab
    with tabs[1]:
        st.markdown("#### 🤖 AI/ML Model Configuration")
        
        # AI Features Toggle
        ai_enabled = st.checkbox(
            "Enable AI Features",
            value=config_mgr.get_config_value("ai_features", "enabled") or False,
            help="Enable AI-powered explanations and suggestions",
            key="ai_enabled"
        )
        config_mgr.set_config_value(ai_enabled, "ai_features", "enabled")
        
        if ai_enabled:
            # Show AI status
            status = config_mgr.get_ai_status_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Provider", status['active_provider'] or "None")
            with col2:
                st.metric("Configured", len(status['configured_providers']))
            with col3:
                st.metric("Enabled", len(status['enabled_providers']))
            
            st.markdown("---")
            
            # Provider selection
            providers = ["openai", "anthropic", "google_ai", "huggingface", "cohere", "azure_openai"]
            
            for provider in providers:
                with st.expander(f"📡 {provider.replace('_', ' ').title()}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        api_key = st.text_input(
                            "API Key",
                            value=config_mgr.get_config_value("apis", provider, "api_key") or "",
                            type="password",
                            key=f"{provider}_api_key"
                        )
                        
                        if provider == "openai":
                            model = st.selectbox(
                                "Model",
                                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                                index=0,
                                key=f"{provider}_model"
                            )
                            temperature = st.slider(
                                "Temperature",
                                0.0, 1.0, 0.7, 0.1,
                                key=f"{provider}_temp"
                            )
                            config_mgr.set_config_value(model, "apis", provider, "model")
                            config_mgr.set_config_value(temperature, "apis", provider, "temperature")
                        
                        elif provider == "anthropic":
                            model = st.selectbox(
                                "Model",
                                ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
                                index=0,
                                key=f"{provider}_model"
                            )
                            config_mgr.set_config_value(model, "apis", provider, "model")
                        
                        elif provider == "google_ai":
                            model = st.selectbox(
                                "Model",
                                ["gemini-pro", "gemini-pro-vision"],
                                index=0,
                                key=f"{provider}_model"
                            )
                            config_mgr.set_config_value(model, "apis", provider, "model")
                        
                        elif provider == "azure_openai":
                            endpoint = st.text_input(
                                "Azure Endpoint",
                                value=config_mgr.get_config_value("apis", provider, "endpoint") or "",
                                key=f"{provider}_endpoint"
                            )
                            deployment = st.text_input(
                                "Deployment Name",
                                value=config_mgr.get_config_value("apis", provider, "deployment_name") or "",
                                key=f"{provider}_deployment"
                            )
                            config_mgr.set_config_value(endpoint, "apis", provider, "endpoint")
                            config_mgr.set_config_value(deployment, "apis", provider, "deployment_name")
                    
                    with col2:
                        enabled = st.checkbox(
                            "Enable",
                            value=config_mgr.get_config_value("apis", provider, "enabled") or False,
                            key=f"{provider}_enabled"
                        )
                        
                        is_preferred = config_mgr.get_config_value("ai_features", "preferred_provider") == provider
                        
                        if st.button(
                            "✓ Preferred" if is_preferred else "Set Preferred",
                            key=f"{provider}_preferred",
                            disabled=is_preferred
                        ):
                            config_mgr.set_preferred_ai_provider(provider)
                            st.success(f"✅ {provider} set as preferred provider")
                            st.rerun()
                    
                    if st.button(f"Save {provider} Config", key=f"save_{provider}"):
                        config_mgr.set_config_value(api_key, "apis", provider, "api_key")
                        config_mgr.set_config_value(enabled, "apis", provider, "enabled")
                        st.success(f"✅ {provider} configuration saved!")
            
            st.markdown("---")
            st.markdown("#### Advanced AI Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_embeddings = st.checkbox(
                    "Enable Embeddings",
                    value=config_mgr.get_config_value("ai_features", "enable_embeddings") or True,
                    help="Use embeddings for semantic search and similarity",
                    key="enable_embeddings"
                )
                config_mgr.set_config_value(enable_embeddings, "ai_features", "enable_embeddings")
            
            with col2:
                detail_level = st.select_slider(
                    "Explanation Detail Level",
                    options=["low", "medium", "high"],
                    value=config_mgr.get_config_value("ai_features", "explanation_detail_level") or "medium",
                    key="detail_level"
                )
                config_mgr.set_config_value(detail_level, "ai_features", "explanation_detail_level")
            
            if st.button("Save AI Settings", key="save_ai_settings"):
                config_mgr.save_config()
                st.success("✅ AI settings saved!")
        
        else:
            st.info("ℹ️ Enable AI features to configure AI providers")
    
    # Database Tab
    with tabs[2]:
        st.markdown("#### Database Connection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input(
                "Host", 
                value=config_mgr.get_config_value("database", "host") or "localhost",
                key="db_host"
            )
            db_port = st.number_input(
                "Port", 
                value=config_mgr.get_config_value("database", "port") or 5432,
                key="db_port"
            )
            db_name = st.text_input(
                "Database Name", 
                value=config_mgr.get_config_value("database", "name") or "",
                key="db_name"
            )
        
        with col2:
            db_user = st.text_input(
                "Username", 
                value=config_mgr.get_config_value("database", "user") or "",
                key="db_user"
            )
            db_password = st.text_input(
                "Password", 
                value=config_mgr.get_config_value("database", "password") or "", 
                type="password",
                key="db_password"
            )
        
        if st.button("Save Database Config", key="save_db"):
            config_mgr.set_config_value(db_host, "database", "host")
            config_mgr.set_config_value(db_port, "database", "port")
            config_mgr.set_config_value(db_name, "database", "name")
            config_mgr.set_config_value(db_user, "database", "user")
            config_mgr.set_config_value(db_password, "database", "password")
            st.success("✅ Database configuration saved!")
    
    # Email Tab
    with tabs[3]:
        st.markdown("#### Email Configuration (For Notifications)")
        
        smtp_server = st.text_input(
            "SMTP Server", 
            value=config_mgr.get_config_value("email", "smtp_server") or "",
            key="smtp_server"
        )
        smtp_port = st.number_input(
            "SMTP Port", 
            value=config_mgr.get_config_value("email", "smtp_port") or 587,
            key="smtp_port"
        )
        sender_email = st.text_input(
            "Sender Email", 
            value=config_mgr.get_config_value("email", "sender_email") or "",
            key="sender_email"
        )
        sender_password = st.text_input(
            "Email Password", 
            value=config_mgr.get_config_value("email", "sender_password") or "", 
            type="password",
            key="sender_password"
        )
        
        if st.button("Save Email Config", key="save_email"):
            config_mgr.set_config_value(smtp_server, "email", "smtp_server")
            config_mgr.set_config_value(smtp_port, "email", "smtp_port")
            config_mgr.set_config_value(sender_email, "email", "sender_email")
            config_mgr.set_config_value(sender_password, "email", "sender_password")
            st.success("✅ Email configuration saved!")
    
    # App Settings Tab
    with tabs[4]:
        st.markdown("#### Application Settings")
        
        max_file_size = st.number_input(
            "Max File Size (MB)",
            value=config_mgr.get_config_value("app_settings", "max_file_size_mb") or 200,
            min_value=1,
            max_value=1000,
            key="max_file_size"
        )
        
        default_export = st.selectbox(
            "Default Export Format",
            ["csv", "excel", "json", "parquet"],
            index=["csv", "excel", "json", "parquet"].index(
                config_mgr.get_config_value("app_settings", "default_export_format") or "csv"
            ),
            key="default_export"
        )
        
        enable_logging = st.checkbox(
            "Enable Logging",
            value=config_mgr.get_config_value("app_settings", "enable_logging") or True,
            key="enable_logging"
        )
        
        chunk_size = st.number_input(
            "Data Processing Chunk Size",
            value=config_mgr.get_config_value("data_processing", "chunk_size") or 10000,
            min_value=1000,
            max_value=100000,
            key="chunk_size"
        )
        
        if st.button("Save App Settings", key="save_app_settings"):
            config_mgr.set_config_value(max_file_size, "app_settings", "max_file_size_mb")
            config_mgr.set_config_value(default_export, "app_settings", "default_export_format")
            config_mgr.set_config_value(enable_logging, "app_settings", "enable_logging")
            config_mgr.set_config_value(chunk_size, "data_processing", "chunk_size")
            st.success("✅ App settings saved!")
    
    # Import/Export Tab
    with tabs[5]:
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
            
            if issues.get("warnings"):
                st.info("ℹ️ Warnings:")
                for item in issues["warnings"]:
                    st.write(f"- {item}")
            
            if not issues["missing"] and not issues["invalid"]:
                st.success("✅ Configuration is valid!")


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Configuration Manager", layout="wide")
    config_ui()