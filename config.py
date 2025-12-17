"""
Enhanced Configuration Manager with API Registration and Auto-Detection
Seamlessly integrates API configuration with chatbot capabilities
"""

import os
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
import streamlit as st


class ConfigManager:
    """Manages API keys and configuration settings with chatbot integration"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.registered_apis = []
        self._detect_available_apis()
    
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
                "anthropic": {
                    "api_key": "",
                    "model": "claude-3-5-sonnet-20241022",  # Latest recommended
                    "enabled": False,
                    "priority": 1  # Lower number = higher priority
                },
                "openai": {
                    "api_key": "",
                    "model": "gpt-4o",  # Latest GPT-4 optimized
                    "enabled": False,
                    "priority": 2
                },
                "google_gemini": {
                    "api_key": "",
                    "model": "gemini-2.0-flash-exp",  # Latest experimental
                    "enabled": False,
                    "priority": 3
                },
                "google_cloud": {
                    "api_key": "",
                    "project_id": "",
                    "enabled": False,
                    "priority": 4
                },
                "firebase": {
                    "api_key": "",
                    "project_id": "",
                    "database_url": "",
                    "storage_bucket": "",
                    "enabled": False,
                    "priority": 5
                }
            },
            "chatbot": {
                "auto_detect_api": True,
                "fallback_to_local": True,
                "preferred_provider": "anthropic",
                "max_tokens": 1000,
                "temperature": 0.7
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
    
    def _detect_available_apis(self):
        """Auto-detect which APIs are configured and available"""
        self.registered_apis = []
        
        # Check each API provider
        for provider, config in self.config.get("apis", {}).items():
            if config.get("api_key") and config.get("api_key").strip():
                # Verify the API is actually usable
                if self._verify_api_availability(provider):
                    self.registered_apis.append({
                        'provider': provider,
                        'priority': config.get('priority', 99),
                        'enabled': config.get('enabled', True),
                        'model': config.get('model', 'default')
                    })
        
        # Sort by priority
        self.registered_apis.sort(key=lambda x: x['priority'])
    
    def _verify_api_availability(self, provider: str) -> bool:
        """Verify if an API provider is actually available"""
        try:
            if provider == "anthropic":
                import anthropic
                return True
            elif provider == "openai":
                import openai
                return True
            elif provider == "google_gemini":
                import google.generativeai
                return True
            elif provider == "google_cloud":
                import google.cloud
                return True
            return False
        except ImportError:
            return False
    
    def get_available_apis(self) -> List[Dict[str, Any]]:
        """Get list of available and configured APIs"""
        return self.registered_apis
    
    def get_best_api(self) -> Optional[Dict[str, Any]]:
        """Get the highest priority available API"""
        enabled_apis = [api for api in self.registered_apis if api['enabled']]
        return enabled_apis[0] if enabled_apis else None
    
    def set_api_key(self, service: str, api_key: str, auto_enable: bool = True) -> bool:
        """Set API key and optionally auto-enable for chatbot"""
        if "apis" not in self.config:
            self.config["apis"] = {}
        if service not in self.config["apis"]:
            self.config["apis"][service] = {}
        
        self.config["apis"][service]["api_key"] = api_key
        
        # Auto-enable if requested and API is available
        if auto_enable and api_key.strip():
            self.config["apis"][service]["enabled"] = True
        
        success = self.save_config()
        
        if success:
            # Re-detect available APIs
            self._detect_available_apis()
        
        return success
    
    def enable_api(self, service: str, enabled: bool = True) -> bool:
        """Enable or disable an API for chatbot use"""
        if service in self.config.get("apis", {}):
            self.config["apis"][service]["enabled"] = enabled
            success = self.save_config()
            if success:
                self._detect_available_apis()
            return success
        return False
    
    def set_api_priority(self, service: str, priority: int) -> bool:
        """Set priority for API (lower = higher priority)"""
        if service in self.config.get("apis", {}):
            self.config["apis"][service]["priority"] = priority
            success = self.save_config()
            if success:
                self._detect_available_apis()
            return success
        return False
    
    def get_chatbot_config(self) -> Dict[str, Any]:
        """Get chatbot-specific configuration"""
        return self.config.get("chatbot", {
            "auto_detect_api": True,
            "fallback_to_local": True,
            "preferred_provider": "anthropic"
        })
    
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
    
    def export_api_status_report(self) -> str:
        """Generate a detailed API status report"""
        report = []
        report.append("=" * 50)
        report.append("API CONFIGURATION STATUS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Registered APIs
        if self.registered_apis:
            report.append("🟢 CONFIGURED & AVAILABLE APIs:")
            for api in self.registered_apis:
                status = "✅ Enabled" if api['enabled'] else "⏸️ Disabled"
                report.append(f"  • {api['provider'].title()}: {status} (Priority: {api['priority']})")
        else:
            report.append("🔴 No APIs configured")
        
        report.append("")
        
        # Chatbot status
        chatbot_config = self.get_chatbot_config()
        report.append("CHATBOT CONFIGURATION:")
        report.append(f"  • Auto-detect: {chatbot_config.get('auto_detect_api', True)}")
        report.append(f"  • Fallback to local: {chatbot_config.get('fallback_to_local', True)}")
        report.append(f"  • Preferred provider: {chatbot_config.get('preferred_provider', 'None')}")
        
        report.append("")
        
        # Best available API
        best_api = self.get_best_api()
        if best_api:
            report.append(f"🎯 ACTIVE API: {best_api['provider'].title()}")
        else:
            report.append("⚠️ NO ACTIVE API - Using local mode")
        
        report.append("")
        report.append("=" * 50)
        
        return "\n".join(report)


# Enhanced Streamlit UI with API Registration Features
def config_ui():
    """Enhanced Streamlit interface for managing configuration with API status"""
    st.subheader("⚙️ Configuration Manager")
    
    # Initialize config manager in session state
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    config_mgr = st.session_state.config_manager
    
    # Show API Status Banner
    show_api_status_banner(config_mgr)
    
    st.markdown("---")
    
    tabs = st.tabs([
        "🔑 API Keys",
        "🤖 Chatbot Config", 
        "🗄️ Database", 
        "📧 Email", 
        "⚙️ App Settings", 
        "📤 Import/Export"
    ])
    
    # API Keys Tab
    with tabs[0]:
        show_api_keys_tab(config_mgr)
    
    # Chatbot Configuration Tab
    with tabs[1]:
        show_chatbot_config_tab(config_mgr)
    
    # Database Tab
    with tabs[2]:
        show_database_tab(config_mgr)
    
    # Email Tab
    with tabs[3]:
        show_email_tab(config_mgr)
    
    # App Settings Tab
    with tabs[4]:
        show_app_settings_tab(config_mgr)
    
    # Import/Export Tab
    with tabs[5]:
        show_import_export_tab(config_mgr)


def show_api_status_banner(config_mgr):
    """Display API status banner at top"""
    st.markdown("### 🔌 API Status")
    
    available_apis = config_mgr.get_available_apis()
    best_api = config_mgr.get_best_api()
    
    if best_api:
        st.success(f"🟢 **Active:** {best_api['provider'].title()} (Priority {best_api['priority']})")
    else:
        st.warning("🟡 **No API Configured** - Chatbot using local mode")
    
    if available_apis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Configured APIs", len(available_apis))
        col2.metric("Enabled APIs", len([a for a in available_apis if a['enabled']]))
        col3.metric("Active Provider", best_api['provider'].title() if best_api else "Local")
        
        # Show all configured APIs
        with st.expander("📋 View All Configured APIs"):
            for api in available_apis:
                status_icon = "✅" if api['enabled'] else "⏸️"
                st.write(f"{status_icon} **{api['provider'].title()}** - Priority: {api['priority']}, Model: {api['model']}")
    
    # Quick refresh button
    if st.button("🔄 Refresh API Status"):
        config_mgr._detect_available_apis()
        st.rerun()


def show_api_keys_tab(config_mgr):
    """Enhanced API keys tab with auto-registration"""
    st.markdown("#### Configure API Keys for Enhanced Chatbot")
    st.info("💡 Configure any API to unlock AI-powered chatbot responses!")
    
    # Anthropic Configuration
    with st.expander("🟣 Anthropic (Claude) - Recommended", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=config_mgr.get_config_value("apis", "anthropic", "api_key") or "",
                type="password",
                key="anthropic_key"
            )
        
        with col2:
            anthropic_enabled = st.checkbox(
                "Enable",
                value=config_mgr.get_config_value("apis", "anthropic", "enabled") or False,
                key="anthropic_enabled"
            )
        
        # Model selection with custom option
        st.markdown("**Model Selection:**")
        model_col1, model_col2 = st.columns([2, 2])
        
        with model_col1:
            anthropic_model_preset = st.selectbox(
                "Preset Models",
                ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "custom"],
                index=0,
                key="anthropic_model_preset",
                help="Select a preset model or choose 'custom' to enter your own"
            )
        
        with model_col2:
            if anthropic_model_preset == "custom":
                anthropic_model_custom = st.text_input(
                    "Custom Model Name",
                    value=config_mgr.get_config_value("apis", "anthropic", "model") or "",
                    placeholder="e.g., claude-3-5-sonnet-20250101",
                    key="anthropic_model_custom"
                )
                anthropic_model = anthropic_model_custom
            else:
                anthropic_model = anthropic_model_preset
                st.info(f"Using: {anthropic_model_preset}")
        
        anthropic_priority = st.slider(
            "Priority (1=highest)",
            min_value=1,
            max_value=10,
            value=config_mgr.get_config_value("apis", "anthropic", "priority") or 1,
            key="anthropic_priority"
        )
        
        if st.button("💾 Save Anthropic Config", key="save_anthropic"):
            config_mgr.set_api_key("anthropic", anthropic_key, auto_enable=anthropic_enabled)
            config_mgr.set_config_value(anthropic_model, "apis", "anthropic", "model")
            config_mgr.set_api_priority("anthropic", anthropic_priority)
            config_mgr.enable_api("anthropic", anthropic_enabled)
            
            st.success("✅ Anthropic API configured!")
            st.info(f"📋 Model: {anthropic_model}")
            
            # Notify about chatbot
            if anthropic_key.strip() and anthropic_enabled:
                st.success("🤖 **Chatbot now using Anthropic Claude!**")
                
                # Reload chatbot if it exists
                if 'pipeline' in st.session_state and hasattr(st.session_state.pipeline, 'chatbot'):
                    st.session_state.pipeline.chatbot.reload_api_config()
            
            st.rerun()
    
    st.markdown("---")
    
    # OpenAI Configuration
    with st.expander("🟢 OpenAI (GPT)"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=config_mgr.get_config_value("apis", "openai", "api_key") or "",
                type="password",
                key="openai_key"
            )
        
        with col2:
            openai_enabled = st.checkbox(
                "Enable",
                value=config_mgr.get_config_value("apis", "openai", "enabled") or False,
                key="openai_enabled"
            )
        
        # Model selection with custom option
        st.markdown("**Model Selection:**")
        model_col1, model_col2 = st.columns([2, 2])
        
        with model_col1:
            openai_model_preset = st.selectbox(
                "Preset Models",
                ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "custom"],
                index=0,
                key="openai_model_preset",
                help="Select a preset model or choose 'custom' to enter your own"
            )
        
        with model_col2:
            if openai_model_preset == "custom":
                openai_model_custom = st.text_input(
                    "Custom Model Name",
                    value=config_mgr.get_config_value("apis", "openai", "model") or "",
                    placeholder="e.g., gpt-4-turbo-2024-04-09",
                    key="openai_model_custom"
                )
                openai_model = openai_model_custom
            else:
                openai_model = openai_model_preset
                st.info(f"Using: {openai_model_preset}")
        
        openai_priority = st.slider(
            "Priority (1=highest)",
            min_value=1,
            max_value=10,
            value=config_mgr.get_config_value("apis", "openai", "priority") or 2,
            key="openai_priority"
        )
        
        if st.button("💾 Save OpenAI Config", key="save_openai"):
            config_mgr.set_api_key("openai", openai_key, auto_enable=openai_enabled)
            config_mgr.set_config_value(openai_model, "apis", "openai", "model")
            config_mgr.set_api_priority("openai", openai_priority)
            config_mgr.enable_api("openai", openai_enabled)
            
            st.success("✅ OpenAI API configured!")
            st.info(f"📋 Model: {openai_model}")
            
            if openai_key.strip() and openai_enabled:
                st.success("🤖 **Chatbot now using OpenAI GPT!**")
                
                if 'pipeline' in st.session_state and hasattr(st.session_state.pipeline, 'chatbot'):
                    st.session_state.pipeline.chatbot.reload_api_config()
            
            st.rerun()
    
    st.markdown("---")
    
    # Google Gemini Configuration
    with st.expander("🔵 Google Gemini"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            gemini_key = st.text_input(
                "Google Gemini API Key",
                value=config_mgr.get_config_value("apis", "google_gemini", "api_key") or "",
                type="password",
                key="gemini_key"
            )
        
        with col2:
            gemini_enabled = st.checkbox(
                "Enable",
                value=config_mgr.get_config_value("apis", "google_gemini", "enabled") or False,
                key="gemini_enabled"
            )
        
        # Model selection with custom option
        st.markdown("**Model Selection:**")
        model_col1, model_col2 = st.columns([2, 2])
        
        with model_col1:
            gemini_model_preset = st.selectbox(
                "Preset Models",
                ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision", "custom"],
                index=0,
                key="gemini_model_preset",
                help="Select a preset model or choose 'custom' to enter your own"
            )
        
        with model_col2:
            if gemini_model_preset == "custom":
                gemini_model_custom = st.text_input(
                    "Custom Model Name",
                    value=config_mgr.get_config_value("apis", "google_gemini", "model") or "",
                    placeholder="e.g., gemini-2.0-flash",
                    key="gemini_model_custom"
                )
                gemini_model = gemini_model_custom
            else:
                gemini_model = gemini_model_preset
                st.info(f"Using: {gemini_model_preset}")
        
        gemini_priority = st.slider(
            "Priority (1=highest)",
            min_value=1,
            max_value=10,
            value=config_mgr.get_config_value("apis", "google_gemini", "priority") or 3,
            key="gemini_priority"
        )
        
        if st.button("💾 Save Gemini Config", key="save_gemini"):
            config_mgr.set_api_key("google_gemini", gemini_key, auto_enable=gemini_enabled)
            config_mgr.set_config_value(gemini_model, "apis", "google_gemini", "model")
            config_mgr.set_api_priority("google_gemini", gemini_priority)
            config_mgr.enable_api("google_gemini", gemini_enabled)
            
            st.success("✅ Google Gemini API configured!")
            st.info(f"📋 Model: {gemini_model}")
            
            if gemini_key.strip() and gemini_enabled:
                st.success("🤖 **Chatbot now using Google Gemini!**")
                
                if 'pipeline' in st.session_state and hasattr(st.session_state.pipeline, 'chatbot'):
                    st.session_state.pipeline.chatbot.reload_api_config()
            
            st.rerun()
    
    st.markdown("---")
    
    # Google Cloud / Firebase
    with st.expander("🔵 Google Cloud & Firebase"):
        st.markdown("##### Google Cloud AI")
        google_api_key = st.text_input(
            "Google Cloud API Key",
            value=config_mgr.get_config_value("apis", "google_cloud", "api_key") or "",
            type="password",
            key="google_key"
        )
        
        google_project_id = st.text_input(
            "Project ID",
            value=config_mgr.get_config_value("apis", "google_cloud", "project_id") or "",
            key="google_project"
        )
        
        st.markdown("##### Firebase Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            firebase_api_key = st.text_input(
                "Firebase API Key",
                value=config_mgr.get_config_value("apis", "firebase", "api_key") or "",
                type="password",
                key="firebase_key"
            )
            firebase_db_url = st.text_input(
                "Database URL",
                value=config_mgr.get_config_value("apis", "firebase", "database_url") or "",
                key="firebase_db"
            )
        
        with col2:
            firebase_project_id = st.text_input(
                "Project ID",
                value=config_mgr.get_config_value("apis", "firebase", "project_id") or "",
                key="firebase_project"
            )
            firebase_storage = st.text_input(
                "Storage Bucket",
                value=config_mgr.get_config_value("apis", "firebase", "storage_bucket") or "",
                key="firebase_storage"
            )
        
        if st.button("💾 Save Google/Firebase Config", key="save_google"):
            # Google Cloud
            config_mgr.set_config_value(google_api_key, "apis", "google_cloud", "api_key")
            config_mgr.set_config_value(google_project_id, "apis", "google_cloud", "project_id")
            
            # Firebase
            config_mgr.set_api_key("firebase", firebase_api_key)
            config_mgr.set_config_value(firebase_project_id, "apis", "firebase", "project_id")
            config_mgr.set_config_value(firebase_db_url, "apis", "firebase", "database_url")
            config_mgr.set_config_value(firebase_storage, "apis", "firebase", "storage_bucket")
            
            st.success("✅ Google/Firebase configuration saved!")
            st.rerun()


def show_chatbot_config_tab(config_mgr):
    """Chatbot-specific configuration"""
    st.markdown("#### 🤖 Chatbot Behavior Settings")
    
    chatbot_config = config_mgr.get_chatbot_config()
    
    auto_detect = st.checkbox(
        "Auto-detect Best API",
        value=chatbot_config.get('auto_detect_api', True),
        help="Automatically use the highest priority available API"
    )
    
    fallback_local = st.checkbox(
        "Fallback to Local Mode",
        value=chatbot_config.get('fallback_to_local', True),
        help="Use rule-based responses if API calls fail"
    )
    
    # Preferred provider dropdown
    available_providers = ["anthropic", "openai", "google_gemini"]
    current_pref = chatbot_config.get('preferred_provider', 'anthropic')
    
    preferred = st.selectbox(
        "Preferred API Provider",
        available_providers,
        index=available_providers.index(current_pref) if current_pref in available_providers else 0,
        help="Which API to prefer when multiple are available"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        max_tokens = st.slider(
            "Max Response Tokens",
            min_value=100,
            max_value=4000,
            value=chatbot_config.get('max_tokens', 1000),
            step=100
        )
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=chatbot_config.get('temperature', 0.7),
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
    
    if st.button("💾 Save Chatbot Settings", type="primary"):
        config_mgr.set_config_value(auto_detect, "chatbot", "auto_detect_api")
        config_mgr.set_config_value(fallback_local, "chatbot", "fallback_to_local")
        config_mgr.set_config_value(preferred, "chatbot", "preferred_provider")
        config_mgr.set_config_value(max_tokens, "chatbot", "max_tokens")
        config_mgr.set_config_value(temperature, "chatbot", "temperature")
        
        st.success("✅ Chatbot settings saved!")
        
        # Reload chatbot
        if 'pipeline' in st.session_state and hasattr(st.session_state.pipeline, 'chatbot'):
            st.session_state.pipeline.chatbot.reload_api_config()
            st.info("🔄 Chatbot reloaded with new settings")
        
        st.rerun()
    
    # Show current API status
    st.markdown("---")
    st.markdown("#### 📊 Current API Status")
    
    status_report = config_mgr.export_api_status_report()
    st.code(status_report, language="text")


def show_database_tab(config_mgr):
    """Database configuration tab"""
    st.markdown("#### Database Connection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        db_host = st.text_input("Host", value=config_mgr.get_config_value("database", "host") or "localhost")
        db_port = st.number_input("Port", value=config_mgr.get_config_value("database", "port") or 5432)
        db_name = st.text_input("Database Name", value=config_mgr.get_config_value("database", "name") or "")
    
    with col2:
        db_user = st.text_input("Username", value=config_mgr.get_config_value("database", "user") or "")
        db_password = st.text_input("Password", value=config_mgr.get_config_value("database", "password") or "", type="password")
    
    if st.button("💾 Save Database Config"):
        config_mgr.set_config_value(db_host, "database", "host")
        config_mgr.set_config_value(db_port, "database", "port")
        config_mgr.set_config_value(db_name, "database", "name")
        config_mgr.set_config_value(db_user, "database", "user")
        config_mgr.set_config_value(db_password, "database", "password")
        st.success("✅ Database configuration saved!")


def show_email_tab(config_mgr):
    """Email configuration tab"""
    st.markdown("#### Email Configuration (For Notifications)")
    
    smtp_server = st.text_input("SMTP Server", value=config_mgr.get_config_value("email", "smtp_server") or "")
    smtp_port = st.number_input("SMTP Port", value=config_mgr.get_config_value("email", "smtp_port") or 587)
    sender_email = st.text_input("Sender Email", value=config_mgr.get_config_value("email", "sender_email") or "")
    sender_password = st.text_input("Email Password", value=config_mgr.get_config_value("email", "sender_password") or "", type="password")
    
    if st.button("💾 Save Email Config"):
        config_mgr.set_config_value(smtp_server, "email", "smtp_server")
        config_mgr.set_config_value(smtp_port, "email", "smtp_port")
        config_mgr.set_config_value(sender_email, "email", "sender_email")
        config_mgr.set_config_value(sender_password, "email", "sender_password")
        st.success("✅ Email configuration saved!")


def show_app_settings_tab(config_mgr):
    """Application settings tab"""
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
    
    if st.button("💾 Save App Settings"):
        config_mgr.set_config_value(max_file_size, "app_settings", "max_file_size_mb")
        config_mgr.set_config_value(default_export, "app_settings", "default_export_format")
        config_mgr.set_config_value(enable_logging, "app_settings", "enable_logging")
        config_mgr.set_config_value(chunk_size, "data_processing", "chunk_size")
        st.success("✅ App settings saved!")


def show_import_export_tab(config_mgr):
    """Import/Export tab"""
    st.markdown("#### Export Configuration")
    
    exclude_sensitive = st.checkbox("Exclude sensitive data (API keys, passwords)", value=True)
    
    if st.button("📤 Export Configuration"):
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
    
    if uploaded_config and st.button("📥 Import Configuration"):
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


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Enhanced Configuration Manager", layout="wide")
    config_ui()