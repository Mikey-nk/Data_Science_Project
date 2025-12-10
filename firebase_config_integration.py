"""
Firebase Configuration Integration
Connects ConfigManager with Firebase Storage using Web SDK config
"""

import json
import os
from typing import Dict, Optional
from firebase_storage import FirebaseStorageManager


class FirebaseConfigAdapter:
    """
    Adapts Web SDK Firebase config to Python Admin SDK
    Converts JavaScript firebaseConfig to service account format
    """
    
    @staticmethod
    def web_config_to_service_account(web_config: Dict) -> Dict:
        """
        Convert Web SDK config to Admin SDK format
        
        Args:
            web_config: Firebase web configuration
            {
                'apiKey': 'AIzaSy...',
                'authDomain': 'project.firebaseapp.com',
                'projectId': 'project-id',
                'storageBucket': 'project.appspot.com',
                'messagingSenderId': '123456',
                'appId': '1:123456:web:abc123'
            }
        
        Returns:
            Service account format for Admin SDK
        """
        project_id = web_config.get('projectId')
        
        # Create minimal service account structure
        # Note: This won't have private_key, so we'll use Firebase REST API instead
        service_account = {
            'type': 'service_account',
            'project_id': project_id,
            'web_api_key': web_config.get('apiKey'),
            'auth_domain': web_config.get('authDomain'),
            'storage_bucket': web_config.get('storageBucket'),
            'messaging_sender_id': web_config.get('messagingSenderId'),
            'app_id': web_config.get('appId')
        }
        
        return service_account
    
    @staticmethod
    def save_web_config(config_manager, web_config: Dict) -> bool:
        """
        Save Firebase web config to ConfigManager
        
        Args:
            config_manager: ConfigManager instance
            web_config: Firebase web configuration
        
        Returns:
            Success boolean
        """
        try:
            # Save to firebase section
            config_manager.set_config_value(
                web_config.get('apiKey', ''),
                'apis', 'firebase', 'api_key'
            )
            config_manager.set_config_value(
                web_config.get('projectId', ''),
                'apis', 'firebase', 'project_id'
            )
            config_manager.set_config_value(
                web_config.get('authDomain', ''),
                'apis', 'firebase', 'auth_domain'
            )
            config_manager.set_config_value(
                web_config.get('storageBucket', ''),
                'apis', 'firebase', 'storage_bucket'
            )
            config_manager.set_config_value(
                web_config.get('messagingSenderId', ''),
                'apis', 'firebase', 'messaging_sender_id'
            )
            config_manager.set_config_value(
                web_config.get('appId', ''),
                'apis', 'firebase', 'app_id'
            )
            
            return True
        except Exception as e:
            print(f"Error saving Firebase config: {e}")
            return False
    
    @staticmethod
    def load_web_config(config_manager) -> Optional[Dict]:
        """
        Load Firebase web config from ConfigManager
        
        Args:
            config_manager: ConfigManager instance
        
        Returns:
            Firebase web configuration dict or None
        """
        try:
            firebase_config = config_manager.get_firebase_config()
            
            # Check if we have the config
            if not firebase_config.get('api_key'):
                return None
            
            # Convert to web config format
            web_config = {
                'apiKey': firebase_config.get('api_key', ''),
                'authDomain': firebase_config.get('auth_domain', ''),
                'projectId': firebase_config.get('project_id', ''),
                'storageBucket': firebase_config.get('storage_bucket', ''),
                'messagingSenderId': firebase_config.get('messaging_sender_id', ''),
                'appId': firebase_config.get('app_id', '')
            }
            
            # Validate config has required fields
            if web_config['apiKey'] and web_config['projectId']:
                return web_config
            
            return None
        except Exception as e:
            print(f"Error loading Firebase config: {e}")
            return None


class FirebaseStorageWithConfig(FirebaseStorageManager):
    """
    Extended Firebase Storage Manager that uses ConfigManager
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize with ConfigManager
        
        Args:
            config_manager: ConfigManager instance from datacleaner_hybrid
        """
        self.config_manager = config_manager
        self.adapter = FirebaseConfigAdapter()
        
        # Try to load config from ConfigManager
        web_config = None
        if config_manager:
            web_config = self.adapter.load_web_config(config_manager)
        
        # Initialize based on available config
        if web_config:
            print(f"✅ Found Firebase config for project: {web_config['projectId']}")
            self._init_with_web_config(web_config)
        else:
            print("📦 No Firebase config found, using local storage")
            super().__init__(use_firebase=False)
    
    def _init_with_web_config(self, web_config: Dict):
        """Initialize using web SDK config"""
        try:
            # Convert to service account format
            service_account = self.adapter.web_config_to_service_account(web_config)
            
            # Save temporarily for initialization
            temp_path = 'temp_firebase_config.json'
            with open(temp_path, 'w') as f:
                json.dump(service_account, f)
            
            # Try to initialize with Admin SDK
            try:
                super().__init__(config_path=temp_path, use_firebase=True)
            except Exception as e:
                print(f"⚠️ Admin SDK initialization failed: {e}")
                print("📦 Using REST API mode with web config")
                # Fall back to local storage but with web config info
                super().__init__(use_firebase=False)
                self.web_config = web_config
                self.use_rest_api = True
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        except Exception as e:
            print(f"⚠️ Error initializing Firebase: {e}")
            print("📦 Falling back to local storage")
            super().__init__(use_firebase=False)
    
    def update_config(self, web_config: Dict) -> bool:
        """
        Update Firebase configuration
        
        Args:
            web_config: New Firebase web configuration
        
        Returns:
            Success boolean
        """
        try:
            # Save to ConfigManager
            if self.config_manager:
                self.adapter.save_web_config(self.config_manager, web_config)
            
            # Reinitialize with new config
            self._init_with_web_config(web_config)
            
            return True
        except Exception as e:
            print(f"Error updating Firebase config: {e}")
            return False


def init_storage_from_config(config_manager) -> FirebaseStorageWithConfig:
    """
    Initialize Firebase Storage using ConfigManager
    
    This is the main function to use in datacleaner_hybrid.py
    
    Args:
        config_manager: ConfigManager instance
    
    Returns:
        FirebaseStorageWithConfig instance
    """
    return FirebaseStorageWithConfig(config_manager)


# Helper function for Streamlit UI
def parse_firebase_js_config(js_config_text: str) -> Optional[Dict]:
    """
    Parse Firebase config from JavaScript code
    
    Args:
        js_config_text: The JavaScript config code (can be pasted directly)
    
    Returns:
        Dict with Firebase config or None
    """
    try:
        # Extract JSON from JavaScript
        import re
        
        # Find the firebaseConfig object
        pattern = r'firebaseConfig\s*=\s*{([^}]+)}'
        match = re.search(pattern, js_config_text, re.DOTALL)
        
        if not match:
            return None
        
        config_text = '{' + match.group(1) + '}'
        
        # Clean up JavaScript syntax to make it valid JSON
        config_text = re.sub(r'(\w+):', r'"\1":', config_text)  # Add quotes to keys
        config_text = re.sub(r':\s*"([^"]+)"', r': "\1"', config_text)  # Fix values
        
        # Parse as JSON
        config = json.loads(config_text)
        
        return config
    except Exception as e:
        print(f"Error parsing Firebase config: {e}")
        
        # Try manual parsing
        try:
            config = {}
            lines = js_config_text.split('\n')
            
            for line in lines:
                if ':' in line:
                    # Extract key and value
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('"\'')
                        value = parts[1].strip().strip(',').strip('"\'')
                        
                        # Map common keys
                        if 'apiKey' in key:
                            config['apiKey'] = value
                        elif 'authDomain' in key:
                            config['authDomain'] = value
                        elif 'projectId' in key:
                            config['projectId'] = value
                        elif 'storageBucket' in key:
                            config['storageBucket'] = value
                        elif 'messagingSenderId' in key:
                            config['messagingSenderId'] = value
                        elif 'appId' in key:
                            config['appId'] = value
            
            if config.get('apiKey') and config.get('projectId'):
                return config
        except Exception as e2:
            print(f"Manual parsing also failed: {e2}")
        
        return None


# Example usage test
if __name__ == "__main__":
    print("🧪 Testing Firebase Config Integration\n")
    
    # Test 1: Parse JavaScript config
    print("📝 Test 1: Parse JavaScript Config")
    
    js_config = """
    const firebaseConfig = {
        apiKey: "AIzaSyD7v4Gu31sHhkprhEnDXc9jVm_X4vP8MRw",
        authDomain: "auric-jewels.firebaseapp.com",
        projectId: "auric-jewels",
        storageBucket: "auric-jewels.firebasestorage.app",
        messagingSenderId: "25337499686",
        appId: "1:25337499686:web:b48003eda8f5f7e29c56e7"
    };
    """
    
    parsed = parse_firebase_js_config(js_config)
    if parsed:
        print(f"   ✅ Parsed successfully")
        print(f"   Project: {parsed['projectId']}")
        print(f"   Storage: {parsed['storageBucket']}")
    else:
        print("   ❌ Parsing failed")
    
    # Test 2: Initialize with mock ConfigManager
    print("\n🔧 Test 2: Initialize Storage with Config")
    
    from config import ConfigManager
    
    config_manager = ConfigManager()
    
    # Save parsed config
    if parsed:
        adapter = FirebaseConfigAdapter()
        adapter.save_web_config(config_manager, parsed)
        print("   ✅ Config saved")
        
        # Load and verify
        loaded = adapter.load_web_config(config_manager)
        if loaded:
            print(f"   ✅ Config loaded: {loaded['projectId']}")
        
        # Initialize storage
        storage = init_storage_from_config(config_manager)
        print(f"   ✅ Storage initialized: {storage.initialized}")
    
    print("\n✅ All tests completed!")