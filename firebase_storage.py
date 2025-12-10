"""
Firebase Storage Implementation
Complete storage system with Firebase Firestore integration
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage as firebase_storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("⚠️ Firebase Admin SDK not installed. Run: pip install firebase-admin")


class FirebaseStorageManager:
    """
    Complete Firebase Storage Manager
    Handles all storage operations with Firestore
    """
    
    def __init__(self, config_path: str = None, use_firebase: bool = True):
        """
        Initialize Firebase Storage Manager
        
        Args:
            config_path: Path to Firebase service account JSON
            use_firebase: Whether to use Firebase or fallback to local storage
        """
        self.use_firebase = use_firebase and FIREBASE_AVAILABLE
        self.db = None
        self.bucket = None
        self.initialized = False
        
        if self.use_firebase:
            self._initialize_firebase(config_path)
        else:
            print("📦 Using local storage (Firebase disabled)")
            self._initialize_local_storage()
    
    def _initialize_firebase(self, config_path: str = None):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                # Use provided config or environment variable
                if config_path and os.path.exists(config_path):
                    cred = credentials.Certificate(config_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'trading-app-b1d30.appspot.com'
                    })
                elif os.environ.get('FIREBASE_SERVICE_ACCOUNT'):
                    cred = credentials.Certificate(
                        json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT'))
                    )
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'trading-app-b1d30.appspot.com'
                    })
                else:
                    # Try default credentials
                    firebase_admin.initialize_app()
            
            self.db = firestore.client()
            self.bucket = firebase_storage.bucket()
            self.initialized = True
            print("✅ Firebase initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Firebase initialization failed: {e}")
            print("📦 Falling back to local storage")
            self.use_firebase = False
            self._initialize_local_storage()
    
    def _initialize_local_storage(self):
        """Initialize local file-based storage as fallback"""
        self.local_db_path = 'local_storage.json'
        self.local_files_path = 'local_files'
        
        # Create directories
        os.makedirs(self.local_files_path, exist_ok=True)
        
        # Load or create local database
        if os.path.exists(self.local_db_path):
            with open(self.local_db_path, 'r') as f:
                self.local_db = json.load(f)
        else:
            self.local_db = {}
            self._save_local_db()
        
        self.initialized = True
        print("✅ Local storage initialized")
    
    def _save_local_db(self):
        """Save local database to file"""
        with open(self.local_db_path, 'w') as f:
            json.dump(self.local_db, f, indent=2)
    
    # =====================================================================
    # CORE STORAGE OPERATIONS
    # =====================================================================
    
    def set(self, key: str, value: Any, collection: str = 'default', 
            shared: bool = False, user_id: str = None) -> bool:
        """
        Store data
        
        Args:
            key: Storage key
            value: Data to store (will be JSON serialized)
            collection: Collection/namespace
            shared: Whether data is shared across users
            user_id: User identifier (required if not shared)
        
        Returns:
            Success boolean
        """
        try:
            if self.use_firebase:
                return self._firebase_set(key, value, collection, shared, user_id)
            else:
                return self._local_set(key, value, collection, shared, user_id)
        except Exception as e:
            print(f"❌ Error storing data: {e}")
            return False
    
    def get(self, key: str, collection: str = 'default', 
            shared: bool = False, user_id: str = None) -> Optional[Any]:
        """
        Retrieve data
        
        Args:
            key: Storage key
            collection: Collection/namespace
            shared: Whether data is shared
            user_id: User identifier
        
        Returns:
            Stored data or None
        """
        try:
            if self.use_firebase:
                return self._firebase_get(key, collection, shared, user_id)
            else:
                return self._local_get(key, collection, shared, user_id)
        except Exception as e:
            print(f"❌ Error retrieving data: {e}")
            return None
    
    def delete(self, key: str, collection: str = 'default',
               shared: bool = False, user_id: str = None) -> bool:
        """Delete data"""
        try:
            if self.use_firebase:
                return self._firebase_delete(key, collection, shared, user_id)
            else:
                return self._local_delete(key, collection, shared, user_id)
        except Exception as e:
            print(f"❌ Error deleting data: {e}")
            return False
    
    def list_keys(self, collection: str = 'default', prefix: str = None,
                  shared: bool = False, user_id: str = None) -> List[str]:
        """List all keys in collection"""
        try:
            if self.use_firebase:
                return self._firebase_list(collection, prefix, shared, user_id)
            else:
                return self._local_list(collection, prefix, shared, user_id)
        except Exception as e:
            print(f"❌ Error listing keys: {e}")
            return []
    
    # =====================================================================
    # FIREBASE OPERATIONS
    # =====================================================================
    
    def _firebase_set(self, key: str, value: Any, collection: str,
                      shared: bool, user_id: str) -> bool:
        """Store data in Firebase Firestore"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        
        # Prepare document
        doc_data = {
            'key': key,
            'value': json.dumps(value) if not isinstance(value, str) else value,
            'collection': collection,
            'shared': shared,
            'user_id': user_id if not shared else None,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        
        # Store in Firestore
        self.db.collection(collection).document(doc_path).set(doc_data)
        return True
    
    def _firebase_get(self, key: str, collection: str,
                      shared: bool, user_id: str) -> Optional[Any]:
        """Retrieve data from Firebase Firestore"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        
        doc = self.db.collection(collection).document(doc_path).get()
        
        if doc.exists:
            data = doc.to_dict()
            value = data.get('value')
            
            # Try to parse JSON
            try:
                return json.loads(value)
            except:
                return value
        
        return None
    
    def _firebase_delete(self, key: str, collection: str,
                        shared: bool, user_id: str) -> bool:
        """Delete data from Firebase Firestore"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        self.db.collection(collection).document(doc_path).delete()
        return True
    
    def _firebase_list(self, collection: str, prefix: str,
                       shared: bool, user_id: str) -> List[str]:
        """List keys from Firebase Firestore"""
        query = self.db.collection(collection)
        
        # Filter by user if not shared
        if not shared and user_id:
            query = query.where('user_id', '==', user_id)
        elif shared:
            query = query.where('shared', '==', True)
        
        # Filter by prefix
        if prefix:
            query = query.where('key', '>=', prefix).where('key', '<', prefix + '\uf8ff')
        
        docs = query.stream()
        return [doc.to_dict().get('key') for doc in docs]
    
    # =====================================================================
    # LOCAL STORAGE OPERATIONS
    # =====================================================================
    
    def _local_set(self, key: str, value: Any, collection: str,
                   shared: bool, user_id: str) -> bool:
        """Store data locally"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        
        if collection not in self.local_db:
            self.local_db[collection] = {}
        
        self.local_db[collection][doc_path] = {
            'key': key,
            'value': value,
            'collection': collection,
            'shared': shared,
            'user_id': user_id if not shared else None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self._save_local_db()
        return True
    
    def _local_get(self, key: str, collection: str,
                   shared: bool, user_id: str) -> Optional[Any]:
        """Retrieve data locally"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        
        if collection in self.local_db and doc_path in self.local_db[collection]:
            return self.local_db[collection][doc_path]['value']
        
        return None
    
    def _local_delete(self, key: str, collection: str,
                      shared: bool, user_id: str) -> bool:
        """Delete data locally"""
        doc_path = self._get_document_path(key, collection, shared, user_id)
        
        if collection in self.local_db and doc_path in self.local_db[collection]:
            del self.local_db[collection][doc_path]
            self._save_local_db()
            return True
        
        return False
    
    def _local_list(self, collection: str, prefix: str,
                    shared: bool, user_id: str) -> List[str]:
        """List keys locally"""
        if collection not in self.local_db:
            return []
        
        keys = []
        for doc_path, doc_data in self.local_db[collection].items():
            # Filter by shared/user
            if shared and not doc_data.get('shared'):
                continue
            if not shared and user_id and doc_data.get('user_id') != user_id:
                continue
            
            key = doc_data['key']
            
            # Filter by prefix
            if prefix and not key.startswith(prefix):
                continue
            
            keys.append(key)
        
        return keys
    
    # =====================================================================
    # HELPER METHODS
    # =====================================================================
    
    def _get_document_path(self, key: str, collection: str,
                          shared: bool, user_id: str) -> str:
        """Generate unique document path"""
        if shared:
            return f"shared_{self._hash_key(key)}"
        else:
            user_hash = self._hash_key(user_id) if user_id else "anonymous"
            return f"user_{user_hash}_{self._hash_key(key)}"
    
    def _hash_key(self, key: str) -> str:
        """Hash a key for storage"""
        if not key:
            return "empty"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    # =====================================================================
    # FILE STORAGE OPERATIONS
    # =====================================================================
    
    def upload_file(self, file_path: str, storage_path: str,
                   user_id: str = None) -> Optional[str]:
        """
        Upload file to storage
        
        Args:
            file_path: Local file path
            storage_path: Destination path in storage
            user_id: User identifier
        
        Returns:
            Public URL or local path
        """
        try:
            if self.use_firebase and self.bucket:
                # Upload to Firebase Storage
                blob = self.bucket.blob(storage_path)
                blob.upload_from_filename(file_path)
                blob.make_public()
                return blob.public_url
            else:
                # Copy to local storage
                import shutil
                dest_path = os.path.join(self.local_files_path, storage_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
                return dest_path
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            return None
    
    def download_file(self, storage_path: str, local_path: str) -> bool:
        """Download file from storage"""
        try:
            if self.use_firebase and self.bucket:
                blob = self.bucket.blob(storage_path)
                blob.download_to_filename(local_path)
            else:
                import shutil
                src_path = os.path.join(self.local_files_path, storage_path)
                shutil.copy2(src_path, local_path)
            return True
        except Exception as e:
            print(f"❌ Error downloading file: {e}")
            return False
    
    def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage"""
        try:
            if self.use_firebase and self.bucket:
                blob = self.bucket.blob(storage_path)
                blob.delete()
            else:
                file_path = os.path.join(self.local_files_path, storage_path)
                if os.path.exists(file_path):
                    os.remove(file_path)
            return True
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
            return False
    
    # =====================================================================
    # HIGH-LEVEL OPERATIONS
    # =====================================================================
    
    def save_model(self, model_data: bytes, model_name: str,
                  user_id: str = None, metadata: Dict = None) -> Optional[str]:
        """
        Save ML model to storage
        
        Args:
            model_data: Serialized model bytes
            model_name: Model identifier
            user_id: User identifier
            metadata: Model metadata (metrics, config, etc.)
        
        Returns:
            Storage path or None
        """
        try:
            # Save model file
            temp_path = f"temp_{model_name}.pkl"
            with open(temp_path, 'wb') as f:
                f.write(model_data)
            
            storage_path = f"models/{user_id or 'shared'}/{model_name}.pkl"
            url = self.upload_file(temp_path, storage_path, user_id)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Save metadata
            if metadata:
                self.set(
                    key=f"model_metadata_{model_name}",
                    value=metadata,
                    collection='models',
                    shared=(user_id is None),
                    user_id=user_id
                )
            
            return url
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def load_model(self, model_name: str, user_id: str = None) -> Optional[bytes]:
        """Load ML model from storage"""
        try:
            storage_path = f"models/{user_id or 'shared'}/{model_name}.pkl"
            local_path = f"temp_{model_name}.pkl"
            
            if self.download_file(storage_path, local_path):
                with open(local_path, 'rb') as f:
                    model_data = f.read()
                os.remove(local_path)
                return model_data
            
            return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def save_chat_history(self, messages: List[Dict], user_id: str,
                         session_id: str = None) -> bool:
        """Save chat conversation history"""
        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return self.set(
            key=f"chat_{session_id}",
            value={
                'messages': messages,
                'timestamp': datetime.now().isoformat(),
                'message_count': len(messages)
            },
            collection='chat_history',
            shared=False,
            user_id=user_id
        )
    
    def load_chat_history(self, user_id: str, session_id: str = None) -> List[Dict]:
        """Load chat conversation history"""
        if session_id:
            data = self.get(
                key=f"chat_{session_id}",
                collection='chat_history',
                user_id=user_id
            )
            return data['messages'] if data else []
        else:
            # Get all chat sessions
            keys = self.list_keys(
                collection='chat_history',
                prefix='chat_',
                user_id=user_id
            )
            return keys
    
    def save_recipe(self, recipe_name: str, recipe_data: Dict,
                   user_id: str, shared: bool = False) -> bool:
        """Save cleaning recipe"""
        return self.set(
            key=f"recipe_{recipe_name}",
            value=recipe_data,
            collection='recipes',
            shared=shared,
            user_id=user_id
        )
    
    def load_recipe(self, recipe_name: str, user_id: str = None) -> Optional[Dict]:
        """Load cleaning recipe"""
        # Try user's recipes first
        if user_id:
            recipe = self.get(
                key=f"recipe_{recipe_name}",
                collection='recipes',
                user_id=user_id
            )
            if recipe:
                return recipe
        
        # Try shared recipes
        return self.get(
            key=f"recipe_{recipe_name}",
            collection='recipes',
            shared=True
        )
    
    def list_recipes(self, user_id: str = None, shared: bool = False) -> List[str]:
        """List available recipes"""
        keys = self.list_keys(
            collection='recipes',
            prefix='recipe_',
            shared=shared,
            user_id=user_id
        )
        # Remove 'recipe_' prefix
        return [key.replace('recipe_', '') for key in keys]
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def get_storage_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'storage_type': 'firebase' if self.use_firebase else 'local',
            'initialized': self.initialized,
            'collections': {}
        }
        
        collections = ['default', 'models', 'chat_history', 'recipes']
        
        for collection in collections:
            keys = self.list_keys(collection, user_id=user_id)
            stats['collections'][collection] = {
                'key_count': len(keys),
                'keys': keys[:10]  # First 10 keys
            }
        
        return stats
    
    def clear_user_data(self, user_id: str) -> bool:
        """Clear all data for a user"""
        try:
            collections = ['default', 'models', 'chat_history', 'recipes']
            
            for collection in collections:
                keys = self.list_keys(collection, user_id=user_id)
                for key in keys:
                    self.delete(key, collection, user_id=user_id)
            
            print(f"✅ Cleared data for user: {user_id}")
            return True
        except Exception as e:
            print(f"❌ Error clearing user data: {e}")
            return False


# Singleton instance
_storage_manager = None

def get_storage_manager(config_path: str = None) -> FirebaseStorageManager:
    """Get or create storage manager singleton"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = FirebaseStorageManager(config_path)
    return _storage_manager


# =======================================================================
# TESTING
# =======================================================================

if __name__ == "__main__":
    print("🧪 Testing Firebase Storage Implementation\n")
    
    # Initialize storage
    storage = get_storage_manager()
    
    # Test 1: Basic storage
    print("📝 Test 1: Basic Storage")
    success = storage.set('test_key', {'data': 'Hello World'}, user_id='test_user')
    print(f"   Set: {success}")
    
    value = storage.get('test_key', user_id='test_user')
    print(f"   Get: {value}")
    
    # Test 2: Shared storage
    print("\n🌍 Test 2: Shared Storage")
    storage.set('shared_data', {'message': 'For everyone'}, shared=True)
    shared_value = storage.get('shared_data', shared=True)
    print(f"   Shared: {shared_value}")
    
    # Test 3: List keys
    print("\n📋 Test 3: List Keys")
    storage.set('item_1', 'data1', user_id='test_user')
    storage.set('item_2', 'data2', user_id='test_user')
    keys = storage.list_keys(user_id='test_user')
    print(f"   Keys: {keys}")
    
    # Test 4: Recipe storage
    print("\n📚 Test 4: Recipe Storage")
    recipe = {
        'name': 'Test Recipe',
        'operations': ['remove_duplicates', 'fill_missing']
    }
    storage.save_recipe('test_recipe', recipe, user_id='test_user')
    loaded = storage.load_recipe('test_recipe', user_id='test_user')
    print(f"   Recipe: {loaded}")
    
    # Test 5: Storage stats
    print("\n📊 Test 5: Storage Statistics")
    stats = storage.get_storage_stats(user_id='test_user')
    print(f"   Type: {stats['storage_type']}")
    print(f"   Collections: {list(stats['collections'].keys())}")
    
    print("\n✅ All tests completed!")