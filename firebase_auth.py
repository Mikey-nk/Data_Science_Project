"""
Firebase Authentication Module
Handles user registration, login, and session management
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import hashlib
import re


class FirebaseAuth:
    """Firebase Authentication Manager"""
    
    def __init__(self):
        # Firebase Web API configuration
        self.api_key = "AIzaSyDkH5j7nUKs5sf0xhoNtGR3zTuUVz1Aekg"
        self.auth_domain = "trading-app-b1d30.firebaseapp.com"
        self.project_id = "trading-app-b1d30"
        
        # Firebase Auth REST API endpoints
        self.sign_up_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={self.api_key}"
        self.sign_in_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        self.refresh_token_url = f"https://securetoken.googleapis.com/v1/token?key={self.api_key}"
        self.user_info_url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={self.api_key}"
        self.update_profile_url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={self.api_key}"
        self.delete_account_url = f"https://identitytoolkit.googleapis.com/v1/accounts:delete?key={self.api_key}"
        self.reset_password_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={self.api_key}"
        
        # Firestore REST API (for storing user data)
        self.firestore_base = f"https://firestore.googleapis.com/v1/projects/{self.project_id}/databases/(default)/documents"
    
    def register_user(self, email: str, password: str, 
                     display_name: str = None) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Register a new user with Firebase Authentication
        
        Returns:
            Tuple of (success: bool, user_data: Dict, error_message: str)
        """
        try:
            # Validate inputs
            if not self._validate_email(email):
                return False, None, "Invalid email format"
            
            if not self._validate_password(password):
                return False, None, "Password must be at least 6 characters"
            
            # Register with Firebase Auth
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(self.sign_up_url, json=payload)
            
            if response.status_code == 200:
                auth_data = response.json()
                
                # Create user profile in Firestore
                user_profile = {
                    "uid": auth_data["localId"],
                    "email": email,
                    "display_name": display_name or email.split('@')[0],
                    "created_at": datetime.now().isoformat(),
                    "last_login": datetime.now().isoformat(),
                    "subscription_tier": "free",
                    "total_cleanings": 0,
                    "total_predictions": 0
                }
                
                # Save to Firestore
                self._save_user_profile(auth_data["idToken"], user_profile)
                
                return True, {
                    "uid": auth_data["localId"],
                    "email": email,
                    "id_token": auth_data["idToken"],
                    "refresh_token": auth_data["refreshToken"],
                    "display_name": user_profile["display_name"]
                }, None
            else:
                error_data = response.json()
                error_message = self._parse_firebase_error(error_data)
                return False, None, error_message
                
        except Exception as e:
            return False, None, f"Registration error: {str(e)}"
    
    def login_user(self, email: str, password: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Login existing user
        
        Returns:
            Tuple of (success: bool, user_data: Dict, error_message: str)
        """
        try:
            # Validate inputs
            if not email or not password:
                return False, None, "Email and password are required"
            
            # Sign in with Firebase Auth
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(self.sign_in_url, json=payload)
            
            if response.status_code == 200:
                auth_data = response.json()
                
                # Get user profile from Firestore
                user_profile = self._get_user_profile(auth_data["idToken"], auth_data["localId"])
                
                # Update last login
                self._update_last_login(auth_data["idToken"], auth_data["localId"])
                
                return True, {
                    "uid": auth_data["localId"],
                    "email": auth_data["email"],
                    "id_token": auth_data["idToken"],
                    "refresh_token": auth_data["refreshToken"],
                    "display_name": user_profile.get("display_name", email.split('@')[0]),
                    "profile": user_profile
                }, None
            else:
                error_data = response.json()
                error_message = self._parse_firebase_error(error_data)
                return False, None, error_message
                
        except Exception as e:
            return False, None, f"Login error: {str(e)}"
    
    def logout_user(self):
        """Logout current user (clear session state)"""
        if 'user' in st.session_state:
            del st.session_state.user
        if 'authenticated' in st.session_state:
            del st.session_state.authenticated
    
    def get_user_info(self, id_token: str) -> Optional[Dict]:
        """Get user information from Firebase"""
        try:
            payload = {"idToken": id_token}
            response = requests.post(self.user_info_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("users"):
                    return data["users"][0]
            return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def reset_password(self, email: str) -> Tuple[bool, Optional[str]]:
        """Send password reset email"""
        try:
            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }
            
            response = requests.post(self.reset_password_url, json=payload)
            
            if response.status_code == 200:
                return True, None
            else:
                error_data = response.json()
                error_message = self._parse_firebase_error(error_data)
                return False, error_message
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def update_user_profile(self, id_token: str, uid: str, 
                           updates: Dict) -> Tuple[bool, Optional[str]]:
        """Update user profile in Firestore"""
        try:
            # Update in Firestore
            doc_path = f"{self.firestore_base}/users/{uid}"
            
            # Create update mask
            update_fields = {}
            for key, value in updates.items():
                update_fields[key] = {"stringValue": str(value)}
            
            payload = {
                "fields": update_fields
            }
            
            headers = {"Authorization": f"Bearer {id_token}"}
            response = requests.patch(
                f"{doc_path}?updateMask.fieldPaths={','.join(updates.keys())}",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return True, None
            else:
                return False, "Failed to update profile"
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _save_user_profile(self, id_token: str, profile: Dict):
        """Save user profile to Firestore"""
        try:
            doc_path = f"{self.firestore_base}/users/{profile['uid']}"
            
            # Convert to Firestore format
            fields = {}
            for key, value in profile.items():
                if isinstance(value, str):
                    fields[key] = {"stringValue": value}
                elif isinstance(value, int):
                    fields[key] = {"integerValue": str(value)}
                elif isinstance(value, float):
                    fields[key] = {"doubleValue": value}
                elif isinstance(value, bool):
                    fields[key] = {"booleanValue": value}
            
            payload = {"fields": fields}
            headers = {"Authorization": f"Bearer {id_token}"}
            
            requests.patch(doc_path, json=payload, headers=headers)
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def _get_user_profile(self, id_token: str, uid: str) -> Dict:
        """Get user profile from Firestore"""
        try:
            doc_path = f"{self.firestore_base}/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}"}
            
            response = requests.get(doc_path, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # Convert from Firestore format
                profile = {}
                for key, value_obj in data.get("fields", {}).items():
                    if "stringValue" in value_obj:
                        profile[key] = value_obj["stringValue"]
                    elif "integerValue" in value_obj:
                        profile[key] = int(value_obj["integerValue"])
                    elif "doubleValue" in value_obj:
                        profile[key] = value_obj["doubleValue"]
                    elif "booleanValue" in value_obj:
                        profile[key] = value_obj["booleanValue"]
                return profile
            return {}
        except Exception as e:
            print(f"Error getting profile: {e}")
            return {}
    
    def _update_last_login(self, id_token: str, uid: str):
        """Update user's last login timestamp"""
        self.update_user_profile(
            id_token, 
            uid, 
            {"last_login": datetime.now().isoformat()}
        )
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password requirements"""
        return len(password) >= 6
    
    def _parse_firebase_error(self, error_data: Dict) -> str:
        """Parse Firebase error response"""
        error = error_data.get("error", {})
        message = error.get("message", "Unknown error")
        
        # Map common Firebase errors to user-friendly messages
        error_map = {
            "EMAIL_EXISTS": "Email already registered. Please login instead.",
            "INVALID_EMAIL": "Invalid email address.",
            "WEAK_PASSWORD": "Password is too weak. Use at least 6 characters.",
            "EMAIL_NOT_FOUND": "No account found with this email.",
            "INVALID_PASSWORD": "Incorrect password.",
            "USER_DISABLED": "This account has been disabled.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Please try again later."
        }
        
        return error_map.get(message, message)


def render_auth_page():
    """Render the authentication page (login/register)"""
    
    # Initialize auth manager
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = FirebaseAuth()
    
    auth = st.session_state.auth_manager
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
    }
    .auth-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='auth-title'>🧹 Data Cleaning System</h1>", unsafe_allow_html=True)
    
    # Tabs for Login and Register
    auth_tab = st.tabs(["🔐 Login", "📝 Register"])
    
    # LOGIN TAB
    with auth_tab[0]:
        st.subheader("Welcome Back!")
        st.caption("Login to access your data cleaning workspace")
        
        with st.form("login_form"):
            login_email = st.text_input(
                "Email",
                placeholder="your.email@example.com",
                key="login_email"
            )
            
            login_password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col2:
                forgot_password = st.form_submit_button("❓", help="Forgot password?")
            
            if login_button:
                if not login_email or not login_password:
                    st.error("⚠️ Please fill in all fields")
                else:
                    with st.spinner("🔐 Logging in..."):
                        success, user_data, error = auth.login_user(login_email, login_password)
                        
                        if success:
                            # Store user data in session
                            st.session_state.authenticated = True
                            st.session_state.user = user_data
                            st.success(f"✅ Welcome back, {user_data['display_name']}!")
                            st.balloons()
                            # Rerun to show main app
                            st.rerun()
                        else:
                            st.error(f"❌ {error}")
            
            if forgot_password:
                if login_email:
                    success, error = auth.reset_password(login_email)
                    if success:
                        st.success("📧 Password reset email sent! Check your inbox.")
                    else:
                        st.error(f"❌ {error}")
                else:
                    st.warning("⚠️ Please enter your email address first")
    
    # REGISTER TAB
    with auth_tab[1]:
        st.subheader("Create Your Account")
        st.caption("Join us to start cleaning your data!")
        
        with st.form("register_form"):
            reg_name = st.text_input(
                "Display Name",
                placeholder="How should we call you?",
                key="reg_name"
            )
            
            reg_email = st.text_input(
                "Email",
                placeholder="your.email@example.com",
                key="reg_email"
            )
            
            reg_password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters",
                key="reg_password",
                help="Password must be at least 6 characters long"
            )
            
            reg_password_confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                key="reg_password_confirm"
            )
            
            agree_terms = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy",
                key="agree_terms"
            )
            
            register_button = st.form_submit_button("🎉 Create Account", use_container_width=True)
            
            if register_button:
                # Validation
                if not all([reg_name, reg_email, reg_password, reg_password_confirm]):
                    st.error("⚠️ Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("⚠️ Passwords do not match")
                elif not agree_terms:
                    st.error("⚠️ Please agree to the Terms of Service")
                else:
                    with st.spinner("🎨 Creating your account..."):
                        success, user_data, error = auth.register_user(
                            reg_email,
                            reg_password,
                            reg_name
                        )
                        
                        if success:
                            # Store user data in session
                            st.session_state.authenticated = True
                            st.session_state.user = user_data
                            st.success(f"🎉 Welcome aboard, {reg_name}!")
                            st.balloons()
                            # Rerun to show main app
                            st.rerun()
                        else:
                            st.error(f"❌ {error}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        <p>🔒 Secured by Firebase Authentication</p>
        <p>Having trouble? Contact support@datacleaner.com</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def require_authentication():
    """
    Decorator/wrapper to require authentication
    Call this at the start of your main app
    """
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        render_auth_page()
        st.stop()  # Stop execution until authenticated
    
    # User is authenticated, show user info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Profile")
        
        user = st.session_state.user
        st.write(f"**{user['display_name']}**")
        st.caption(user['email'])
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth_manager.logout_user()
            st.rerun()


# Example usage in main app
if __name__ == "__main__":
    st.set_page_config(
        page_title="Data Cleaning System - Auth",
        page_icon="🧹",
        layout="wide"
    )
    
    # Require authentication
    require_authentication()
    
    # Main app content (only shown if authenticated)
    st.title("🎉 Welcome to Your Dashboard!")
    st.write(f"Hello, {st.session_state.user['display_name']}!")
    st.write("You are now authenticated and can access all features.")
    
    # Show user profile
    with st.expander("📊 Your Profile"):
        user = st.session_state.user
        st.json(user.get('profile', {}))