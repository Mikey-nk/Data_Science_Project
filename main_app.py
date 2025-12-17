"""
Data Cleaning System - Main Application with Integrated Authentication
Complete integration of Firebase Authentication with the hybrid data cleaning system
"""

import streamlit as st
from firebase_auth import FirebaseAuth
from datacleaner_hybrid import (
    DataPipeline, show_manual_mode, show_assisted_mode, show_automatic_mode,
    show_chatbot_interface
)
from hybrid_intelligence import CleaningMode


class AuthenticatedApp:
    """Main application with authentication wrapper"""
    
    def __init__(self):
        self.auth = FirebaseAuth()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if 'user' not in st.session_state:
            st.session_state.user = None
        
        if 'auth_manager' not in st.session_state:
            st.session_state.auth_manager = self.auth
        
        if 'show_main_app' not in st.session_state:
            st.session_state.show_main_app = False
        
        if 'show_profile' not in st.session_state:
            st.session_state.show_profile = False
        
        if 'show_chatbot' not in st.session_state:
            st.session_state.show_chatbot = False
    
    def render_auth_page(self):
        """Render the authentication page (login/register)"""
        
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
            font-size: 2.5rem;
            font-weight: bold;
        }
        .auth-subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 3rem;
            font-size: 1.1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            padding: 0.75rem;
            font-size: 1rem;
            border-radius: 5px;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1557b0;
        }
        .feature-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
        st.markdown("<h1 class='auth-title'>🧹 Data Cleaning & Profiling System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='auth-subtitle'>AI-Powered Data Cleaning with Intelligent Automation</p>", unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='feature-box' style='text-align: center;'>
                <h3>🤖</h3>
                <p><b>AI Assistant</b></p>
                <small>Smart suggestions</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-box' style='text-align: center;'>
                <h3>📊</h3>
                <p><b>Visual Insights</b></p>
                <small>Interactive dashboards</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='feature-box' style='text-align: center;'>
                <h3>🔮</h3>
                <p><b>ML Models</b></p>
                <small>Predictive analytics</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
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
                            success, user_data, error = self.auth.login_user(login_email, login_password)
                            
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
                        success, error = self.auth.reset_password(login_email)
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
                            success, user_data, error = self.auth.register_user(
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
        <div style='text-align: center; color: gray; font-size: 0.85rem;'>
            <p>🔒 Secured by Firebase Authentication</p>
            <p>Having trouble? Contact support@datacleaner.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def show_welcome_dashboard(self):
        """Show welcome dashboard with user stats"""
        st.title("🎉 Welcome to Your Data Cleaning Workspace!")
        
        user = st.session_state.user
        profile = user.get('profile', {})
        
        # Display user stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👤 Member Since",
                profile.get('created_at', 'N/A')[:10] if profile.get('created_at') else 'N/A'
            )
        
        with col2:
            st.metric(
                "🧹 Total Cleanings",
                profile.get('total_cleanings', 0)
            )
        
        with col3:
            st.metric(
                "🔮 Total Predictions",
                profile.get('total_predictions', 0)
            )
        
        with col4:
            tier = profile.get('subscription_tier', 'free')
            tier_emoji = "🆓" if tier == "free" else "⭐" if tier == "pro" else "💎"
            st.metric(
                "Subscription",
                f"{tier_emoji} {tier.title()}"
            )
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Start Data Cleaning", use_container_width=True):
                st.session_state.show_main_app = True
                st.rerun()
        
        with col2:
            if st.button("📚 View Documentation", use_container_width=True):
                st.info("Documentation will open here")
        
        with col3:
            if st.button("⚙️ Profile Settings", use_container_width=True):
                st.session_state.show_profile = True
                st.rerun()
        
        # Recent activity placeholder
        st.markdown("---")
        st.subheader("📈 Recent Activity")
        st.info("Your recent data cleaning sessions will appear here")
        
        # Feature showcase
        with st.expander("🌟 What's New in This Version"):
            st.markdown("""
            ### ✨ Latest Features
            
            - 🤖 **AI-Powered Cleaning**: Intelligent suggestions with explanations
            - 📊 **Visual Insights**: Interactive dashboards and comparisons
            - ⚡ **Power Tools**: Undo/Redo, Code generation, Recipes
            - 🔮 **ML Models**: Build predictive models from cleaned data
            - 💬 **Chat Assistant**: Ask questions about your data
            - 🧠 **Auto-Learning**: AI learns from your preferences
            """)
    
    def show_profile_settings(self):
        """Show and edit user profile"""
        st.title("⚙️ Profile Settings")
        
        user = st.session_state.user
        profile = user.get('profile', {})
        
        # Back button
        if st.button("← Back to Dashboard"):
            st.session_state.show_profile = False
            st.rerun()
        
        st.markdown("---")
        
        # Profile information
        st.subheader("👤 Your Information")
        
        with st.form("profile_form"):
            new_display_name = st.text_input(
                "Display Name",
                value=user.get('display_name', ''),
                help="This is how we'll address you in the app"
            )
            
            st.text_input(
                "Email",
                value=user.get('email', ''),
                disabled=True,
                help="Email cannot be changed"
            )
            
            # Additional profile fields
            col1, col2 = st.columns(2)
            
            with col1:
                company = st.text_input(
                    "Company/Organization",
                    value=profile.get('company', ''),
                    placeholder="Optional"
                )
            
            with col2:
                role = st.selectbox(
                    "Your Role",
                    ["Data Analyst", "Data Scientist", "Business Analyst", 
                     "Developer", "Student", "Other"],
                    index=0
                )
            
            # Preferences
            st.markdown("### 🎨 Preferences")
            
            default_mode = st.selectbox(
                "Default Cleaning Mode",
                ["Manual", "Assisted", "Automatic"],
                help="Your preferred mode when starting a new cleaning session"
            )
            
            enable_notifications = st.checkbox(
                "Enable email notifications",
                value=True,
                help="Receive updates about your cleaning jobs"
            )
            
            submit_button = st.form_submit_button("💾 Save Changes", use_container_width=True)
            
            if submit_button:
                # Update profile
                updates = {
                    "display_name": new_display_name,
                    "company": company,
                    "role": role,
                    "default_mode": default_mode,
                    "notifications": str(enable_notifications)
                }
                
                success, error = self.auth.update_user_profile(
                    user['id_token'],
                    user['uid'],
                    updates
                )
                
                if success:
                    # Update session state
                    st.session_state.user['display_name'] = new_display_name
                    st.success("✅ Profile updated successfully!")
                else:
                    st.error(f"❌ Failed to update profile: {error}")
        
        # Account management
        st.markdown("---")
        st.subheader("🔐 Account Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔑 Change Password", use_container_width=True):
                st.info("Password reset email will be sent to your email address")
                success, error = self.auth.reset_password(user['email'])
                if success:
                    st.success("📧 Password reset email sent!")
                else:
                    st.error(f"❌ {error}")
        
        with col2:
            st.button("🗑️ Delete Account", use_container_width=True, disabled=True)
            st.caption("Contact support to delete your account")
    
    def show_user_sidebar(self):
        """Display user info in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Profile")
            
            user = st.session_state.user
            st.write(f"**{user['display_name']}**")
            st.caption(user['email'])
            
            if st.button("🚪 Logout", use_container_width=True):
                self.auth.logout_user()
                st.rerun()
    
    def run(self):
        """Main application entry point"""
        
        # Configure page
        st.set_page_config(
            page_title="Data Cleaning & Profiling System",
            page_icon="🧹",
            layout="wide"
        )
        
        # Check authentication
        if not st.session_state.authenticated:
            self.render_auth_page()
            return
        
        # User is authenticated, show user info in sidebar
        self.show_user_sidebar()
        
        # Show app based on state
        if st.session_state.show_profile:
            # Show profile settings
            self.show_profile_settings()
        
        elif st.session_state.show_main_app:
            # Show main data cleaning application
            self.show_main_cleaning_app()
        
        else:
            # Show welcome dashboard
            self.show_welcome_dashboard()
    
    def show_main_cleaning_app(self):
        """Display the main data cleaning application"""
        from datacleaner_hybrid import main as datacleaner_main
        
        # Add a back button in sidebar
        with st.sidebar:
            st.markdown("---")
            if st.button("🏠 Back to Dashboard", use_container_width=True):
                st.session_state.show_main_app = False
                st.rerun()
        
        # Run the main datacleaner app
        datacleaner_main()


def main():
    """Application entry point"""
    app = AuthenticatedApp()
    app.run()


if __name__ == "__main__":
    main()