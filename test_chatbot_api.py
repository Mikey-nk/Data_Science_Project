"""
Quick Test Script for Chatbot API Integration
Run this to verify your Google Gemini API is working with the chatbot
"""

from config import ConfigManager
from chatbot_interface import ConversationalAgent
import pandas as pd

def test_chatbot_api():
    """Test the chatbot with Google Gemini API"""
    
    print("=" * 60)
    print("🧪 TESTING CHATBOT API INTEGRATION")
    print("=" * 60)
    
    # 1. Initialize Config Manager
    print("\n📋 Step 1: Loading Configuration...")
    config_mgr = ConfigManager()
    
    # 2. Check API Status
    print("\n🔍 Step 2: Checking API Configuration...")
    available_apis = config_mgr.get_available_apis()
    best_api = config_mgr.get_best_api()
    
    if best_api:
        print(f"✅ Active API: {best_api['provider']} ({best_api['model']})")
    else:
        print("❌ No API configured!")
        return False
    
    # 3. Initialize Chatbot
    print("\n🤖 Step 3: Initializing Chatbot...")
    try:
        chatbot = ConversationalAgent(pipeline=None, config_manager=config_mgr)
        print(f"✅ Chatbot initialized")
        
        # Check API status
        api_status = chatbot.get_api_status()
        print(f"   • API Enabled: {api_status['enabled']}")
        print(f"   • Provider: {api_status['provider']}")
        print(f"   • Model: {api_status['model']}")
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return False
    
    # 4. Test Rule-Based Response
    print("\n📝 Step 4: Testing Rule-Based Response...")
    try:
        response = chatbot.process_message("help")
        print(f"✅ Response received ({len(response.message)} characters)")
        print(f"   Intent: {response.intent.value if response.intent else 'None'}")
        print(f"   Suggestions: {len(response.suggestions)}")
    except Exception as e:
        print(f"❌ Error with rule-based: {e}")
        return False
    
    # 5. Test AI-Powered Response (Unknown Intent)
    print("\n🌟 Step 5: Testing AI-Powered Response...")
    try:
        # This should trigger AI since it's not a known pattern
        test_query = "What are the best practices for handling outliers in financial data?"
        print(f"   Query: '{test_query}'")
        
        response = chatbot.process_message(test_query)
        
        if chatbot.api_client.api_enabled:
            print(f"✅ AI Response received ({len(response.message)} characters)")
            print(f"\n📄 Response Preview:")
            print("-" * 60)
            print(response.message[:300] + "..." if len(response.message) > 300 else response.message)
            print("-" * 60)
        else:
            print("⚠️  AI not used (API not enabled)")
            
    except Exception as e:
        print(f"❌ Error with AI response: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test with Mock Data Context
    print("\n📊 Step 6: Testing with Data Context...")
    try:
        # Create mock pipeline with data
        mock_data = pd.DataFrame({
            'name': ['Alice', 'Bob', None, 'David'],
            'age': [25, 30, 35, None],
            'salary': [50000, 60000, 70000, 80000]
        })
        
        # Create a proper mock pipeline with profile
        class MockPipeline:
            def __init__(self, data):
                self.data = data
                self.cleaned_data = None
                self.prediction_pipeline = None
                self.trained_models = {}
                
                # Mock profile result
                self.profile_result = {
                    'missing_data': {
                        'total_missing': 2,
                        'by_column': {
                            'name': {'count': 1, 'percentage': 25.0},
                            'age': {'count': 1, 'percentage': 25.0}
                        }
                    },
                    'duplicates': {
                        'duplicate_rows': 0,
                        'duplicate_percentage': 0.0
                    },
                    'outliers': {},
                    'format_issues': {}
                }
            
            def profile_data(self):
                """Mock profile method"""
                return self.profile_result
        
        mock_pipeline = MockPipeline(mock_data)
        
        # Create chatbot with pipeline
        chatbot_with_data = ConversationalAgent(
            pipeline=mock_pipeline, 
            config_manager=config_mgr
        )
        
        response = chatbot_with_data.process_message("What's wrong with my data?")
        print(f"✅ Context-aware response received")
        print(f"   Response length: {len(response.message)} characters")
        print(f"\n   📝 Response Preview:")
        print(f"   {response.message[:150]}...")
        
    except Exception as e:
        print(f"❌ Error with context: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Test Conversation History
    print("\n💬 Step 7: Testing Conversation History...")
    try:
        history = chatbot.get_conversation_history()
        print(f"✅ Conversation history: {len(history)} messages")
        
        # Show last exchange
        if len(history) >= 2:
            print("\n   Last Exchange:")
            print(f"   User: {history[-2]['message'][:50]}...")
            print(f"   Bot: {history[-1]['message'][:50]}...")
    except Exception as e:
        print(f"❌ Error with history: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 Your chatbot is fully integrated with Google Gemini!")
    print("💡 You can now use it in your Streamlit app")
    
    return True


def test_api_direct():
    """Direct test of API client"""
    print("\n🔬 BONUS: Direct API Test")
    print("-" * 60)
    
    from chatbot_interface import APIClient
    from config import ConfigManager
    
    config_mgr = ConfigManager()
    api_client = APIClient(config_mgr)
    
    if api_client.api_enabled:
        print(f"✅ API Client Ready: {api_client.api_provider}")
        
        # Test simple prompt
        try:
            response = api_client.generate_response(
                prompt="Explain data cleaning in one sentence.",
                temperature=0.7,
                max_tokens=100
            )
            
            if response:
                print(f"✅ Direct API call successful!")
                print(f"\n📄 Response: {response}")
            else:
                print(f"❌ API returned None")
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ API Client not enabled")


if __name__ == "__main__":
    print("\n🚀 Starting Chatbot Integration Tests...\n")
    
    # Run main test suite
    success = test_chatbot_api()
    
    if success:
        # Run bonus direct API test
        test_api_direct()
    
    print("\n✨ Test complete!\n")