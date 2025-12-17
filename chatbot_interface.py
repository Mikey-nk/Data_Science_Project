"""
Enhanced Chatbot Interface Module with FULL API Integration
Complete implementation with AI-powered responses
"""

import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json


class IntentType(Enum):
    """Types of user intents"""
    # Data operations
    UPLOAD_DATA = "upload_data"
    SHOW_DATA = "show_data"
    PROFILE_DATA = "profile_data"
    CLEAN_DATA = "clean_data"
    
    # Analysis
    ANALYZE_QUALITY = "analyze_quality"
    FIND_ISSUES = "find_issues"
    SHOW_STATISTICS = "show_statistics"
    
    # ML operations
    BUILD_MODEL = "build_model"
    PREDICT = "predict"
    EXPLAIN_MODEL = "explain_model"
    
    # Help & guidance
    GET_HELP = "get_help"
    EXPLAIN_FEATURE = "explain_feature"
    SUGGEST_NEXT = "suggest_next"
    
    # Workflow
    SAVE_WORKFLOW = "save_workflow"
    LOAD_RECIPE = "load_recipe"
    UNDO_ACTION = "undo_action"
    
    # Unknown
    UNKNOWN = "unknown"


class ChatbotResponse:
    """Structured chatbot response"""
    
    def __init__(self, message: str, intent: IntentType = None, 
                 data: Dict = None, suggestions: List[str] = None,
                 action_required: bool = False):
        self.message = message
        self.intent = intent
        self.data = data or {}
        self.suggestions = suggestions or []
        self.action_required = action_required
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'message': self.message,
            'intent': self.intent.value if self.intent else None,
            'data': self.data,
            'suggestions': self.suggestions,
            'action_required': self.action_required,
            'timestamp': self.timestamp
        }


class IntentClassifier:
    """Classifies user intent from natural language"""
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.UPLOAD_DATA: [
                r'upload.*data', r'load.*file', r'import.*data', r'add.*dataset',
                r'new.*data', r'use.*data', r'work with.*data'
            ],
            IntentType.SHOW_DATA: [
                r'show.*data', r'display.*data', r'view.*data', r'see.*data',
                r'look at.*data', r'preview', r'head', r'first.*rows'
            ],
            IntentType.PROFILE_DATA: [
                r'profile', r'analyze.*data', r'check.*quality', r'data.*summary',
                r'statistics', r'overview', r'describe.*data'
            ],
            IntentType.CLEAN_DATA: [
                r'clean.*data', r'fix.*data', r'remove.*duplicate', r'handle.*missing',
                r'fix.*issue', r'prepare.*data', r'process.*data'
            ],
            IntentType.ANALYZE_QUALITY: [
                r'quality.*score', r'data.*quality', r'how.*good', r'quality.*check',
                r'assess.*quality'
            ],
            IntentType.FIND_ISSUES: [
                r'what.*wrong', r'find.*issue', r'what.*problem', r'detect.*issue',
                r'any.*error', r'check.*problem'
            ],
            IntentType.SHOW_STATISTICS: [
                r'statistics', r'stats', r'summary', r'describe', r'info',
                r'basic.*info'
            ],
            IntentType.BUILD_MODEL: [
                r'build.*model', r'train.*model', r'create.*model', r'machine.*learning',
                r'predict', r'ml.*model', r'ai.*model'
            ],
            IntentType.PREDICT: [
                r'predict', r'forecast', r'estimate', r'what.*will', r'future.*value'
            ],
            IntentType.EXPLAIN_MODEL: [
                r'explain.*model', r'how.*work', r'why.*predict', r'feature.*importance',
                r'understand.*model'
            ],
            IntentType.GET_HELP: [
                r'help', r'how.*do', r'what.*can', r'guide', r'tutorial',
                r'show.*me', r'teach.*me'
            ],
            IntentType.SUGGEST_NEXT: [
                r'what.*next', r'now.*what', r'suggest', r'what.*should',
                r'recommend', r'next.*step'
            ],
            IntentType.SAVE_WORKFLOW: [
                r'save.*workflow', r'save.*recipe', r'export.*workflow', r'store.*process'
            ],
            IntentType.LOAD_RECIPE: [
                r'load.*recipe', r'use.*recipe', r'apply.*recipe', r'reuse.*workflow'
            ],
            IntentType.UNDO_ACTION: [
                r'undo', r'revert', r'go.*back', r'cancel', r'rollback'
            ]
        }
    
    def classify(self, user_input: str) -> IntentType:
        """Classify user intent from input"""
        user_input = user_input.lower().strip()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input):
                    return intent
        
        return IntentType.UNKNOWN


class APIClient:
    """Handles API calls to various LLM providers"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.api_enabled = False
        self.api_provider = None
        self.api_model = None
        self.api_key = None
        self._initialize()
    
    def _initialize(self):
        """Initialize API client with configured provider"""
        if not self.config_manager:
            return
        
        try:
            best_api = self.config_manager.get_best_api()
            if best_api:
                self.api_enabled = True
                self.api_provider = best_api['provider']
                self.api_model = best_api['model']
                self.api_key = self.config_manager.get_api_key(self.api_provider)
        except Exception as e:
            print(f"Error initializing API: {e}")
            self.api_enabled = False
    
    def reload(self):
        """Reload API configuration"""
        self._initialize()
    
    def generate_response(self, prompt: str, context: Dict = None, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Generate AI response using configured API"""
        
        if not self.api_enabled or not self.api_key:
            return None
        
        try:
            if self.api_provider == "anthropic":
                return self._call_anthropic(prompt, context, temperature, max_tokens)
            elif self.api_provider == "openai":
                return self._call_openai(prompt, context, temperature, max_tokens)
            elif self.api_provider == "google_gemini":
                return self._call_gemini(prompt, context, temperature, max_tokens)
            else:
                return None
        except Exception as e:
            print(f"API call error: {e}")
            return None
    
    def _call_anthropic(self, prompt: str, context: Dict, temperature: float, max_tokens: int) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Build system message with context
            system_message = self._build_system_message(context)
            
            message = client.messages.create(
                model=self.api_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except ImportError:
            print("⚠️ anthropic library not installed. Run: pip install anthropic")
            return None
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None
    
    def _call_openai(self, prompt: str, context: Dict, temperature: float, max_tokens: int) -> str:
        """Call OpenAI GPT API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Build system message
            system_message = self._build_system_message(context)
            
            response = client.chat.completions.create(
                model=self.api_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
        except ImportError:
            print("⚠️ openai library not installed. Run: pip install openai")
            return None
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    def _call_gemini(self, prompt: str, context: Dict, temperature: float, max_tokens: int) -> str:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.api_model)
            
            # Build full prompt with context
            system_message = self._build_system_message(context)
            full_prompt = f"{system_message}\n\nUser: {prompt}"
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            return response.text
        except ImportError:
            print("⚠️ google-generativeai library not installed. Run: pip install google-generativeai")
            return None
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def _build_system_message(self, context: Dict) -> str:
        """Build system message with data context"""
        system_msg = """You are an expert data cleaning and analysis assistant. You help users:
- Understand their data quality issues
- Clean and prepare data for analysis
- Build predictive models
- Generate insights from data

Keep responses concise, friendly, and actionable. Use emojis appropriately."""

        if context:
            system_msg += "\n\n**Current Context:**"
            
            if context.get('data_loaded'):
                system_msg += f"\n- Dataset loaded: {context.get('rows', 0):,} rows, {context.get('columns', 0)} columns"
            
            if context.get('data_cleaned'):
                system_msg += "\n- Data has been cleaned"
            
            if context.get('profile_available'):
                system_msg += "\n- Data profile available"
                if context.get('quality_score'):
                    system_msg += f" (Quality Score: {context.get('quality_score')}/100)"
            
            if context.get('issues'):
                system_msg += f"\n- Known issues: {', '.join(context.get('issues', []))}"
            
            if context.get('model_trained'):
                system_msg += "\n- ML model has been trained"
        
        return system_msg


class ConversationalAgent:
    """Main chatbot agent with FULL API integration"""
    
    def __init__(self, pipeline=None, config_manager=None):
        self.pipeline = pipeline
        self.config_manager = config_manager
        self.classifier = IntentClassifier()
        self.conversation_history = []
        self.context = {
            'last_action': None,
            'awaiting_confirmation': False,
            'pending_operation': None,
            'data_loaded': False,
            'data_cleaned': False,
            'model_trained': False
        }
        
        # Initialize API client
        self.api_client = APIClient(config_manager)
        
        # Chatbot configuration
        chatbot_config = config_manager.get_chatbot_config() if config_manager else {}
        self.use_ai_for_unknown = chatbot_config.get('fallback_to_local', True)
        self.temperature = chatbot_config.get('temperature', 0.7)
        self.max_tokens = chatbot_config.get('max_tokens', 1000)
    
    def reload_api_config(self):
        """Reload API configuration (called when config changes)"""
        self.api_client.reload()
        
        if self.api_client.api_enabled:
            print(f"✅ API reloaded: {self.api_client.api_provider} ({self.api_client.api_model})")
        else:
            print("ℹ️ API disabled - using rule-based responses")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API configuration status"""
        return {
            'enabled': self.api_client.api_enabled,
            'provider': self.api_client.api_provider,
            'model': self.api_client.api_model
        }
    
    def _build_data_context(self) -> Dict[str, Any]:
        """Build context about current data state"""
        context = {
            'data_loaded': False,
            'data_cleaned': False,
            'profile_available': False,
            'model_trained': False,
            'rows': 0,
            'columns': 0,
            'issues': []
        }
        
        if self.pipeline and self.pipeline.data is not None:
            context['data_loaded'] = True
            context['rows'] = len(self.pipeline.data)
            context['columns'] = len(self.pipeline.data.columns)
            
            if self.pipeline.cleaned_data is not None:
                context['data_cleaned'] = True
            
            if self.pipeline.profile_result is not None:
                context['profile_available'] = True
                
                # Add quality score
                profile = self.pipeline.profile_result
                total_cells = context['rows'] * context['columns']
                missing_cells = profile['missing_data']['total_missing']
                quality_score = max(0, 100 - (missing_cells / total_cells * 100)) if total_cells > 0 else 0
                context['quality_score'] = int(quality_score)
                
                # Add issues
                if profile['missing_data']['total_missing'] > 0:
                    context['issues'].append(f"{profile['missing_data']['total_missing']} missing values")
                if profile['duplicates']['duplicate_rows'] > 0:
                    context['issues'].append(f"{profile['duplicates']['duplicate_rows']} duplicates")
                if profile['outliers']:
                    context['issues'].append(f"Outliers in {len(profile['outliers'])} columns")
            
            if self.pipeline.prediction_pipeline and self.pipeline.trained_models:
                context['model_trained'] = True
        
        return context
    
    def process_message(self, user_input: str) -> ChatbotResponse:
        """Process user message and return response"""
        
        # Store in history
        self.conversation_history.append({
            'role': 'user',
            'message': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Classify intent
        intent = self.classifier.classify(user_input)
        
        # Try rule-based response first
        response = self._handle_intent(intent, user_input)
        
        # If unknown intent and API is enabled, try AI response
        if intent == IntentType.UNKNOWN and self.api_client.api_enabled:
            ai_response = self._generate_ai_response(user_input)
            if ai_response:
                response = ChatbotResponse(
                    message=ai_response,
                    intent=intent,
                    suggestions=self._extract_suggestions_from_ai(ai_response)
                )
        
        # Store response in history
        self.conversation_history.append({
            'role': 'assistant',
            'message': response.message,
            'intent': intent.value,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_ai_response(self, user_input: str) -> Optional[str]:
        """Generate AI-powered response"""
        
        # Build context
        context = self._build_data_context()
        
        # Create enhanced prompt
        prompt = f"""User question: {user_input}

Please provide a helpful, concise response about their data cleaning task. 
If they're asking about their specific data, refer to the context provided.
Keep responses under 200 words and actionable."""

        # Get AI response
        ai_response = self.api_client.generate_response(
            prompt=prompt,
            context=context,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return ai_response
    
    def _extract_suggestions_from_ai(self, ai_response: str) -> List[str]:
        """Extract action suggestions from AI response"""
        suggestions = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for patterns like "1.", "•", "-", "*"
            if re.match(r'^[\d\-\*\•]+[\.\):]?\s+', line):
                # Clean up the line
                suggestion = re.sub(r'^[\d\-\*\•]+[\.\):]?\s+', '', line)
                if len(suggestion) > 10 and len(suggestion) < 100:
                    suggestions.append(suggestion)
        
        # Limit to 3 suggestions
        return suggestions[:3]
    
    def _handle_intent(self, intent: IntentType, user_input: str) -> ChatbotResponse:
        """Handle specific intent with rule-based responses"""
        
        handlers = {
            IntentType.UPLOAD_DATA: self._handle_upload_data,
            IntentType.SHOW_DATA: self._handle_show_data,
            IntentType.PROFILE_DATA: self._handle_profile_data,
            IntentType.ANALYZE_QUALITY: self._handle_analyze_quality,
            IntentType.FIND_ISSUES: self._handle_find_issues,
            IntentType.SHOW_STATISTICS: self._handle_show_statistics,
            IntentType.PREDICT: self._handle_predict,
            IntentType.GET_HELP: self._handle_get_help,
            IntentType.SUGGEST_NEXT: self._handle_suggest_next,
            IntentType.SAVE_WORKFLOW: self._handle_save_workflow,
            IntentType.UNDO_ACTION: self._handle_undo,
        }
        
        # Handlers that need user_input parameter
        if intent == IntentType.CLEAN_DATA:
            return self._handle_clean_data(user_input)
        elif intent == IntentType.BUILD_MODEL:
            return self._handle_build_model(user_input)
        elif intent in handlers:
            return handlers[intent]()
        else:
            return self._handle_unknown(user_input)
    
    def _handle_upload_data(self) -> ChatbotResponse:
        """Handle data upload request"""
        if self.pipeline and self.pipeline.data is not None:
            return ChatbotResponse(
                message=f"📊 You already have data loaded ({len(self.pipeline.data)} rows, {len(self.pipeline.data.columns)} columns).\n\n"
                        f"Would you like to:\n"
                        f"- View the data: 'show me the data'\n"
                        f"- Profile it: 'analyze the data'\n"
                        f"- Clean it: 'clean the data'\n"
                        f"- Upload new data: Use the upload button in the sidebar",
                suggestions=[
                    "Show me the data",
                    "Analyze the data quality",
                    "Clean the data"
                ]
            )
        else:
            return ChatbotResponse(
                message="📁 To upload data, please use the file upload button in the sidebar.\n\n"
                        "Supported formats:\n"
                        "- CSV (.csv)\n"
                        "- Excel (.xlsx)\n"
                        "- JSON (.json)\n"
                        "- Parquet (.parquet)\n\n"
                        "Once uploaded, I'll help you analyze and clean it!",
                action_required=True
            )
    
    def _handle_show_data(self) -> ChatbotResponse:
        """Handle show data request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded yet. Please upload a dataset first.\n\n"
                        "Use the upload button in the sidebar to get started!",
                suggestions=["How do I upload data?"]
            )
        
        data = self.pipeline.data
        preview = data.head(5)
        
        return ChatbotResponse(
            message=f"📊 **Your Data Overview:**\n\n"
                    f"- **Rows:** {len(data):,}\n"
                    f"- **Columns:** {len(data.columns)}\n"
                    f"- **Memory:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
                    f"**Column Names:**\n" + ", ".join(data.columns.tolist()) + "\n\n"
                    f"First 5 rows are displayed in the Data Preview tab.",
            data={'preview': preview.to_dict()},
            suggestions=[
                "Analyze data quality",
                "Clean the data",
                "What problems does it have?"
            ]
        )
    
    def _handle_profile_data(self) -> ChatbotResponse:
        """Handle data profiling request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload data first!",
                suggestions=["How do I upload data?"]
            )
        
        # Profile if not already done
        if self.pipeline.profile_result is None:
            self.pipeline.profile_data()
        
        profile = self.pipeline.profile_result
        
        # Calculate quality score
        total_cells = len(self.pipeline.data) * len(self.pipeline.data.columns)
        missing_cells = profile['missing_data']['total_missing']
        quality_score = max(0, 100 - (missing_cells / total_cells * 100)) if total_cells > 0 else 0
        
        message = f"📊 **Data Quality Report:**\n\n"
        message += f"**Quality Score:** {quality_score:.0f}/100 "
        
        if quality_score >= 90:
            message += "🟢 Excellent!\n\n"
        elif quality_score >= 75:
            message += "🔵 Good\n\n"
        elif quality_score >= 60:
            message += "🟡 Fair\n\n"
        else:
            message += "🔴 Needs Work\n\n"
        
        message += f"**Issues Found:**\n"
        message += f"- Missing values: {profile['missing_data']['total_missing']:,}\n"
        message += f"- Duplicate rows: {profile['duplicates']['duplicate_rows']}\n"
        message += f"- Outliers detected: {len(profile['outliers'])} columns\n"
        message += f"- Format issues: {len(profile['format_issues'])} columns\n\n"
        
        if quality_score < 85:
            message += "💡 **Recommendation:** Clean your data to improve quality!"
        else:
            message += "✅ **Recommendation:** Data looks good! Ready for analysis or ML."
        
        return ChatbotResponse(
            message=message,
            data=profile,
            suggestions=[
                "Clean the data",
                "Show me the issues",
                "What should I do next?"
            ]
        )
    
    def _handle_clean_data(self, user_input: str) -> ChatbotResponse:
        """Handle data cleaning request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload data first!",
                suggestions=["How do I upload data?"]
            )
        
        # Check for specific cleaning requests
        if 'automatic' in user_input.lower() or 'auto' in user_input.lower():
            return ChatbotResponse(
                message="🤖 **Automatic Cleaning Mode**\n\n"
                        "I can clean your data automatically! Here's what will happen:\n\n"
                        "1. ✅ Remove duplicates\n"
                        "2. ✅ Fill missing values (smart strategies)\n"
                        "3. ✅ Handle outliers\n"
                        "4. ✅ Fix data types\n"
                        "5. ✅ Normalize text\n\n"
                        "Go to the **Automatic** mode tab and click '🚀 Start Auto-Clean'\n\n"
                        "I'll explain every decision I make!",
                suggestions=[
                    "What's the quality score?",
                    "What issues will be fixed?",
                    "Show me a preview first"
                ]
            )
        
        return ChatbotResponse(
            message="🧹 **Let's Clean Your Data!**\n\n"
                    "I can help you in 3 ways:\n\n"
                    "**1. 🤖 Automatic (Recommended)**\n"
                    "   - I clean everything automatically\n"
                    "   - Fast and smart\n"
                    "   - Great for routine data\n\n"
                    "**2. 🤝 Assisted**\n"
                    "   - I suggest, you approve\n"
                    "   - Full transparency\n"
                    "   - Best for learning\n\n"
                    "**3. 🖐️ Manual**\n"
                    "   - You control everything\n"
                    "   - Maximum flexibility\n\n"
                    "Which would you prefer?",
            suggestions=[
                "Use automatic cleaning",
                "I want to review suggestions",
                "Show me what needs cleaning"
            ]
        )
    
    def _handle_analyze_quality(self) -> ChatbotResponse:
        """Handle quality analysis request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload data first!"
            )
        
        if self.pipeline.profile_result is None:
            self.pipeline.profile_data()
        
        profile = self.pipeline.profile_result
        total_cells = len(self.pipeline.data) * len(self.pipeline.data.columns)
        missing_pct = (profile['missing_data']['total_missing'] / total_cells * 100) if total_cells > 0 else 0
        dup_pct = profile['duplicates']['duplicate_percentage']
        
        quality_score = max(0, 100 - missing_pct - dup_pct)
        
        if quality_score >= 90:
            verdict = "🟢 **Excellent!** Your data is in great shape!"
        elif quality_score >= 75:
            verdict = "🔵 **Good!** Minor issues to address."
        elif quality_score >= 60:
            verdict = "🟡 **Fair.** Cleaning recommended."
        else:
            verdict = "🔴 **Poor.** Significant cleaning needed."
        
        return ChatbotResponse(
            message=f"📊 **Quality Score: {quality_score:.0f}/100**\n\n{verdict}\n\n"
                    f"**Breakdown:**\n"
                    f"- Completeness: {100-missing_pct:.0f}%\n"
                    f"- Uniqueness: {100-dup_pct:.0f}%\n"
                    f"- Format Issues: {len(profile['format_issues'])}\n"
                    f"- Outliers: {len(profile['outliers'])} columns affected",
            suggestions=[
                "How can I improve quality?",
                "Clean the data",
                "What are the main issues?"
            ]
        )
    
    def _handle_find_issues(self) -> ChatbotResponse:
        """Handle find issues request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload data first!"
            )
        
        if self.pipeline.profile_result is None:
            self.pipeline.profile_data()
        
        profile = self.pipeline.profile_result
        issues = []
        
        # Check for issues
        if profile['missing_data']['total_missing'] > 0:
            issues.append(f"🔴 **Missing Data:** {profile['missing_data']['total_missing']:,} missing values across {len(profile['missing_data']['by_column'])} columns")
        
        if profile['duplicates']['duplicate_rows'] > 0:
            issues.append(f"🟡 **Duplicates:** {profile['duplicates']['duplicate_rows']} duplicate rows ({profile['duplicates']['duplicate_percentage']:.1f}%)")
        
        if profile['outliers']:
            issues.append(f"🟠 **Outliers:** Detected in {len(profile['outliers'])} columns")
        
        if profile['format_issues']:
            issues.append(f"🟣 **Format Issues:** {len(profile['format_issues'])} columns have formatting problems")
        
        if not issues:
            return ChatbotResponse(
                message="✅ **Great news!** No major issues detected in your data.\n\n"
                        "Your data appears to be clean and ready for analysis or machine learning!",
                suggestions=[
                    "Build a prediction model",
                    "Show me statistics",
                    "What can I do with this data?"
                ]
            )
        
        message = f"🔍 **Found {len(issues)} Issues:**\n\n" + "\n\n".join(issues)
        message += "\n\n💡 **Recommendation:** Use automatic cleaning to fix these issues quickly!"
        
        return ChatbotResponse(
            message=message,
            suggestions=[
                "Clean the data automatically",
                "Show me more details",
                "How do I fix these?"
            ]
        )
    
    def _handle_show_statistics(self) -> ChatbotResponse:
        """Handle statistics request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload data first!"
            )
        
        data = self.pipeline.data
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        message = f"📈 **Data Statistics:**\n\n"
        message += f"**Overall:**\n"
        message += f"- Total Rows: {len(data):,}\n"
        message += f"- Total Columns: {len(data.columns)}\n"
        message += f"- Numeric Columns: {len(numeric_cols)}\n"
        message += f"- Text Columns: {len(data.select_dtypes(include=['object']).columns)}\n\n"
        
        if len(numeric_cols) > 0:
            message += f"**Numeric Summary (first numeric column):**\n"
            col = numeric_cols[0]
            message += f"- Column: {col}\n"
            message += f"- Mean: {data[col].mean():.2f}\n"
            message += f"- Median: {data[col].median():.2f}\n"
            message += f"- Min: {data[col].min():.2f}\n"
            message += f"- Max: {data[col].max():.2f}\n\n"
        
        message += "Check the Profile tab for detailed visualizations!"
        
        return ChatbotResponse(
            message=message,
            suggestions=[
                "Show more details",
                "Visualize the data",
                "What should I do next?"
            ]
        )
    
    def _handle_build_model(self, user_input: str) -> ChatbotResponse:
        """Handle model building request"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="❌ No data loaded. Upload and clean data first!"
            )
        
        data_to_use = self.pipeline.cleaned_data if self.pipeline.cleaned_data is not None else self.pipeline.data
        
        # Check if data is clean enough
        if self.pipeline.profile_result:
            total_cells = len(data_to_use) * len(data_to_use.columns)
            missing_pct = (self.pipeline.profile_result['missing_data']['total_missing'] / total_cells * 100) if total_cells > 0 else 0
            
            if missing_pct > 15:
                return ChatbotResponse(
                    message=f"⚠️ **Data Quality Warning**\n\n"
                            f"Your data has {missing_pct:.1f}% missing values.\n\n"
                            f"For best ML results, I recommend cleaning the data first.\n\n"
                            f"Would you like me to clean it automatically?",
                    suggestions=[
                        "Yes, clean it first",
                        "Build model anyway",
                        "Show me what's missing"
                    ]
                )
        
        return ChatbotResponse(
            message="🤖 **Let's Build a Prediction Model!**\n\n"
                    "To build a model, I need to know:\n\n"
                    "1. **What do you want to predict?** (target column)\n"
                    "   - Example: 'price', 'category', 'outcome'\n\n"
                    "2. **What data should I use?** (feature columns)\n"
                    "   - I can suggest the best ones!\n\n"
                    "Go to the **Predictions** tab to:\n"
                    "- Select your target\n"
                    "- Choose features\n"
                    "- Get AI recommendations\n"
                    "- Train and compare models\n\n"
                    "I'll explain everything along the way!",
            suggestions=[
                "What can I predict with this data?",
                "Recommend a target column",
                "What's the best model to use?"
            ]
        )
    
    def _handle_predict(self) -> ChatbotResponse:
        """Handle prediction request"""
        return ChatbotResponse(
            message="🔮 **Making Predictions**\n\n"
                    "To make predictions, you'll need:\n\n"
                    "1. ✅ A trained model\n"
                    "2. ✅ New data to predict on\n\n"
                    "If you haven't trained a model yet:\n"
                    "- Go to the Predictions tab\n"
                    "- Build and train a model first\n\n"
                    "Then you can use it to predict on new data!",
            suggestions=[
                "Build a model first",
                "How do I train a model?",
                "What can I predict?"
            ]
        )
    
    def _handle_get_help(self) -> ChatbotResponse:
        """Handle help request"""
        api_status = " (AI-powered)" if self.api_client.api_enabled else ""
        
        return ChatbotResponse(
            message=f"👋 **I'm here to help!{api_status}**\n\n"
                    "I can assist you with:\n\n"
                    "**📊 Data Operations:**\n"
                    "- Upload and view data\n"
                    "- Profile and analyze quality\n"
                    "- Clean and fix issues\n\n"
                    "**🤖 Machine Learning:**\n"
                    "- Build prediction models\n"
                    "- Train and compare models\n"
                    "- Make predictions\n\n"
                    "**⚡ Advanced Features:**\n"
                    "- Undo/redo changes\n"
                    "- Generate Python code\n"
                    "- Save and reuse workflows\n\n"
                    "**Just ask me anything!** I understand natural language.\n\n"
                    "Try:\n"
                    "- 'What's wrong with my data?'\n"
                    "- 'Clean my data automatically'\n"
                    "- 'Build a prediction model'\n"
                    "- 'What should I do next?'",
            suggestions=[
                "Show me the data",
                "Analyze data quality",
                "What can I do with this?",
                "Guide me through the process"
            ]
        )
    
    def _handle_suggest_next(self) -> ChatbotResponse:
        """Handle next steps suggestion"""
        if not self.pipeline or self.pipeline.data is None:
            return ChatbotResponse(
                message="**📁 Step 1: Upload Data**\n\nStart by uploading your dataset using the file upload button in the sidebar.",
                suggestions=["How do I upload data?"]
            )
        
        if self.pipeline.profile_result is None:
            return ChatbotResponse(
                message="**📊 Step 2: Analyze Your Data**\n\nLet's profile your data to understand its quality and identify issues.",
                suggestions=["Analyze my data", "Check data quality"]
            )
        
        if self.pipeline.cleaned_data is None:
            return ChatbotResponse(
                message="**🧹 Step 3: Clean Your Data**\n\nYour data has some issues. I recommend cleaning it for better results!",
                suggestions=["Clean automatically", "Show me the issues"]
            )
        
        return ChatbotResponse(
            message="**🚀 Step 4: Build Models or Export**\n\n"
                    "Your data is clean! You can now:\n"
                    "- Build ML models for predictions\n"
                    "- Export the cleaned data\n"
                    "- Generate Python code\n"
                    "- Save your workflow as a recipe",
            suggestions=[
                "Build a prediction model",
                "Export cleaned data",
                "Generate Python code",
                "Save as recipe"
            ]
        )
    
    def _handle_save_workflow(self) -> ChatbotResponse:
        """Handle save workflow request"""
        return ChatbotResponse(
            message="💾 **Save Your Workflow**\n\n"
                    "You can save your cleaning workflow as a recipe to reuse later!\n\n"
                    "Go to: **Power Tools → Recipes → Save Recipe**\n\n"
                    "Give it a name like:\n"
                    "- 'Weekly Sales Cleaning'\n"
                    "- 'Customer Data Standard'\n"
                    "- 'Monthly Report Prep'\n\n"
                    "Next time, just load the recipe and apply it in seconds!",
            suggestions=[
                "How do I load a recipe?",
                "What are industry templates?",
                "Export as Python code"
            ]
        )
    
    def _handle_undo(self) -> ChatbotResponse:
        """Handle undo request"""
        if self.pipeline and hasattr(self.pipeline, 'snapshot_manager') and self.pipeline.snapshot_manager.can_undo():
            return ChatbotResponse(
                message="⏪ **Undo Available!**\n\n"
                        "Go to: **Power Tools → Undo/Redo**\n\n"
                        "You can:\n"
                        "- Undo your last action\n"
                        "- Jump to any previous version\n"
                        "- See your complete history\n\n"
                        "Your data is safe - all changes are tracked!",
                suggestions=["Show version history"]
            )
        else:
            return ChatbotResponse(
                message="No actions to undo yet. Make some changes first!",
                suggestions=["What can I do?"]
            )
    
    def _handle_unknown(self, user_input: str) -> ChatbotResponse:
        """Handle unknown intent"""
        return ChatbotResponse(
            message="🤔 I'm not sure I understood that.\n\n"
                    "You can ask me about:\n"
                    "- 'Show me my data'\n"
                    "- 'What's wrong with my data?'\n"
                    "- 'Clean my data'\n"
                    "- 'Build a model'\n"
                    "- 'Help me get started'\n\n"
                    "Or try: 'What should I do next?'",
            suggestions=[
                "Help me",
                "What can you do?",
                "Show me my data",
                "What should I do next?"
            ]
        )
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.context = {
            'last_action': None,
            'awaiting_confirmation': False,
            'pending_operation': None,
            'data_loaded': False,
            'data_cleaned': False,
            'model_trained': False
        }
    
    def export_conversation(self, format: str = 'json') -> str:
        """Export conversation history"""
        if format == 'json':
            return json.dumps(self.conversation_history, indent=2)
        elif format == 'text':
            text = []
            for msg in self.conversation_history:
                role = msg['role'].upper()
                text.append(f"{role}: {msg['message']}\n")
            return "\n".join(text)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Utility functions for integration
def create_chatbot(pipeline=None, config_manager=None) -> ConversationalAgent:
    """Factory function to create chatbot instance"""
    return ConversationalAgent(pipeline=pipeline, config_manager=config_manager)


def format_response_for_display(response: ChatbotResponse) -> str:
    """Format response for display in UI"""
    output = response.message
    
    if response.suggestions:
        output += "\n\n**Suggested Actions:**\n"
        for i, suggestion in enumerate(response.suggestions, 1):
            output += f"{i}. {suggestion}\n"
    
    return output