"""
Chatbot Interface Module
Conversational AI assistant for data cleaning and ML
"""

import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


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


class ConversationalAgent:
    """Main chatbot agent"""
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
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
        
        # Generate response based on intent
        response = self._handle_intent(intent, user_input)
        
        # Store response in history
        self.conversation_history.append({
            'role': 'assistant',
            'message': response.message,
            'intent': intent.value,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _handle_intent(self, intent: IntentType, user_input: str) -> ChatbotResponse:
        """Handle specific intent"""
        
        if intent == IntentType.UPLOAD_DATA:
            return self._handle_upload_data()
        
        elif intent == IntentType.SHOW_DATA:
            return self._handle_show_data()
        
        elif intent == IntentType.PROFILE_DATA:
            return self._handle_profile_data()
        
        elif intent == IntentType.CLEAN_DATA:
            return self._handle_clean_data(user_input)
        
        elif intent == IntentType.ANALYZE_QUALITY:
            return self._handle_analyze_quality()
        
        elif intent == IntentType.FIND_ISSUES:
            return self._handle_find_issues()
        
        elif intent == IntentType.SHOW_STATISTICS:
            return self._handle_show_statistics()
        
        elif intent == IntentType.BUILD_MODEL:
            return self._handle_build_model(user_input)
        
        elif intent == IntentType.PREDICT:
            return self._handle_predict()
        
        elif intent == IntentType.GET_HELP:
            return self._handle_get_help()
        
        elif intent == IntentType.SUGGEST_NEXT:
            return self._handle_suggest_next()
        
        elif intent == IntentType.SAVE_WORKFLOW:
            return self._handle_save_workflow()
        
        elif intent == IntentType.UNDO_ACTION:
            return self._handle_undo()
        
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
        quality_score = max(0, 100 - (missing_cells / total_cells * 100))
        
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
        missing_pct = (profile['missing_data']['total_missing'] / total_cells * 100)
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
        return ChatbotResponse(
            message="👋 **I'm here to help!**\n\n"
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
                message="**🔍 Step 2: Analyze Your Data**\n\nLet's profile your data to understand its quality and identify issues.",
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
        if self.pipeline and self.pipeline.snapshot_manager.can_undo():
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