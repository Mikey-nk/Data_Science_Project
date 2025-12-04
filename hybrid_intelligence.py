"""
Hybrid Intelligence Module
Adds automatic cleaning with explanations and approval system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import json


class CleaningMode(Enum):
    """Available cleaning modes"""
    MANUAL = "manual"
    ASSISTED = "assisted"
    AUTOMATIC = "automatic"


class RiskLevel(Enum):
    """Risk levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CleaningRule:
    """Represents a single cleaning rule with metadata"""
    column: str
    operation: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float  # 0-1
    risk_level: RiskLevel
    impact_description: str
    alternatives: List[Dict[str, str]]
    expected_changes: int
    
    def to_dict(self):
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        return data


@dataclass
class OperationExplanation:
    """Detailed explanation for a cleaning operation"""
    title: str
    issue_detected: str
    decision_made: str
    reasoning: List[str]
    impact: str
    risk_warning: Optional[str]
    alternatives_considered: List[str]
    confidence_score: float
    
    def to_markdown(self) -> str:
        """Convert to markdown for display"""
        md = f"### {self.title}\n\n"
        md += f"**🔍 Issue Detected:** {self.issue_detected}\n\n"
        md += f"**🤖 Decision:** {self.decision_made}\n\n"
        md += f"**📖 Reasoning:**\n"
        for reason in self.reasoning:
            md += f"- {reason}\n"
        md += f"\n**⚠️ Impact:** {self.impact}\n\n"
        
        if self.risk_warning:
            md += f"**⚠️ Risk Warning:** {self.risk_warning}\n\n"
        
        if self.alternatives_considered:
            md += f"**💡 Alternatives Considered:**\n"
            for alt in self.alternatives_considered:
                md += f"- {alt}\n"
        
        md += f"\n**Confidence Score:** {self.confidence_score:.0%}\n"
        return md


class IntelligentRuleGenerator:
    """Generates cleaning rules with explanations"""
    
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.df = df
        self.profile = profile
        self.rules = []
        self.explanations = []
    
    def generate_all_rules(self) -> Tuple[List[CleaningRule], List[OperationExplanation]]:
        """Generate all cleaning rules with explanations"""
        self.rules = []
        self.explanations = []
        
        # 1. Handle duplicates
        if self.profile['duplicates']['duplicate_rows'] > 0:
            self._generate_duplicate_rule()
        
        # 2. Handle missing values
        for col, info in self.profile['missing_data']['by_column'].items():
            self._generate_missing_value_rule(col, info)
        
        # 3. Handle type conversions
        self._generate_type_conversion_rules()
        
        # 4. Handle outliers
        for col, info in self.profile['outliers'].items():
            self._generate_outlier_rule(col, info)
        
        # 5. Handle format issues
        for col, issues in self.profile['format_issues'].items():
            self._generate_format_fix_rule(col, issues)
        
        return self.rules, self.explanations
    
    def _generate_duplicate_rule(self):
        """Generate rule for handling duplicates"""
        dup_count = self.profile['duplicates']['duplicate_rows']
        dup_pct = self.profile['duplicates']['duplicate_percentage']
        
        reasoning = [
            f"Found {dup_count} duplicate rows ({dup_pct}%)",
            "Duplicate rows provide no additional information",
            "Keeping duplicates can skew analysis and inflate counts",
            "Removal is a standard best practice"
        ]
        
        alternatives = [
            {"option": "Keep duplicates", "reason": "If duplicates represent valid repeated events"},
            {"option": "Mark duplicates", "reason": "Add a flag column instead of removing"}
        ]
        
        rule = CleaningRule(
            column="ALL",
            operation="remove_duplicates",
            parameters={},
            reasoning=f"Duplicate rows waste memory and can distort analysis. {dup_count} exact duplicates found.",
            confidence=0.95,
            risk_level=RiskLevel.LOW,
            impact_description=f"Will remove {dup_count} rows, reducing dataset size by {dup_pct}%",
            alternatives=alternatives,
            expected_changes=dup_count
        )
        
        explanation = OperationExplanation(
            title="Remove Duplicate Rows",
            issue_detected=f"{dup_count} duplicate rows found ({dup_pct}% of data)",
            decision_made="Remove all duplicate rows",
            reasoning=reasoning,
            impact=f"{dup_count} rows will be removed",
            risk_warning="Low risk - duplicates are exact copies" if dup_pct < 5 else "Medium risk - large percentage of data",
            alternatives_considered=[alt['option'] + ': ' + alt['reason'] for alt in alternatives],
            confidence_score=0.95
        )
        
        self.rules.append(rule)
        self.explanations.append(explanation)
    
    def _generate_missing_value_rule(self, col: str, info: Dict):
        """Generate rule for handling missing values"""
        missing_count = info['count']
        missing_pct = info['percentage']
        
        # Detect column context
        col_lower = col.lower()
        dtype = str(self.df[col].dtype)
        
        # Intelligent strategy selection
        strategy, reasoning, confidence, risk = self._select_missing_strategy(
            col, col_lower, dtype, missing_count, missing_pct
        )
        
        alternatives = self._get_missing_alternatives(col, dtype, strategy)
        
        rule = CleaningRule(
            column=col,
            operation="handle_missing",
            parameters={"method": strategy},
            reasoning=reasoning,
            confidence=confidence,
            risk_level=risk,
            impact_description=f"Will modify {missing_count} rows ({missing_pct}%)",
            alternatives=alternatives,
            expected_changes=missing_count
        )
        
        explanation = self._create_missing_explanation(
            col, missing_count, missing_pct, strategy, reasoning, confidence, alternatives
        )
        
        self.rules.append(rule)
        self.explanations.append(explanation)
    
    def _select_missing_strategy(self, col: str, col_lower: str, dtype: str, 
                                  count: int, pct: float) -> Tuple[str, str, float, RiskLevel]:
        """Intelligently select the best missing value strategy"""
        
        # High percentage missing - consider dropping
        if pct > 50:
            return (
                "drop",
                f"Column has {pct}% missing values. More than half the data is missing, making imputation unreliable.",
                0.85,
                RiskLevel.MEDIUM
            )
        
        # ID columns - forward fill or drop
        if any(x in col_lower for x in ['id', '_id', 'key', 'code']):
            return (
                "drop",
                "ID columns should not be imputed as each ID should be unique. Dropping rows with missing IDs.",
                0.90,
                RiskLevel.LOW
            )
        
        # Numeric columns
        if 'int' in dtype or 'float' in dtype:
            col_data = self.df[col].dropna()
            
            # Check for outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum()
            
            if outliers > len(col_data) * 0.1:  # More than 10% outliers
                return (
                    "median",
                    f"Using MEDIAN because column has {outliers} outliers. Median is robust to outliers, mean would be skewed.",
                    0.85,
                    RiskLevel.LOW
                )
            else:
                return (
                    "mean",
                    "Using MEAN for numeric data. Distribution appears relatively normal without significant outliers.",
                    0.80,
                    RiskLevel.LOW
                )
        
        # Date columns
        if 'date' in dtype.lower() or 'datetime' in dtype.lower():
            return (
                "forward_fill",
                "Using FORWARD FILL for date/time data. Assumes temporal continuity (e.g., dates carry forward).",
                0.75,
                RiskLevel.MEDIUM
            )
        
        # Categorical/text columns
        if dtype == 'object' or 'category' in dtype:
            unique_ratio = self.df[col].nunique() / len(self.df[col].dropna())
            
            if unique_ratio < 0.1:  # Low cardinality - use mode
                return (
                    "mode",
                    f"Using MODE (most common value) for categorical data with {self.df[col].nunique()} unique values.",
                    0.85,
                    RiskLevel.LOW
                )
            elif unique_ratio > 0.5:  # High cardinality - use constant
                return (
                    "constant:UNKNOWN",
                    "Using 'UNKNOWN' placeholder for high-cardinality text field. Too many unique values to use mode.",
                    0.70,
                    RiskLevel.MEDIUM
                )
            else:
                return (
                    "mode",
                    "Using MODE for categorical data. Most frequent value is a reasonable default.",
                    0.75,
                    RiskLevel.MEDIUM
                )
        
        # Default fallback
        return (
            "drop",
            "Dropping rows with missing values as safe default when data type is unclear.",
            0.60,
            RiskLevel.MEDIUM
        )
    
    def _get_missing_alternatives(self, col: str, dtype: str, chosen: str) -> List[Dict]:
        """Get alternative strategies for missing values"""
        alternatives = []
        
        if 'int' in dtype or 'float' in dtype:
            if chosen != "mean":
                alternatives.append({"option": "mean", "reason": "Simple average of all values"})
            if chosen != "median":
                alternatives.append({"option": "median", "reason": "Middle value, robust to outliers"})
            if chosen != "forward_fill":
                alternatives.append({"option": "forward_fill", "reason": "Carry forward last known value"})
        
        if chosen != "mode":
            alternatives.append({"option": "mode", "reason": "Use most common value"})
        
        if chosen != "drop":
            alternatives.append({"option": "drop", "reason": "Remove rows with missing values"})
        
        alternatives.append({"option": "constant", "reason": "Fill with custom placeholder value"})
        
        return alternatives
    
    def _create_missing_explanation(self, col: str, count: int, pct: float, 
                                     strategy: str, reasoning: str, confidence: float,
                                     alternatives: List[Dict]) -> OperationExplanation:
        """Create explanation for missing value handling"""
        
        strategy_name = strategy.split(':')[0].upper()
        
        return OperationExplanation(
            title=f"Handle Missing Values: '{col}'",
            issue_detected=f"Column has {count} missing values ({pct:.1f}% of data)",
            decision_made=f"Fill with {strategy_name}",
            reasoning=[reasoning],
            impact=f"{count} missing values will be filled",
            risk_warning="Medium risk - imputation introduces assumptions" if pct > 20 else None,
            alternatives_considered=[alt['option'] + ': ' + alt['reason'] for alt in alternatives],
            confidence_score=confidence
        )
    
    def _generate_type_conversion_rules(self):
        """Generate rules for type conversions"""
        for col, issues in self.profile.get('format_issues', {}).items():
            if "Numeric data stored as text" in issues:
                reasoning = [
                    f"Column '{col}' contains numeric values stored as text",
                    "Converting to numeric enables mathematical operations",
                    "Improves memory efficiency",
                    "Prevents type errors in calculations"
                ]
                
                rule = CleaningRule(
                    column=col,
                    operation="type_conversion",
                    parameters={"target_type": "numeric"},
                    reasoning="Numeric data stored as text should be converted for proper analysis.",
                    confidence=0.90,
                    risk_level=RiskLevel.LOW,
                    impact_description=f"Convert '{col}' from text to numeric type",
                    alternatives=[
                        {"option": "Keep as text", "reason": "If leading zeros are important (e.g., ZIP codes)"}
                    ],
                    expected_changes=len(self.df)
                )
                
                explanation = OperationExplanation(
                    title=f"Convert '{col}' to Numeric",
                    issue_detected="Numeric values stored as text",
                    decision_made="Convert to numeric type",
                    reasoning=reasoning,
                    impact="All values in column will be converted",
                    risk_warning=None,
                    alternatives_considered=["Keep as text if leading zeros matter"],
                    confidence_score=0.90
                )
                
                self.rules.append(rule)
                self.explanations.append(explanation)
            
            if "Possible date/time data" in issues:
                rule = CleaningRule(
                    column=col,
                    operation="type_conversion",
                    parameters={"target_type": "datetime"},
                    reasoning="Column appears to contain date/time values. Converting enables temporal analysis.",
                    confidence=0.75,
                    risk_level=RiskLevel.MEDIUM,
                    impact_description=f"Convert '{col}' to datetime type",
                    alternatives=[
                        {"option": "Keep as text", "reason": "If dates are in inconsistent formats"}
                    ],
                    expected_changes=len(self.df)
                )
                
                explanation = OperationExplanation(
                    title=f"Convert '{col}' to DateTime",
                    issue_detected="Column contains date-like patterns",
                    decision_made="Convert to datetime type",
                    reasoning=[
                        "Enables date arithmetic and filtering",
                        "Allows time-series analysis",
                        "Improves data validation"
                    ],
                    impact="Date parsing will be attempted on all values",
                    risk_warning="Medium risk - may fail if formats are inconsistent",
                    alternatives_considered=["Keep as text if multiple date formats present"],
                    confidence_score=0.75
                )
                
                self.rules.append(rule)
                self.explanations.append(explanation)
            
            if "Contains leading/trailing whitespace" in issues:
                rule = CleaningRule(
                    column=col,
                    operation="normalize_text",
                    parameters={},
                    reasoning="Whitespace can cause matching issues and inconsistencies.",
                    confidence=0.95,
                    risk_level=RiskLevel.LOW,
                    impact_description=f"Trim whitespace from '{col}'",
                    alternatives=[],
                    expected_changes=0  # Won't know exact count without checking
                )
                
                explanation = OperationExplanation(
                    title=f"Normalize Text: '{col}'",
                    issue_detected="Contains leading/trailing whitespace",
                    decision_made="Strip whitespace and normalize spacing",
                    reasoning=[
                        "Prevents matching errors",
                        "Standardizes data format",
                        "Improves data quality"
                    ],
                    impact="Text will be trimmed and spaces normalized",
                    risk_warning=None,
                    alternatives_considered=[],
                    confidence_score=0.95
                )
                
                self.rules.append(rule)
                self.explanations.append(explanation)
    
    def _generate_outlier_rule(self, col: str, info: Dict):
        """Generate rule for handling outliers"""
        count = info['count']
        pct = info['percentage']
        lower = info['lower_bound']
        upper = info['upper_bound']
        
        # Decide strategy based on percentage
        if pct > 10:
            strategy = "cap"
            reasoning = f"Capping {count} outliers ({pct}%) to reasonable bounds. Too many to remove without losing significant data."
            risk = RiskLevel.MEDIUM
            confidence = 0.75
        else:
            strategy = "cap"  # Generally safer than removing
            reasoning = f"Capping {count} outliers ({pct}%) to [{lower:.2f}, {upper:.2f}]. Preserves data while limiting extreme values."
            risk = RiskLevel.MEDIUM
            confidence = 0.80
        
        rule = CleaningRule(
            column=col,
            operation="handle_outliers",
            parameters={"method": strategy},
            reasoning=reasoning,
            confidence=confidence,
            risk_level=risk,
            impact_description=f"Will modify {count} outlier values",
            alternatives=[
                {"option": "remove", "reason": "Delete outlier rows entirely"},
                {"option": "keep", "reason": "Leave outliers unchanged if they're valid"}
            ],
            expected_changes=count
        )
        
        explanation = OperationExplanation(
            title=f"Handle Outliers: '{col}'",
            issue_detected=f"{count} outliers detected ({pct}%) using IQR method",
            decision_made=f"{strategy.upper()} outliers to [{lower:.2f}, {upper:.2f}]",
            reasoning=[
                reasoning,
                "IQR method: values beyond 1.5 × IQR are considered outliers",
                "Capping preserves data points while limiting extreme influence"
            ],
            impact=f"{count} values will be capped to boundary values",
            risk_warning="Medium risk - verify outliers aren't valid extreme values",
            alternatives_considered=[
                "Remove outliers: Would lose data points",
                "Keep outliers: May skew analysis"
            ],
            confidence_score=confidence
        )
        
        self.rules.append(rule)
        self.explanations.append(explanation)
    
    def _generate_format_fix_rule(self, col: str, issues: List[str]):
        """Already handled in type conversion rules"""
        pass


class ExplanationEngine:
    """Generates human-readable explanations for cleaning operations"""
    
    @staticmethod
    def explain_impact(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                       operation: str) -> str:
        """Explain the impact of an operation"""
        rows_before = len(original_df)
        rows_after = len(cleaned_df)
        
        if rows_before != rows_after:
            diff = rows_before - rows_after
            pct = (diff / rows_before * 100)
            return f"Removed {diff} rows ({pct:.1f}%). Dataset now has {rows_after} rows."
        else:
            return f"Modified data in-place. Row count unchanged ({rows_after} rows)."
    
    @staticmethod
    def explain_rule_choice(rule: CleaningRule) -> str:
        """Explain why a rule was chosen"""
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        emoji = risk_emoji.get(rule.risk_level.value, "⚪")
        
        explanation = f"{emoji} **{rule.operation.replace('_', ' ').title()}** on '{rule.column}'\n\n"
        explanation += f"**Reasoning:** {rule.reasoning}\n\n"
        explanation += f"**Confidence:** {rule.confidence:.0%}\n"
        explanation += f"**Risk Level:** {rule.risk_level.value.upper()}\n"
        explanation += f"**Impact:** {rule.impact_description}\n"
        
        if rule.alternatives:
            explanation += f"\n**Alternatives:**\n"
            for alt in rule.alternatives:
                explanation += f"- {alt['option']}: {alt['reason']}\n"
        
        return explanation
    
    @staticmethod
    def generate_summary(rules: List[CleaningRule]) -> str:
        """Generate overall summary of cleaning plan"""
        total_ops = len(rules)
        low_risk = sum(1 for r in rules if r.risk_level == RiskLevel.LOW)
        medium_risk = sum(1 for r in rules if r.risk_level == RiskLevel.MEDIUM)
        high_risk = sum(1 for r in rules if r.risk_level == RiskLevel.HIGH)
        avg_confidence = np.mean([r.confidence for r in rules]) if rules else 0
        
        summary = "## 🤖 Cleaning Plan Summary\n\n"
        summary += f"**Total Operations:** {total_ops}\n\n"
        summary += f"**Risk Breakdown:**\n"
        summary += f"- 🟢 Low Risk: {low_risk} operations\n"
        summary += f"- 🟡 Medium Risk: {medium_risk} operations\n"
        summary += f"- 🔴 High Risk: {high_risk} operations\n\n"
        summary += f"**Average Confidence:** {avg_confidence:.0%}\n\n"
        
        if high_risk > 0:
            summary += "⚠️ **Warning:** High-risk operations require careful review\n\n"
        
        return summary


class ApprovalManager:
    """Manages user approval for cleaning operations"""
    
    def __init__(self):
        self.approved_rules = []
        self.rejected_rules = []
        self.modified_rules = []
        self.pending_rules = []
    
    def add_pending_rule(self, rule: CleaningRule):
        """Add a rule that needs approval"""
        self.pending_rules.append(rule)
    
    def approve_rule(self, rule: CleaningRule):
        """Approve a rule for execution"""
        self.approved_rules.append(rule)
        if rule in self.pending_rules:
            self.pending_rules.remove(rule)
    
    def reject_rule(self, rule: CleaningRule):
        """Reject a rule"""
        self.rejected_rules.append(rule)
        if rule in self.pending_rules:
            self.pending_rules.remove(rule)
    
    def modify_rule(self, original_rule: CleaningRule, modified_rule: CleaningRule):
        """Accept a modified version of a rule"""
        self.modified_rules.append({
            'original': original_rule,
            'modified': modified_rule
        })
        self.approved_rules.append(modified_rule)
        if original_rule in self.pending_rules:
            self.pending_rules.remove(original_rule)
    
    def auto_approve_safe_operations(self, rules: List[CleaningRule]):
        """Automatically approve low-risk operations"""
        for rule in rules:
            if rule.risk_level == RiskLevel.LOW and rule.confidence > 0.85:
                self.approve_rule(rule)
            else:
                self.add_pending_rule(rule)
    
    def get_approved_rules(self) -> List[CleaningRule]:
        """Get all approved rules"""
        return self.approved_rules
    
    def get_pending_rules(self) -> List[CleaningRule]:
        """Get rules awaiting approval"""
        return self.pending_rules
    
    def get_status_summary(self) -> Dict[str, int]:
        """Get summary of approval status"""
        return {
            'approved': len(self.approved_rules),
            'rejected': len(self.rejected_rules),
            'modified': len(self.modified_rules),
            'pending': len(self.pending_rules)
        }


class ProgressNarrator:
    """Narrates the cleaning process in real-time"""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
    
    def start_operation(self, operation_name: str, description: str):
        """Start narrating an operation"""
        self.current_step += 1
        step = {
            'step': self.current_step,
            'operation': operation_name,
            'description': description,
            'status': 'in_progress',
            'details': []
        }
        self.steps.append(step)
        return step
    
    def add_detail(self, detail: str):
        """Add detail to current operation"""
        if self.steps:
            self.steps[-1]['details'].append(detail)
    
    def complete_operation(self, result: str):
        """Mark operation as complete"""
        if self.steps:
            self.steps[-1]['status'] = 'complete'
            self.steps[-1]['result'] = result
    
    def fail_operation(self, error: str):
        """Mark operation as failed"""
        if self.steps:
            self.steps[-1]['status'] = 'failed'
            self.steps[-1]['error'] = error
    
    def get_narrative(self) -> List[Dict]:
        """Get full narrative"""
        return self.steps
    
    def format_for_display(self) -> str:
        """Format narrative for display"""
        output = "## 🤖 Cleaning Progress\n\n"
        
        for step in self.steps:
            status_emoji = {
                'in_progress': '⏳',
                'complete': '✅',
                'failed': '❌'
            }
            emoji = status_emoji.get(step['status'], '⚪')
            
            output += f"{emoji} **Step {step['step']}: {step['operation']}**\n"
            output += f"   {step['description']}\n"
            
            for detail in step.get('details', []):
                output += f"   • {detail}\n"
            
            if step['status'] == 'complete':
                output += f"   ✓ {step.get('result', 'Complete')}\n"
            elif step['status'] == 'failed':
                output += f"   ✗ Error: {step.get('error', 'Unknown error')}\n"
            
            output += "\n"
        
        return output