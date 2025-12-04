"""
Visual Insights Module
Provides comprehensive data visualization and quality dashboards
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import streamlit as st


class DataQualityVisualizer:
    """Creates visualizations for data quality analysis"""
    
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.df = df
        self.profile = profile
        self.color_scheme = {
            'primary': '#1f77b4',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db'
        }
    
    def create_missing_data_chart(self) -> go.Figure:
        """Create bar chart for missing data"""
        missing_data = self.profile['missing_data']['by_column']
        
        if not missing_data:
            # No missing data - show success message
            fig = go.Figure()
            fig.add_annotation(
                text="🎉 No Missing Data Found!<br>Your dataset is complete",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=self.color_scheme['success'])
            )
            fig.update_layout(
                title="Missing Data Analysis",
                height=400,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
            return fig
        
        # Extract data for plotting
        columns = list(missing_data.keys())
        counts = [missing_data[col]['count'] for col in columns]
        percentages = [missing_data[col]['percentage'] for col in columns]
        
        # Create bar chart
        fig = go.Figure()
        
        # Color bars based on severity
        colors = []
        for pct in percentages:
            if pct < 5:
                colors.append(self.color_scheme['success'])
            elif pct < 20:
                colors.append(self.color_scheme['warning'])
            else:
                colors.append(self.color_scheme['danger'])
        
        fig.add_trace(go.Bar(
            x=columns,
            y=counts,
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(counts, percentages)],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Missing: %{y}<br>Percentage: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Column",
            yaxis_title="Number of Missing Values",
            height=400,
            showlegend=False,
            hovermode='x'
        )
        
        return fig
    
    def create_missing_data_heatmap(self) -> go.Figure:
        """Create heatmap showing missing data patterns"""
        # Create binary matrix (1 = missing, 0 = present)
        missing_matrix = self.df.isnull().astype(int)
        
        # Sample if too many rows
        if len(missing_matrix) > 1000:
            missing_matrix = missing_matrix.sample(n=1000, random_state=42)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.values,
            x=missing_matrix.index,
            y=missing_matrix.columns,
            colorscale=[[0, self.color_scheme['success']], [1, self.color_scheme['danger']]],
            showscale=True,
            colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=["Present", "Missing"])
        ))
        
        fig.update_layout(
            title="Missing Data Pattern (Heatmap)",
            xaxis_title="Row Index",
            yaxis_title="Column",
            height=max(400, len(missing_matrix.columns) * 20),
        )
        
        return fig
    
    def create_data_types_chart(self) -> go.Figure:
        """Create pie chart for data types distribution"""
        data_types = self.profile['data_types']
        
        # Count each type
        type_counts = {}
        for dtype in data_types.values():
            # Simplify dtype names
            if 'int' in dtype:
                dtype = 'Integer'
            elif 'float' in dtype:
                dtype = 'Float'
            elif 'object' in dtype:
                dtype = 'Text'
            elif 'datetime' in dtype:
                dtype = 'DateTime'
            elif 'bool' in dtype:
                dtype = 'Boolean'
            elif 'category' in dtype:
                dtype = 'Category'
            else:
                dtype = 'Other'
            
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title="Data Types Distribution",
            height=400
        )
        
        return fig
    
    def create_outliers_chart(self) -> go.Figure:
        """Create box plots for outliers"""
        outliers = self.profile['outliers']
        
        if not outliers:
            fig = go.Figure()
            fig.add_annotation(
                text="✅ No Outliers Detected!<br>All numeric data within normal ranges",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=self.color_scheme['success'])
            )
            fig.update_layout(
                title="Outlier Analysis",
                height=400,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
            return fig
        
        # Create subplots for each column with outliers
        num_cols = len(outliers)
        cols_per_row = 2
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=num_rows,
            cols=cols_per_row,
            subplot_titles=list(outliers.keys())
        )
        
        for idx, col in enumerate(outliers.keys()):
            row = idx // cols_per_row + 1
            col_idx = idx % cols_per_row + 1
            
            fig.add_trace(
                go.Box(
                    y=self.df[col].dropna(),
                    name=col,
                    marker_color=self.color_scheme['info'],
                    boxmean='sd'
                ),
                row=row,
                col=col_idx
            )
        
        fig.update_layout(
            title="Outlier Detection (Box Plots)",
            height=300 * num_rows,
            showlegend=False
        )
        
        return fig
    
    def create_data_quality_gauge(self) -> go.Figure:
        """Create gauge chart showing overall data quality score"""
        # Calculate quality score
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.profile['missing_data']['total_missing']
        duplicate_rows = self.profile['duplicates']['duplicate_rows']
        
        # Count outliers
        total_outliers = sum(info['count'] for info in self.profile['outliers'].values())
        
        # Quality score calculation (0-100)
        missing_penalty = (missing_cells / total_cells) * 30
        duplicate_penalty = (duplicate_rows / len(self.df)) * 30
        outlier_penalty = (total_outliers / len(self.df)) * 20
        format_penalty = len(self.profile['format_issues']) * 2
        
        quality_score = max(0, 100 - missing_penalty - duplicate_penalty - outlier_penalty - format_penalty)
        
        # Determine color
        if quality_score >= 90:
            color = self.color_scheme['success']
            status = "Excellent"
        elif quality_score >= 75:
            color = self.color_scheme['info']
            status = "Good"
        elif quality_score >= 60:
            color = self.color_scheme['warning']
            status = "Fair"
        else:
            color = self.color_scheme['danger']
            status = "Poor"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Data Quality Score<br><span style='font-size:0.6em'>{status}</span>"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 75], 'color': "lightblue"},
                    {'range': [75, 90], 'color': "lightgreen"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_column_completeness_chart(self) -> go.Figure:
        """Create horizontal bar chart showing column completeness"""
        columns = list(self.df.columns)
        completeness = []
        
        for col in columns:
            non_null = self.df[col].notna().sum()
            pct = (non_null / len(self.df)) * 100
            completeness.append(pct)
        
        # Sort by completeness
        sorted_data = sorted(zip(columns, completeness), key=lambda x: x[1])
        columns_sorted, completeness_sorted = zip(*sorted_data)
        
        # Color based on completeness
        colors = []
        for pct in completeness_sorted:
            if pct >= 95:
                colors.append(self.color_scheme['success'])
            elif pct >= 80:
                colors.append(self.color_scheme['info'])
            elif pct >= 50:
                colors.append(self.color_scheme['warning'])
            else:
                colors.append(self.color_scheme['danger'])
        
        fig = go.Figure(go.Bar(
            x=completeness_sorted,
            y=columns_sorted,
            orientation='h',
            marker_color=colors,
            text=[f"{pct:.1f}%" for pct in completeness_sorted],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>Completeness: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Column Completeness",
            xaxis_title="Completeness (%)",
            yaxis_title="Column",
            height=max(400, len(columns) * 25),
            xaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_numeric_distributions(self) -> go.Figure:
        """Create histograms for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric columns found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=400)
            return fig
        
        # Limit to first 6 numeric columns
        numeric_cols = numeric_cols[:6]
        
        cols_per_row = 2
        num_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=num_rows,
            cols=cols_per_row,
            subplot_titles=numeric_cols
        )
        
        for idx, col in enumerate(numeric_cols):
            row = idx // cols_per_row + 1
            col_idx = idx % cols_per_row + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.df[col].dropna(),
                    name=col,
                    marker_color=self.color_scheme['primary'],
                    showlegend=False
                ),
                row=row,
                col=col_idx
            )
        
        fig.update_layout(
            title="Numeric Column Distributions",
            height=300 * num_rows,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 numeric columns for correlation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=400)
            return fig
        
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix (Numeric Columns)",
            height=max(400, len(corr_matrix) * 30)
        )
        
        return fig
    
    def create_duplicates_chart(self) -> go.Figure:
        """Create pie chart showing duplicate vs unique rows"""
        dup_count = self.profile['duplicates']['duplicate_rows']
        unique_count = len(self.df) - dup_count
        
        fig = go.Figure(data=[go.Pie(
            labels=['Unique Rows', 'Duplicate Rows'],
            values=[unique_count, dup_count],
            hole=0.3,
            marker=dict(colors=[self.color_scheme['success'], self.color_scheme['danger']]),
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title=f"Duplicate Analysis ({dup_count} duplicates found)",
            height=400
        )
        
        return fig
    
    def create_cardinality_chart(self) -> go.Figure:
        """Create chart showing unique value counts per column"""
        columns = list(self.df.columns)
        unique_counts = [self.df[col].nunique() for col in columns]
        total_counts = [len(self.df)] * len(columns)
        
        # Calculate uniqueness ratio
        uniqueness = [(unique / total * 100) for unique, total in zip(unique_counts, total_counts)]
        
        # Sort by uniqueness
        sorted_data = sorted(zip(columns, unique_counts, uniqueness), key=lambda x: x[2])
        columns_sorted, unique_sorted, uniqueness_sorted = zip(*sorted_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=columns_sorted,
            y=unique_sorted,
            name='Unique Values',
            marker_color=self.color_scheme['primary'],
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(unique_sorted, uniqueness_sorted)],
            textposition='auto'
        ))
        
        fig.add_trace(go.Scatter(
            x=columns_sorted,
            y=[len(self.df)] * len(columns_sorted),
            name='Total Rows',
            mode='lines',
            line=dict(color=self.color_scheme['danger'], dash='dash')
        ))
        
        fig.update_layout(
            title="Column Cardinality (Unique Values)",
            xaxis_title="Column",
            yaxis_title="Count",
            height=400,
            hovermode='x unified'
        )
        
        return fig


class ComparisonVisualizer:
    """Creates before/after comparison visualizations"""
    
    def __init__(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        self.original_df = original_df
        self.cleaned_df = cleaned_df
        self.color_scheme = {
            'before': '#e74c3c',
            'after': '#2ecc71'
        }
    
    def create_row_count_comparison(self) -> go.Figure:
        """Compare row counts before and after"""
        original_count = len(self.original_df)
        cleaned_count = len(self.cleaned_df)
        removed = original_count - cleaned_count
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Original', 'Cleaned'],
            y=[original_count, cleaned_count],
            marker_color=[self.color_scheme['before'], self.color_scheme['after']],
            text=[f"{original_count:,}", f"{cleaned_count:,}"],
            textposition='auto'
        ))
        
        if removed > 0:
            fig.add_annotation(
                x=0.5, y=max(original_count, cleaned_count) * 0.9,
                text=f"⬇ {removed:,} rows removed ({removed/original_count*100:.1f}%)",
                showarrow=False,
                font=dict(size=14, color='red')
            )
        
        fig.update_layout(
            title="Row Count: Before vs After",
            yaxis_title="Number of Rows",
            height=400
        )
        
        return fig
    
    def create_missing_data_comparison(self) -> go.Figure:
        """Compare missing data before and after"""
        columns = list(self.original_df.columns)
        
        original_missing = [self.original_df[col].isnull().sum() for col in columns]
        cleaned_missing = [self.cleaned_df[col].isnull().sum() if col in self.cleaned_df.columns else 0 
                          for col in columns]
        
        # Only show columns that had missing data
        data_to_plot = [(col, orig, clean) for col, orig, clean in zip(columns, original_missing, cleaned_missing)
                       if orig > 0 or clean > 0]
        
        if not data_to_plot:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing data in original dataset",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400)
            return fig
        
        cols, orig_vals, clean_vals = zip(*data_to_plot)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cols,
            y=orig_vals,
            name='Before Cleaning',
            marker_color=self.color_scheme['before']
        ))
        
        fig.add_trace(go.Bar(
            x=cols,
            y=clean_vals,
            name='After Cleaning',
            marker_color=self.color_scheme['after']
        ))
        
        fig.update_layout(
            title="Missing Data: Before vs After",
            xaxis_title="Column",
            yaxis_title="Missing Values Count",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_quality_score_comparison(self, original_profile: Dict, cleaned_profile: Dict) -> go.Figure:
        """Compare overall quality scores"""
        # Calculate scores (simplified version)
        def calc_score(df, profile):
            total_cells = len(df) * len(df.columns)
            missing = profile['missing_data']['total_missing']
            return max(0, 100 - (missing / total_cells * 100))
        
        original_score = calc_score(self.original_df, original_profile)
        cleaned_score = calc_score(self.cleaned_df, cleaned_profile)
        improvement = cleaned_score - original_score
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=cleaned_score,
            delta={'reference': original_score, 'relative': False, 'valueformat': '.1f'},
            title={'text': "Data Quality Score"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(height=300)
        
        return fig
    
    def create_memory_usage_comparison(self) -> go.Figure:
        """Compare memory usage before and after"""
        original_memory = self.original_df.memory_usage(deep=True).sum() / 1024**2  # MB
        cleaned_memory = self.cleaned_df.memory_usage(deep=True).sum() / 1024**2  # MB
        saved = original_memory - cleaned_memory
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Original', 'Cleaned'],
            y=[original_memory, cleaned_memory],
            marker_color=[self.color_scheme['before'], self.color_scheme['after']],
            text=[f"{original_memory:.2f} MB", f"{cleaned_memory:.2f} MB"],
            textposition='auto'
        ))
        
        if saved > 0:
            fig.add_annotation(
                x=0.5, y=max(original_memory, cleaned_memory) * 0.9,
                text=f"⬇ {saved:.2f} MB saved ({saved/original_memory*100:.1f}%)",
                showarrow=False,
                font=dict(size=14, color='green')
            )
        
        fig.update_layout(
            title="Memory Usage: Before vs After",
            yaxis_title="Memory (MB)",
            height=400
        )
        
        return fig


class InteractiveDashboard:
    """Creates interactive dashboard with multiple visualizations"""
    
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.visualizer = DataQualityVisualizer(df, profile)
        self.df = df
        self.profile = profile
    
    def render_overview_dashboard(self):
        """Render main overview dashboard"""
        st.subheader("📊 Data Quality Dashboard")
        
        # Top row - Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Rows", 
                f"{len(self.df):,}",
                help="Number of rows in dataset"
            )
        
        with col2:
            missing_pct = (self.profile['missing_data']['total_missing'] / 
                          (len(self.df) * len(self.df.columns)) * 100)
            st.metric(
                "Missing Data", 
                f"{missing_pct:.1f}%",
                delta=f"-{missing_pct:.1f}%" if missing_pct > 0 else "0%",
                delta_color="inverse",
                help="Percentage of missing values"
            )
        
        with col3:
            dup_pct = self.profile['duplicates']['duplicate_percentage']
            st.metric(
                "Duplicates", 
                f"{dup_pct:.1f}%",
                delta=f"-{dup_pct:.1f}%" if dup_pct > 0 else "0%",
                delta_color="inverse",
                help="Percentage of duplicate rows"
            )
        
        with col4:
            outlier_count = sum(info['count'] for info in self.profile['outliers'].values())
            st.metric(
                "Outliers", 
                f"{outlier_count:,}",
                help="Total outliers across all numeric columns"
            )
        
        st.markdown("---")
        
        # Quality gauge
        st.plotly_chart(
            self.visualizer.create_data_quality_gauge(),
            use_container_width=True
        )
        
        # Two column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                self.visualizer.create_missing_data_chart(),
                use_container_width=True
            )
            
            st.plotly_chart(
                self.visualizer.create_duplicates_chart(),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.visualizer.create_data_types_chart(),
                use_container_width=True
            )
            
            st.plotly_chart(
                self.visualizer.create_column_completeness_chart(),
                use_container_width=True
            )
    
    def render_detailed_analysis(self):
        """Render detailed analysis tab"""
        st.subheader("🔍 Detailed Analysis")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Missing Data Patterns",
            "📊 Distributions", 
            "🔗 Correlations",
            "📈 Cardinality"
        ])
        
        with tab1:
            st.plotly_chart(
                self.visualizer.create_missing_data_heatmap(),
                use_container_width=True
            )
        
        with tab2:
            st.plotly_chart(
                self.visualizer.create_numeric_distributions(),
                use_container_width=True
            )
            
            st.plotly_chart(
                self.visualizer.create_outliers_chart(),
                use_container_width=True
            )
        
        with tab3:
            st.plotly_chart(
                self.visualizer.create_correlation_heatmap(),
                use_container_width=True
            )
        
        with tab4:
            st.plotly_chart(
                self.visualizer.create_cardinality_chart(),
                use_container_width=True
            )
    
    def render_comparison_dashboard(self, original_df: pd.DataFrame, 
                                   original_profile: Dict, 
                                   cleaned_profile: Dict):
        """Render before/after comparison dashboard"""
        st.subheader("🔄 Before vs After Comparison")
        
        comp_viz = ComparisonVisualizer(original_df, self.df)
        
        # Key improvements
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_missing = original_profile['missing_data']['total_missing']
            cleaned_missing = cleaned_profile['missing_data']['total_missing']
            improvement = original_missing - cleaned_missing
            st.metric(
                "Missing Values Handled",
                f"{cleaned_missing:,}",
                delta=f"-{improvement:,}" if improvement > 0 else "0",
                delta_color="inverse"
            )
        
        with col2:
            original_rows = len(original_df)
            cleaned_rows = len(self.df)
            removed = original_rows - cleaned_rows
            st.metric(
                "Rows After Cleaning",
                f"{cleaned_rows:,}",
                delta=f"-{removed:,}" if removed > 0 else "0",
                delta_color="off"
            )
        
        with col3:
            original_mem = original_df.memory_usage(deep=True).sum() / 1024**2
            cleaned_mem = self.df.memory_usage(deep=True).sum() / 1024**2
            saved = original_mem - cleaned_mem
            st.metric(
                "Memory Usage (MB)",
                f"{cleaned_mem:.2f}",
                delta=f"-{saved:.2f}" if saved > 0 else "0",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                comp_viz.create_row_count_comparison(),
                use_container_width=True
            )
            
            st.plotly_chart(
                comp_viz.create_memory_usage_comparison(),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                comp_viz.create_missing_data_comparison(),
                use_container_width=True
            )
            
            st.plotly_chart(
                comp_viz.create_quality_score_comparison(original_profile, cleaned_profile),
                use_container_width=True
            )