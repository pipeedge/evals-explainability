"""
Visualization and Interactive Reporting

This module provides comprehensive visualization and interactive reporting capabilities
for the LLM explainability framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time

from ..core.explainability_engine import ExplainabilityReport
from ..core.failure_classifier import FailureClassification
from ..core.root_cause_analyzer import RootCauseAnalysis
from ..core.recommendation_engine import RecommendationSuite


class ExplainabilityReporter:
    """
    Comprehensive visualization and reporting system
    
    Innovation: Interactive multi-modal visualizations that combine
    statistical analysis, causal graphs, and stakeholder-specific views.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(self, reports: List[ExplainabilityReport], 
                                    output_name: str = "explainability_analysis") -> str:
        """
        Generate comprehensive analysis report with visualizations
        
        Args:
            reports: List of explainability reports to analyze
            output_name: Base name for output files
            
        Returns:
            Path to generated HTML report
        """
        print("📊 Generating comprehensive explainability report...")
        
        # Generate individual visualizations
        plots = self._generate_all_plots(reports, output_name)
        
        # Create HTML report
        html_report = self._create_html_report(reports, plots, output_name)
        
        # Save HTML report
        html_path = self.output_dir / f"{output_name}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ Comprehensive report saved to {html_path}")
        return str(html_path)
    
    def _generate_all_plots(self, reports: List[ExplainabilityReport], 
                           output_name: str) -> Dict[str, str]:
        """Generate all visualization plots"""
        plots = {}
        
        # 1. Failure distribution analysis
        plots['failure_distribution'] = self._plot_failure_distribution(reports, output_name)
        
        # 2. Confidence analysis
        plots['confidence_analysis'] = self._plot_confidence_analysis(reports, output_name)
        
        # 3. Task type analysis
        plots['task_analysis'] = self._plot_task_analysis(reports, output_name)
        
        # 4. Root cause patterns
        plots['root_cause_patterns'] = self._plot_root_cause_patterns(reports, output_name)
        
        # 5. Recommendation effectiveness
        plots['recommendation_analysis'] = self._plot_recommendation_analysis(reports, output_name)
        
        # 6. Performance metrics
        plots['performance_metrics'] = self._plot_performance_metrics(reports, output_name)
        
        # 7. Causal network visualization
        plots['causal_networks'] = self._plot_causal_networks(reports, output_name)
        
        return plots
    
    def _plot_failure_distribution(self, reports: List[ExplainabilityReport], 
                                  output_name: str) -> str:
        """Plot failure category distribution"""
        # Extract failure categories
        categories = [report.failure_classification.failure_category for report in reports]
        category_counts = pd.Series(categories).value_counts()
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add pie chart
        fig.add_trace(go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hole=0.4,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Failure Category Distribution",
            font=dict(size=14),
            showlegend=True,
            height=500
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_failure_distribution.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_confidence_analysis(self, reports: List[ExplainabilityReport], 
                                 output_name: str) -> str:
        """Plot confidence score analysis"""
        # Extract confidence data
        data = []
        for report in reports:
            data.append({
                'instance_id': report.instance_id,
                'overall_confidence': report.confidence_score,
                'classification_confidence': report.failure_classification.confidence_score,
                'root_cause_confidence': report.root_cause_analysis.confidence_score,
                'failure_category': report.failure_classification.failure_category,
                'task_type': report.task_type
            })
        
        df = pd.DataFrame(data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Overall Confidence Distribution',
                'Confidence by Failure Category',
                'Confidence by Task Type',
                'Confidence Components Correlation'
            ],
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Overall confidence histogram
        fig.add_trace(
            go.Histogram(x=df['overall_confidence'], nbinsx=20, name='Overall Confidence'),
            row=1, col=1
        )
        
        # Confidence by failure category
        for category in df['failure_category'].unique():
            category_data = df[df['failure_category'] == category]
            fig.add_trace(
                go.Box(y=category_data['overall_confidence'], name=category),
                row=1, col=2
            )
        
        # Confidence by task type
        for task_type in df['task_type'].unique():
            task_data = df[df['task_type'] == task_type]
            fig.add_trace(
                go.Box(y=task_data['overall_confidence'], name=task_type),
                row=2, col=1
            )
        
        # Confidence correlation
        fig.add_trace(
            go.Scatter(
                x=df['classification_confidence'],
                y=df['root_cause_confidence'],
                mode='markers',
                text=df['instance_id'],
                name='Confidence Correlation'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Confidence Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_confidence_analysis.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_task_analysis(self, reports: List[ExplainabilityReport], 
                           output_name: str) -> str:
        """Plot task type analysis"""
        # Extract task data
        data = []
        for report in reports:
            data.append({
                'task_type': report.task_type,
                'failure_category': report.failure_classification.failure_category,
                'confidence': report.confidence_score,
                'processing_time': report.processing_time,
                'quality_score': report.quality_metrics.get('overall_quality', 0.0)
            })
        
        df = pd.DataFrame(data)
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Task Type Distribution',
                'Processing Time by Task Type',
                'Quality Score by Task Type',
                'Failure Pattern by Task Type'
            ],
            specs=[[{"type": "pie"}, {"type": "violin"}],
                   [{"type": "violin"}, {"type": "bar"}]]
        )
        
        # Task type distribution
        task_counts = df['task_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=task_counts.index, values=task_counts.values),
            row=1, col=1
        )
        
        # Processing time by task type
        for task_type in df['task_type'].unique():
            task_data = df[df['task_type'] == task_type]
            fig.add_trace(
                go.Violin(y=task_data['processing_time'], name=task_type),
                row=1, col=2
            )
        
        # Quality score by task type
        for task_type in df['task_type'].unique():
            task_data = df[df['task_type'] == task_type]
            fig.add_trace(
                go.Violin(y=task_data['quality_score'], name=task_type),
                row=2, col=1
            )
        
        # Failure pattern heatmap data
        failure_matrix = pd.crosstab(df['task_type'], df['failure_category'])
        fig.add_trace(
            go.Heatmap(
                z=failure_matrix.values,
                x=failure_matrix.columns,
                y=failure_matrix.index,
                colorscale='Blues'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Task Type Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_task_analysis.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_root_cause_patterns(self, reports: List[ExplainabilityReport], 
                                 output_name: str) -> str:
        """Plot root cause analysis patterns"""
        # Extract root cause data
        data = []
        for report in reports:
            root_cause = report.root_cause_analysis
            for factor in root_cause.causal_factors[:3]:  # Top 3 factors
                data.append({
                    'instance_id': report.instance_id,
                    'failure_category': report.failure_classification.failure_category,
                    'factor_name': factor.factor_name,
                    'factor_type': factor.factor_type,
                    'causal_strength': factor.causal_strength,
                    'confidence': factor.confidence
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No root cause data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=20
            )
        else:
            # Create analysis plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Causal Factor Types',
                    'Causal Strength Distribution',
                    'Factor Confidence vs Strength',
                    'Top Causal Factors'
                ]
            )
            
            # Factor types distribution
            factor_counts = df['factor_type'].value_counts()
            fig.add_trace(
                go.Bar(x=factor_counts.index, y=factor_counts.values),
                row=1, col=1
            )
            
            # Causal strength distribution
            fig.add_trace(
                go.Histogram(x=df['causal_strength'], nbinsx=20),
                row=1, col=2
            )
            
            # Confidence vs strength scatter
            fig.add_trace(
                go.Scatter(
                    x=df['causal_strength'],
                    y=df['confidence'],
                    mode='markers',
                    text=df['factor_name']
                ),
                row=2, col=1
            )
            
            # Top factors
            top_factors = df.groupby('factor_name')['causal_strength'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=top_factors.values, y=top_factors.index, orientation='h'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Root Cause Analysis Patterns",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_root_cause_patterns.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_recommendation_analysis(self, reports: List[ExplainabilityReport], 
                                    output_name: str) -> str:
        """Plot recommendation analysis"""
        # Extract recommendation data
        data = []
        for report in reports:
            recommendations = report.recommendation_suite.recommendations
            for rec in recommendations:
                data.append({
                    'instance_id': report.instance_id,
                    'failure_category': report.failure_classification.failure_category,
                    'rec_type': rec.recommendation_type.value,
                    'stakeholder': rec.stakeholder_type.value,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'confidence': rec.confidence,
                    'priority_score': rec.priority_score
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No recommendation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=20
            )
        else:
            # Create recommendation analysis with proper subplot types
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=[
                    'Recommendation Types',
                    'Impact vs Effort Analysis',
                    'Stakeholder Distribution',
                    'Priority Scores by Type'
                ]
            )
            
            # Recommendation types
            type_counts = df['rec_type'].value_counts()
            fig.add_trace(
                go.Bar(x=type_counts.index, y=type_counts.values),
                row=1, col=1
            )
            
            # Impact vs Effort bubble plot
            fig.add_trace(
                go.Scatter(
                    x=df['implementation_effort'],
                    y=df['expected_impact'],
                    mode='markers',
                    marker=dict(
                        size=df['confidence'] * 20,
                        color=df['priority_score'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=df['rec_type']
                ),
                row=1, col=2
            )
            
            # Stakeholder distribution
            stakeholder_counts = df['stakeholder'].value_counts()
            fig.add_trace(
                go.Pie(labels=stakeholder_counts.index, values=stakeholder_counts.values),
                row=2, col=1
            )
            
            # Priority scores by type
            priority_by_type = df.groupby('rec_type')['priority_score'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=priority_by_type.values, y=priority_by_type.index, orientation='h'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Recommendation Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_recommendation_analysis.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_performance_metrics(self, reports: List[ExplainabilityReport], 
                                 output_name: str) -> str:
        """Plot performance metrics analysis"""
        # Extract performance data
        data = []
        for i, report in enumerate(reports):
            data.append({
                'instance_id': report.instance_id,
                'processing_time': report.processing_time,
                'confidence_score': report.confidence_score,
                'quality_score': report.quality_metrics.get('overall_quality', 0.0),
                'failure_category': report.failure_classification.failure_category,
                'analysis_order': i + 1
            })
        
        df = pd.DataFrame(data)
        
        # Create performance dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Processing Time Distribution',
                'Quality vs Confidence',
                'Performance Over Time',
                'Performance by Failure Category'
            ]
        )
        
        # Processing time distribution
        fig.add_trace(
            go.Histogram(x=df['processing_time'], nbinsx=20),
            row=1, col=1
        )
        
        # Quality vs Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=df['confidence_score'],
                y=df['quality_score'],
                mode='markers',
                text=df['instance_id'],
                marker=dict(
                    size=df['processing_time'] * 10,
                    color=df['processing_time'],
                    colorscale='Plasma',
                    showscale=True
                )
            ),
            row=1, col=2
        )
        
        # Performance over time
        fig.add_trace(
            go.Scatter(
                x=df['analysis_order'],
                y=df['processing_time'],
                mode='lines+markers',
                name='Processing Time'
            ),
            row=2, col=1
        )
        
        # Add quality trend
        fig.add_trace(
            go.Scatter(
                x=df['analysis_order'],
                y=df['quality_score'],
                mode='lines+markers',
                name='Quality Score',
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # Performance by failure category
        performance_by_category = df.groupby('failure_category').agg({
            'processing_time': 'mean',
            'quality_score': 'mean',
            'confidence_score': 'mean'
        })
        
        fig.add_trace(
            go.Bar(
                x=performance_by_category.index,
                y=performance_by_category['processing_time'],
                name='Avg Processing Time'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Performance Metrics Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_performance_metrics.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _plot_causal_networks(self, reports: List[ExplainabilityReport], 
                             output_name: str) -> str:
        """Plot causal network visualizations"""
        # Create a combined causal network from all reports
        combined_graph = nx.DiGraph()
        
        for report in reports:
            if report.root_cause_analysis.causal_graph:
                # Merge graphs
                for node, data in report.root_cause_analysis.causal_graph.nodes(data=True):
                    if node not in combined_graph:
                        combined_graph.add_node(node, **data)
                
                for source, target, data in report.root_cause_analysis.causal_graph.edges(data=True):
                    if combined_graph.has_edge(source, target):
                        # Average the weights
                        current_weight = combined_graph[source][target].get('weight', 0)
                        new_weight = data.get('weight', 0)
                        combined_graph[source][target]['weight'] = (current_weight + new_weight) / 2
                    else:
                        combined_graph.add_edge(source, target, **data)
        
        if len(combined_graph.nodes()) == 0:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No causal network data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=20
            )
        else:
            # Create network visualization using plotly
            pos = nx.spring_layout(combined_graph, k=1, iterations=50)
            
            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in combined_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                weight = combined_graph[edge[0]][edge[1]].get('weight', 0)
                edge_info.append(f"{edge[0]} → {edge[1]}<br>Weight: {weight:.3f}")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            node_colors = []
            
            for node in combined_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Node info
                node_data = combined_graph.nodes[node]
                node_type = node_data.get('type', 'unknown')
                importance = node_data.get('importance', 0.0)
                
                node_info.append(f"{node}<br>Type: {node_type}<br>Importance: {importance:.3f}")
                
                # Color by type
                type_colors = {
                    'input': '#FF9999',
                    'processing': '#99FF99', 
                    'output': '#9999FF',
                    'semantic': '#FFFF99',
                    'unknown': '#CCCCCC'
                }
                node_colors.append(type_colors.get(node_type, '#CCCCCC'))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_info,
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='black')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=dict(text='Causal Network Analysis', font=dict(size=16)),
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Node colors: Input (red), Processing (green), Output (blue), Semantic (yellow)",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor="left", yanchor="bottom",
                                   font=dict(size=12)
                               ) ],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                           ))
        
        # Save plot
        plot_path = self.output_dir / f"{output_name}_causal_networks.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)
    
    def _create_html_report(self, reports: List[ExplainabilityReport], 
                           plots: Dict[str, str], output_name: str) -> str:
        """Create comprehensive HTML report"""
        
        # Calculate summary statistics
        total_reports = len(reports)
        avg_confidence = np.mean([r.confidence_score for r in reports])
        avg_processing_time = np.mean([r.processing_time for r in reports])
        
        failure_categories = [r.failure_classification.failure_category for r in reports]
        most_common_failure = max(set(failure_categories), key=failure_categories.count) if failure_categories else "N/A"
        
        task_types = [r.task_type for r in reports]
        task_distribution = {task: task_types.count(task) for task in set(task_types)}
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Explainability Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .plot-section {{
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .plot-header {{
            background-color: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }}
        .plot-content {{
            padding: 20px;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .navigation {{
            position: sticky;
            top: 20px;
            background-color: #343a40;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .navigation a {{
            color: #ffc107;
            text-decoration: none;
            margin-right: 20px;
        }}
        .navigation a:hover {{
            text-decoration: underline;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LLM Explainability Analysis Report</h1>
            <p>Comprehensive analysis of {total_reports} failure instances</p>
            <p><em>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</em></p>
        </div>
        
        <div class="navigation">
            <strong>Quick Navigation:</strong>
            <a href="#summary">Summary</a>
            <a href="#failures">Failure Analysis</a>
            <a href="#confidence">Confidence Analysis</a>
            <a href="#tasks">Task Analysis</a>
            <a href="#causes">Root Causes</a>
            <a href="#recommendations">Recommendations</a>
            <a href="#performance">Performance</a>
            <a href="#networks">Causal Networks</a>
        </div>
        
        <section id="summary">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Analyses</h3>
                    <div class="value">{total_reports}</div>
                    <p>Failure instances analyzed</p>
                </div>
                <div class="summary-card">
                    <h3>Average Confidence</h3>
                    <div class="value">{avg_confidence:.2f}</div>
                    <p>Overall analysis confidence</p>
                </div>
                <div class="summary-card">
                    <h3>Processing Time</h3>
                    <div class="value">{avg_processing_time:.1f}s</div>
                    <p>Average analysis time</p>
                </div>
                <div class="summary-card">
                    <h3>Common Failure</h3>
                    <div class="value" style="font-size: 1.2em;">{most_common_failure}</div>
                    <p>Most frequent failure type</p>
                </div>
            </div>
            
            <h3>Task Distribution</h3>
            <ul>
        """
        
        for task_type, count in task_distribution.items():
            percentage = (count / total_reports) * 100
            html_template += f"<li><strong>{task_type}:</strong> {count} instances ({percentage:.1f}%)</li>"
        
        html_template += """
            </ul>
        </section>
        """
        
        # Add plot sections
        plot_sections = [
            ("failures", "Failure Distribution Analysis", "failure_distribution"),
            ("confidence", "Confidence Analysis", "confidence_analysis"),
            ("tasks", "Task Type Analysis", "task_analysis"),
            ("causes", "Root Cause Patterns", "root_cause_patterns"),
            ("recommendations", "Recommendation Analysis", "recommendation_analysis"),
            ("performance", "Performance Metrics", "performance_metrics"),
            ("networks", "Causal Networks", "causal_networks")
        ]
        
        for section_id, title, plot_key in plot_sections:
            if plot_key in plots:
                plot_file = Path(plots[plot_key]).name
                html_template += f"""
        <section id="{section_id}" class="plot-section">
            <div class="plot-header">
                <h2>{title}</h2>
            </div>
            <div class="plot-content">
                <iframe src="{plot_file}"></iframe>
            </div>
        </section>
                """
        
        # Add individual report summaries
        html_template += """
        <section id="reports">
            <h2>📋 Individual Report Summaries</h2>
            <div class="plot-grid">
        """
        
        for i, report in enumerate(reports[:10]):  # Show first 10 reports
            html_template += f"""
                <div class="summary-card">
                    <h3>{report.instance_id}</h3>
                    <p><strong>Task:</strong> {report.task_type}</p>
                    <p><strong>Failure:</strong> {report.failure_classification.failure_category}</p>
                    <p><strong>Confidence:</strong> {report.confidence_score:.3f}</p>
                    <p><strong>Time:</strong> {report.processing_time:.2f}s</p>
                </div>
            """
        
        if len(reports) > 10:
            html_template += f"<p><em>... and {len(reports) - 10} more reports</em></p>"
        
        html_template += """
            </div>
        </section>
        
        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;">
            <p><em>Generated by LLM Explainability Framework v1.0.0</em></p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def generate_individual_report_html(self, report: ExplainabilityReport) -> str:
        """Generate HTML version of individual report"""
        # Convert markdown to HTML (simplified)
        html_content = report.markdown_report.replace('\n', '<br>\n')
        html_content = html_content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        html_content = html_content.replace('## ', '<h2>').replace('\n', '</h2>\n')
        html_content = html_content.replace('### ', '<h3>').replace('\n', '</h3>\n')
        html_content = html_content.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        html_content = html_content.replace('```', '<pre><code>').replace('```', '</code></pre>')
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainability Report - {report.instance_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; background-color: #f9f9f9; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .metadata {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="metadata">
            <strong>Analysis Metadata:</strong><br>
            Confidence: {report.confidence_score:.3f} | 
            Processing Time: {report.processing_time:.2f}s | 
            Quality Score: {report.quality_metrics.get('overall_quality', 0.0):.3f}
        </div>
        {html_content}
    </div>
</body>
</html>
        """
        
        return html_template 