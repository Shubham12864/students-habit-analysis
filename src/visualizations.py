import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_gpa_distribution(data):
    """Create GPA distribution histogram"""
    fig = px.histogram(data, x='GPA', nbins=20, title='GPA Distribution',
                       color_discrete_sequence=['#3498db'])
    fig.update_layout(
        xaxis_title="GPA",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def create_study_hours_distribution(data):
    """Create study hours distribution histogram"""
    fig = px.histogram(data, x='Study_Hours_per_Week', nbins=20, 
                       title='Study Hours per Week Distribution',
                       color_discrete_sequence=['#e74c3c'])
    fig.update_layout(
        xaxis_title="Study Hours per Week",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def create_sleep_distribution(data):
    """Create sleep hours distribution histogram"""
    fig = px.histogram(data, x='Sleep_Hours_per_Night', nbins=20, 
                       title='Sleep Hours per Night Distribution',
                       color_discrete_sequence=['#9b59b6'])
    fig.update_layout(
        xaxis_title="Sleep Hours per Night",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def create_attendance_distribution(data):
    """Create attendance distribution histogram"""
    fig = px.histogram(data, x='Attendance_Percentage', nbins=20, 
                       title='Attendance Percentage Distribution',
                       color_discrete_sequence=['#f39c12'])
    fig.update_layout(
        xaxis_title="Attendance Percentage",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Student Habits and Academic Performance',
        width=600,
        height=500,
        template="plotly_white"
    )
    return fig

def create_histogram_with_kde(data, variable):
    """Create histogram with marginal box plot"""
    fig = px.histogram(data, x=variable, marginal="box", 
                       title=f'{variable} Distribution with Box Plot',
                       template="plotly_white")
    return fig

def create_box_plot(data, variable):
    """Create box plot for a variable"""
    fig = px.box(data, y=variable, title=f'{variable} Box Plot')
    fig.update_layout(showlegend=False, template="plotly_white")
    return fig

def create_study_vs_gpa_scatter(data):
    """Create scatter plot of study hours vs GPA"""
    fig = px.scatter(data, x='Study_Hours_per_Week', y='GPA',
                     color='Attendance_Percentage',
                     size='Sleep_Hours_per_Night',
                     title='Study Hours vs GPA (colored by Attendance, sized by Sleep Hours)',
                     color_continuous_scale='viridis',
                     template="plotly_white")
    
    # Add trend line
    z = np.polyfit(data['Study_Hours_per_Week'], data['GPA'], 1)
    p = np.poly1d(z)
    fig.add_traces(go.Scatter(x=data['Study_Hours_per_Week'], y=p(data['Study_Hours_per_Week']),
                             mode='lines', name='Trend Line', line=dict(color='red', dash='dash')))
    
    return fig

def create_sleep_vs_gpa_scatter(data):
    """Create scatter plot of sleep hours vs GPA"""
    fig = px.scatter(data, x='Sleep_Hours_per_Night', y='GPA',
                     color='Study_Hours_per_Week',
                     title='Sleep Hours vs GPA (colored by Study Hours)',
                     color_continuous_scale='plasma',
                     template="plotly_white")
    
    # Add trend line
    z = np.polyfit(data['Sleep_Hours_per_Night'], data['GPA'], 1)
    p = np.poly1d(z)
    fig.add_traces(go.Scatter(x=data['Sleep_Hours_per_Night'], y=p(data['Sleep_Hours_per_Night']),
                             mode='lines', name='Trend Line', line=dict(color='red', dash='dash')))
    
    return fig

def create_screen_time_vs_gpa_scatter(data):
    """Create scatter plot of screen time vs GPA"""
    fig = px.scatter(data, x='Screen_Time_per_Day', y='GPA',
                     color='Attendance_Percentage',
                     title='Screen Time vs GPA (colored by Attendance)',
                     color_continuous_scale='cividis',
                     template="plotly_white")
    
    # Add trend line
    z = np.polyfit(data['Screen_Time_per_Day'], data['GPA'], 1)
    p = np.poly1d(z)
    fig.add_traces(go.Scatter(x=data['Screen_Time_per_Day'], y=p(data['Screen_Time_per_Day']),
                             mode='lines', name='Trend Line', line=dict(color='red', dash='dash')))
    
    return fig

def create_3d_scatter(data):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(data, x='Study_Hours_per_Week', y='Sleep_Hours_per_Night', z='GPA',
                        color='Attendance_Percentage',
                        size='Screen_Time_per_Day',
                        title='3D Relationship: Study Hours, Sleep Hours, and GPA',
                        template="plotly_white")
    return fig

def create_attendance_vs_gpa_analysis(data):
    """Create attendance vs GPA analysis"""
    # Create attendance categories
    data_copy = data.copy()
    data_copy['Attendance_Category'] = pd.cut(data_copy['Attendance_Percentage'], 
                                              bins=[0, 70, 80, 90, 100], 
                                              labels=['Low (<70%)', 'Medium (70-80%)', 
                                                     'High (80-90%)', 'Excellent (>90%)'])
    
    fig = px.box(data_copy, x='Attendance_Category', y='GPA',
                 title='GPA Distribution by Attendance Category',
                 template="plotly_white")
    return fig

def create_multi_variable_analysis(data):
    """Create multi-variable analysis subplot"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Study Hours vs GPA', 'Sleep Hours vs GPA', 
                       'Screen Time vs GPA', 'Attendance vs GPA'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Study Hours vs GPA
    fig.add_trace(
        go.Scatter(x=data['Study_Hours_per_Week'], y=data['GPA'], 
                  mode='markers', name='Study Hours', marker=dict(color='blue', size=6)),
        row=1, col=1
    )
    
    # Sleep Hours vs GPA
    fig.add_trace(
        go.Scatter(x=data['Sleep_Hours_per_Night'], y=data['GPA'], 
                  mode='markers', name='Sleep Hours', marker=dict(color='green', size=6)),
        row=1, col=2
    )
    
    # Screen Time vs GPA
    fig.add_trace(
        go.Scatter(x=data['Screen_Time_per_Day'], y=data['GPA'], 
                  mode='markers', name='Screen Time', marker=dict(color='red', size=6)),
        row=2, col=1
    )
    
    # Attendance vs GPA
    fig.add_trace(
        go.Scatter(x=data['Attendance_Percentage'], y=data['GPA'], 
                  mode='markers', name='Attendance', marker=dict(color='orange', size=6)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600, 
        showlegend=False, 
        title_text="Multi-Variable Analysis",
        template="plotly_white"
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Study Hours per Week", row=1, col=1)
    fig.update_xaxes(title_text="Sleep Hours per Night", row=1, col=2)
    fig.update_xaxes(title_text="Screen Time per Day", row=2, col=1)
    fig.update_xaxes(title_text="Attendance Percentage", row=2, col=2)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="GPA", row=1, col=1)
    fig.update_yaxes(title_text="GPA", row=1, col=2)
    fig.update_yaxes(title_text="GPA", row=2, col=1)
    fig.update_yaxes(title_text="GPA", row=2, col=2)
    
    return fig

def create_performance_comparison(data):
    """Create performance comparison chart"""
    data_copy = data.copy()
    data_copy['Performance_Category'] = pd.cut(data_copy['GPA'], 
                                               bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                               labels=['Low', 'Medium', 'High', 'Excellent'])
    
    # Group by performance and calculate means
    performance_stats = data_copy.groupby('Performance_Category').agg({
        'Study_Hours_per_Week': 'mean',
        'Sleep_Hours_per_Night': 'mean',
        'Screen_Time_per_Day': 'mean',
        'Attendance_Percentage': 'mean'
    }).reset_index()
    
    fig = px.bar(performance_stats, x='Performance_Category', 
                 y=['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Screen_Time_per_Day', 'Attendance_Percentage'],
                 title='Average Habits by Performance Category',
                 barmode='group',
                 template="plotly_white")
    
    return fig

# Legacy functions for backward compatibility
def plot_study_hours_vs_gpa(data):
    """Legacy function - use create_study_vs_gpa_scatter instead"""
    return create_study_vs_gpa_scatter(data)

def plot_sleep_hours_vs_gpa(data):
    """Legacy function - use create_sleep_vs_gpa_scatter instead"""
    return create_sleep_vs_gpa_scatter(data)

def plot_attendance_vs_gpa(data):
    """Legacy function - use create_attendance_vs_gpa_analysis instead"""
    return create_attendance_vs_gpa_analysis(data)

def plot_screen_time_vs_gpa(data):
    """Legacy function - use create_screen_time_vs_gpa_scatter instead"""
    return create_screen_time_vs_gpa_scatter(data)