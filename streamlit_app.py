"""
Main Streamlit Application Entry Point
This file serves as the primary entry point for the Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Page config
st.set_page_config(
    page_title="Student Habits Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_safe(file_path):
    """Safely load data with error handling"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data_safe(df):
    """Safely clean data with error handling"""
    if df is None:
        return None
    # Remove any rows with missing values
    cleaned = df.dropna()
    return cleaned

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Student Habits and Academic Performance',
        width=700,
        height=600
    )
    return fig

def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3498db;
        }
        .insight-box {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Student Habits vs Academic Performance Analysis")
    st.markdown("""
    <div class="insight-box">
    This comprehensive analysis explores the relationship between student habits 
    (study hours, sleep, screen time, attendance) and academic performance (GPA).
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    
    # Load data
    try:
        # Try different possible paths
        possible_paths = [
            "data/Study_Hours_per_Week,Sleep_Hours.csv",
            os.path.join("data", "Study_Hours_per_Week,Sleep_Hours.csv"),
            os.path.join(current_dir, "data", "Study_Hours_per_Week,Sleep_Hours.csv")
        ]
        
        data = None
        for path in possible_paths:
            try:
                data = load_data_safe(path)
                if data is not None:
                    st.sidebar.success(f"✅ Data loaded from: {path}")
                    break
            except:
                continue
        
        if data is None:
            st.error("❌ Dataset not found! Please ensure the CSV file is in the data/ directory.")
            st.info("Expected location: data/Study_Hours_per_Week,Sleep_Hours.csv")
            st.stop()
        
        # Clean data
        cleaned_data = clean_data_safe(data)
        if cleaned_data is None:
            st.error("❌ Failed to process data!")
            st.stop()
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Correlations", "Distributions", "Performance Analysis", "Insights"]
        )
        
        # Show raw data option
        show_raw_data = st.sidebar.checkbox("Show Raw Data")
        if show_raw_data:
            st.subheader("📋 Raw Dataset")
            st.dataframe(cleaned_data)
            st.write(f"Dataset shape: {cleaned_data.shape}")
        
        # Main analysis sections
        if analysis_type == "Overview":
            st.header("📈 Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(cleaned_data))
            with col2:
                st.metric("Average GPA", f"{cleaned_data['GPA'].mean():.2f}")
            with col3:
                st.metric("Avg Study Hours/Week", f"{cleaned_data['Study_Hours_per_Week'].mean():.1f}")
            with col4:
                st.metric("Avg Sleep Hours/Night", f"{cleaned_data['Sleep_Hours_per_Night'].mean():.1f}")
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            st.dataframe(cleaned_data.describe())
            
            # Basic visualizations
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(cleaned_data, x='GPA', title='GPA Distribution', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(cleaned_data, x='Study_Hours_per_Week', 
                                 title='Study Hours Distribution', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Correlations":
            st.header("🔗 Correlation Analysis")
            
            # Correlation heatmap
            fig = create_correlation_heatmap(cleaned_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation with GPA
            st.subheader("🎯 Correlations with GPA")
            corr_matrix = cleaned_data.corr()
            gpa_corr = corr_matrix['GPA'].sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Positive Correlations:**")
                positive_corr = gpa_corr[gpa_corr > 0].drop('GPA')
                for var, corr in positive_corr.items():
                    st.write(f"• {var}: {corr:.3f}")
            
            with col2:
                st.write("**Negative Correlations:**")
                negative_corr = gpa_corr[gpa_corr < 0]
                for var, corr in negative_corr.items():
                    st.write(f"• {var}: {corr:.3f}")
                    
        elif analysis_type == "Distributions":
            st.header("📊 Variable Distributions")
            
            # Select variable
            selected_var = st.selectbox("Select Variable:", cleaned_data.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(cleaned_data, x=selected_var, 
                                 title=f'Distribution of {selected_var}', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.box(cleaned_data, y=selected_var,
                           title=f'Box Plot of {selected_var}')
                st.plotly_chart(fig, use_container_width=True)
                
            # Statistics
            st.subheader(f"📈 {selected_var} Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{cleaned_data[selected_var].mean():.2f}")
            with col2:
                st.metric("Median", f"{cleaned_data[selected_var].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{cleaned_data[selected_var].std():.2f}")
            with col4:
                st.metric("Range", f"{cleaned_data[selected_var].max() - cleaned_data[selected_var].min():.2f}")
                
        elif analysis_type == "Performance Analysis":
            st.header("🎯 Performance Analysis")
            
            # Create performance categories
            data_copy = cleaned_data.copy()
            data_copy['Performance'] = pd.cut(data_copy['GPA'], 
                                           bins=[0, 2.5, 3.0, 3.5, 4.0],
                                           labels=['Low', 'Medium', 'High', 'Excellent'])
            
            # Performance distribution
            perf_counts = data_copy['Performance'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=perf_counts.values, names=perf_counts.index,
                            title="Performance Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(x=perf_counts.index, y=perf_counts.values,
                           title="Performance Category Counts")
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            st.subheader("📊 Performance Comparison")
            high_performers = cleaned_data[cleaned_data['GPA'] >= 3.5]
            low_performers = cleaned_data[cleaned_data['GPA'] < 2.5]
            
            if len(high_performers) > 0 and len(low_performers) > 0:
                comparison_data = []
                variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                           'Screen_Time_per_Day', 'Attendance_Percentage']
                
                for var in variables:
                    high_avg = high_performers[var].mean()
                    low_avg = low_performers[var].mean()
                    comparison_data.append({
                        'Variable': var.replace('_', ' '),
                        'High Performers': f"{high_avg:.1f}",
                        'Low Performers': f"{low_avg:.1f}",
                        'Difference': f"{high_avg - low_avg:+.1f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data))
                
        elif analysis_type == "Insights":
            st.header("💡 Key Insights & Recommendations")
            
            # Key insights
            corr_matrix = cleaned_data.corr()
            gpa_corr_abs = corr_matrix['GPA'].drop('GPA').abs().sort_values(ascending=False)
            strongest_factor = gpa_corr_abs.index[0]
            strongest_corr = corr_matrix['GPA'][strongest_factor]
            
            st.subheader("🔍 Key Findings")
            st.write(f"🏆 **{strongest_factor.replace('_', ' ')}** has the strongest relationship with GPA (r = {strongest_corr:.3f})")
            
            # Screen time analysis
            screen_corr = corr_matrix['GPA']['Screen_Time_per_Day']
            if screen_corr < -0.2:
                st.write(f"📱 Screen time negatively impacts GPA (r = {screen_corr:.3f})")
            
            # Sleep analysis
            sleep_corr = corr_matrix['GPA']['Sleep_Hours_per_Night']
            if sleep_corr > 0.2:
                st.write(f"😴 Adequate sleep positively impacts GPA (r = {sleep_corr:.3f})")
            
            # Success formula
            high_performers = cleaned_data[cleaned_data['GPA'] >= 3.5]
            if len(high_performers) > 0:
                st.subheader("🏆 Success Formula")
                st.success("**Based on high performers (GPA ≥ 3.5):**")
                st.write(f"• 📖 Study: {high_performers['Study_Hours_per_Week'].mean():.1f}+ hours/week")
                st.write(f"• 😴 Sleep: {high_performers['Sleep_Hours_per_Night'].mean():.1f}+ hours/night")
                st.write(f"• 🎯 Attendance: {high_performers['Attendance_Percentage'].mean():.1f}%+")
                st.write(f"• 📱 Screen Time: ≤{high_performers['Screen_Time_per_Day'].mean():.1f} hours/day")
            
            # 3D visualization
            st.subheader("📊 3D Relationship Analysis")
            fig = px.scatter_3d(cleaned_data, 
                              x='Study_Hours_per_Week', 
                              y='Sleep_Hours_per_Night', 
                              z='GPA',
                              color='Attendance_Percentage',
                              size='Screen_Time_per_Day',
                              title='3D Relationship: Study Hours, Sleep Hours, and GPA')
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure all required files are in place and dependencies are installed.")
        st.code("pip install -r requirements.txt")

if __name__ == "__main__":
    main()