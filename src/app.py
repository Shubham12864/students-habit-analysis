import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_analysis import load_data, clean_data, analyze_data, correlation_matrix
from visualizations import *
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Student Habits vs Academic Performance Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Custom CSS for better styling
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
    
    # Load and clean data
    try:
        # Use the correct path that matches your file structure
        data = load_data("data/Study_Hours_per_Week,Sleep_Hours.csv")
        cleaned_data = clean_data(data)
        
        # Sidebar options
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Correlation Analysis", "Distribution Analysis", 
             "Relationship Analysis", "Statistical Tests", "Predictive Insights"]
        )
        
        show_raw_data = st.sidebar.checkbox("Show Raw Data")
        
        if show_raw_data:
            st.subheader("📋 Raw Dataset")
            st.dataframe(cleaned_data)
            st.write(f"Dataset shape: {cleaned_data.shape}")
        
        # Main analysis sections
        if analysis_type == "Overview":
            show_overview(cleaned_data)
        elif analysis_type == "Correlation Analysis":
            show_correlation_analysis(cleaned_data)
        elif analysis_type == "Distribution Analysis":
            show_distribution_analysis(cleaned_data)
        elif analysis_type == "Relationship Analysis":
            show_relationship_analysis(cleaned_data)
        elif analysis_type == "Statistical Tests":
            show_statistical_tests(cleaned_data)
        elif analysis_type == "Predictive Insights":
            show_predictive_insights(cleaned_data)
            
    except FileNotFoundError as e:
        st.error("❌ Dataset not found! Please ensure the CSV file is in the correct location.")
        st.info("Expected path: data/Study_Hours_per_Week,Sleep_Hours.csv")
        st.error(f"Error details: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def show_overview(data):
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(data))
    with col2:
        st.metric("Average GPA", f"{data['GPA'].mean():.2f}")
    with col3:
        st.metric("Avg Study Hours/Week", f"{data['Study_Hours_per_Week'].mean():.1f}")
    with col4:
        st.metric("Avg Sleep Hours/Night", f"{data['Sleep_Hours_per_Night'].mean():.1f}")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.dataframe(data.describe())
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_gpa_distribution(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_study_hours_distribution(data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional overview charts
    st.subheader("📊 Overview Charts")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sleep_distribution(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_attendance_distribution(data)
        st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(data):
    st.header("🔗 Correlation Analysis")
    
    # Correlation heatmap
    fig = create_correlation_heatmap(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    corr_matrix = data.corr()
    gpa_correlations = corr_matrix['GPA'].sort_values(ascending=False)
    
    st.subheader("🎯 Key Correlations with GPA")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Positive Correlations:**")
        positive_corr = gpa_correlations[gpa_correlations > 0].drop('GPA')
        for var, corr in positive_corr.items():
            st.write(f"• {var}: {corr:.3f}")
    
    with col2:
        st.write("**Negative Correlations:**")
        negative_corr = gpa_correlations[gpa_correlations < 0]
        for var, corr in negative_corr.items():
            st.write(f"• {var}: {corr:.3f}")

def show_distribution_analysis(data):
    st.header("📊 Distribution Analysis")
    
    # Variable selection
    variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Screen_Time_per_Day', 'Attendance_Percentage', 'GPA']
    selected_var = st.selectbox("Select variable to analyze:", variables)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_histogram_with_kde(data, selected_var)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_box_plot(data, selected_var)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution statistics
    st.subheader(f"📈 {selected_var} Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{data[selected_var].mean():.2f}")
    with col2:
        st.metric("Median", f"{data[selected_var].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{data[selected_var].std():.2f}")
    with col4:
        st.metric("Range", f"{data[selected_var].max() - data[selected_var].min():.2f}")

def show_relationship_analysis(data):
    st.header("🔍 Relationship Analysis")
    
    # Scatter plots
    st.subheader("Study Hours vs GPA")
    fig = create_study_vs_gpa_scatter(data)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep Hours vs GPA")
        fig = create_sleep_vs_gpa_scatter(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Screen Time vs GPA")
        fig = create_screen_time_vs_gpa_scatter(data)
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Relationship: Study Hours, Sleep Hours, and GPA")
    fig = create_3d_scatter(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-variable analysis
    st.subheader("Multi-Variable Analysis")
    fig = create_multi_variable_analysis(data)
    st.plotly_chart(fig, use_container_width=True)

def show_statistical_tests(data):
    st.header("📊 Statistical Tests")
    
    # Performance categories
    data_copy = data.copy()
    data_copy['Performance_Category'] = pd.cut(data_copy['GPA'], 
                                               bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                               labels=['Low', 'Medium', 'High', 'Excellent'])
    
    # Display performance distribution
    st.subheader("📈 Performance Category Distribution")
    perf_counts = data_copy['Performance_Category'].value_counts()
    fig = px.pie(values=perf_counts.values, names=perf_counts.index, 
                 title="Distribution of Performance Categories")
    st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA tests
    st.subheader("🧪 ANOVA Tests (Performance Categories)")
    
    variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Screen_Time_per_Day', 'Attendance_Percentage']
    
    for var in variables:
        groups = [group[var].values for name, group in data_copy.groupby('Performance_Category') if len(group) > 0]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{var}**")
            with col2:
                st.write(f"F-statistic: {f_stat:.3f}")
            with col3:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.write(f"P-value: {p_value:.3f} ({significance})")

def show_predictive_insights(data):
    st.header("🔮 Predictive Insights")
    
    try:
        # Simple linear regression insights
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Prepare features
        features = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Screen_Time_per_Day', 'Attendance_Percentage']
        X = data[features]
        y = data['GPA']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.3f}")
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Coefficient', 
                     title='Feature Coefficients in GPA Prediction',
                     color='Coefficient', color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        # GPA Predictor
        st.subheader("🎯 GPA Predictor")
        st.write("Enter student habits to predict GPA:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            study_hours = st.slider("Study Hours/Week", 0.0, 30.0, 15.0)
        with col2:
            sleep_hours = st.slider("Sleep Hours/Night", 4.0, 12.0, 7.0)
        with col3:
            screen_time = st.slider("Screen Time/Day", 0.0, 10.0, 4.0)
        with col4:
            attendance = st.slider("Attendance %", 50.0, 100.0, 85.0)
        
        # Predict
        prediction_input = np.array([[study_hours, sleep_hours, screen_time, attendance]])
        predicted_gpa = model.predict(prediction_input)[0]
        
        # Display prediction with color coding
        if predicted_gpa >= 3.5:
            st.success(f"🎓 Predicted GPA: {predicted_gpa:.2f} (Excellent Performance)")
        elif predicted_gpa >= 3.0:
            st.info(f"📚 Predicted GPA: {predicted_gpa:.2f} (Good Performance)")
        elif predicted_gpa >= 2.5:
            st.warning(f"⚠️ Predicted GPA: {predicted_gpa:.2f} (Average Performance)")
        else:
            st.error(f"📉 Predicted GPA: {predicted_gpa:.2f} (Needs Improvement)")
            
    except ImportError:
        st.error("Scikit-learn is required for predictive analysis. Please install it using: pip install scikit-learn")

if __name__ == "__main__":
    main()