import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

try:
    from data_analysis import load_data, clean_data
    from visualizations import *
except ImportError:
    # Fallback imports if modules don't exist
    pass

# Page config
st.set_page_config(
    page_title="Student Habits vs Academic Performance Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_fallback(file_path):
    """Fallback data loading function"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data_fallback(df):
    """Fallback data cleaning function"""
    if df is None:
        return None
    # Basic cleaning
    df = df.dropna()
    return df

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

def create_scatter_plot(data, x_col, y_col, color_col=None, size_col=None):
    """Create scatter plot"""
    fig = px.scatter(
        data, 
        x=x_col, 
        y=y_col,
        color=color_col,
        size=size_col,
        title=f'{x_col} vs {y_col}',
        hover_data=data.columns.tolist()
    )
    return fig

def create_histogram(data, column):
    """Create histogram with KDE"""
    fig = px.histogram(
        data, 
        x=column, 
        marginal="kde",
        title=f'Distribution of {column}',
        nbins=20
    )
    return fig

def create_box_plot(data, column):
    """Create box plot"""
    fig = px.box(
        data, 
        y=column,
        title=f'Box Plot of {column}'
    )
    return fig

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
        .stSelectbox > div > div > select {
            background-color: #ffffff;
        }
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #3498db, #2980b9);
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
        # Try different paths to find the data file
        possible_paths = [
            "data/Study_Hours_per_Week,Sleep_Hours.csv",
            "../data/Study_Hours_per_Week,Sleep_Hours.csv",
            os.path.join(parent_dir, "data", "Study_Hours_per_Week,Sleep_Hours.csv"),
            os.path.join(os.path.dirname(parent_dir), "data", "Study_Hours_per_Week,Sleep_Hours.csv")
        ]
        
        data = None
        data_path = None
        
        for path in possible_paths:
            try:
                # Try custom load_data function first, then fallback
                try:
                    data = load_data(path)
                except:
                    data = load_data_fallback(path)
                    
                if data is not None:
                    data_path = path
                    st.sidebar.success(f"✅ Data loaded from: {path}")
                    break
            except:
                continue
        
        if data is None:
            st.error("❌ Dataset not found! Please ensure the CSV file is in the data/ directory.")
            st.stop()
        
        # Clean data
        try:
            cleaned_data = clean_data(data)
        except:
            cleaned_data = clean_data_fallback(data)
        
        if cleaned_data is None:
            st.error("❌ Failed to clean data!")
            st.stop()
        
        # Sidebar options
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Correlation Analysis", "Distribution Analysis", 
             "Relationship Analysis", "Statistical Tests", "Predictive Insights",
             "Complete EDA Report"]
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
        elif analysis_type == "Complete EDA Report":
            show_complete_eda_report(cleaned_data)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check that all required files are in place and dependencies are installed.")

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
        fig = create_histogram(data, 'GPA')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_histogram(data, 'Study_Hours_per_Week')
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
        fig = create_histogram(data, selected_var)
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
    fig = create_scatter_plot(data, 'Study_Hours_per_Week', 'GPA', 'Attendance_Percentage', 'Sleep_Hours_per_Night')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep Hours vs GPA")
        fig = create_scatter_plot(data, 'Sleep_Hours_per_Night', 'GPA')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Screen Time vs GPA")
        fig = create_scatter_plot(data, 'Screen_Time_per_Day', 'GPA')
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Relationship: Study Hours, Sleep Hours, and GPA")
    fig = px.scatter_3d(
        data, 
        x='Study_Hours_per_Week', 
        y='Sleep_Hours_per_Night', 
        z='GPA',
        color='Attendance_Percentage',
        size='Screen_Time_per_Day',
        title='3D Relationship Analysis'
    )
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

def show_complete_eda_report(data):
    """Show complete EDA report"""
    st.header("📊 Complete Exploratory Data Analysis Report")
    
    with st.spinner("🔄 Generating comprehensive analysis..."):
        try:
            # Basic Information
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(data))
            with col2:
                st.metric("Variables", len(data.columns))
            with col3:
                st.metric("Average GPA", f"{data['GPA'].mean():.2f}")
            with col4:
                st.metric("Missing Values", data.isnull().sum().sum())
            
            # Show first few rows
            st.subheader("🔍 Sample Data")
            st.dataframe(data.head())
            
            # Correlation Analysis
            st.subheader("🔗 Correlation Analysis")
            correlation_matrix = data.corr()
            gpa_correlations = correlation_matrix['GPA'].sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Correlations with GPA:**")
                for var, corr in gpa_correlations.items():
                    if var != 'GPA':
                        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                        direction = "↗️" if corr > 0 else "↘️"
                        st.write(f"{direction} {var}: {corr:.3f} ({strength})")
            
            with col2:
                # Correlation heatmap
                fig = create_correlation_heatmap(data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance Analysis
            st.subheader("🎯 Performance Analysis")
            
            # Create performance categories
            data_copy = data.copy()
            data_copy['Performance_Category'] = pd.cut(data_copy['GPA'], 
                                                     bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                                     labels=['Low', 'Medium', 'High', 'Excellent'])
            
            performance_counts = data_copy['Performance_Category'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                # Performance distribution
                fig = px.pie(values=performance_counts.values, names=performance_counts.index,
                           title="Performance Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance comparison
                high_performers = data[data['GPA'] >= 3.5]
                low_performers = data[data['GPA'] < 2.5]
                
                if len(high_performers) > 0 and len(low_performers) > 0:
                    st.write("**High vs Low Performers:**")
                    variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                               'Screen_Time_per_Day', 'Attendance_Percentage']
                    
                    comparison_data = []
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
            
            # Key Insights
            st.subheader("💡 Key Insights & Recommendations")
            
            # Find strongest factor
            gpa_corr_abs = correlation_matrix['GPA'].drop('GPA').abs().sort_values(ascending=False)
            strongest_factor = gpa_corr_abs.index[0]
            strongest_corr = correlation_matrix['GPA'][strongest_factor]
            
            insights = []
            insights.append(f"🏆 **{strongest_factor.replace('_', ' ')}** has the strongest relationship with GPA (r = {strongest_corr:.3f})")
            
            screen_corr = correlation_matrix['GPA']['Screen_Time_per_Day']
            if screen_corr < -0.2:
                insights.append(f"📱 Screen time negatively impacts GPA (r = {screen_corr:.3f})")
            
            sleep_corr = correlation_matrix['GPA']['Sleep_Hours_per_Night']
            if sleep_corr > 0.2:
                insights.append(f"😴 Adequate sleep positively impacts GPA (r = {sleep_corr:.3f})")
            
            for insight in insights:
                st.write(insight)
            
            # Recommendations
            st.subheader("📋 Recommendations")
            if len(high_performers) > 0:
                st.success("**Success Formula (based on high performers):**")
                st.write(f"• 📖 Study: {high_performers['Study_Hours_per_Week'].mean():.1f}+ hours/week")
                st.write(f"• 😴 Sleep: {high_performers['Sleep_Hours_per_Night'].mean():.1f}+ hours/night")
                st.write(f"• 🎯 Attendance: {high_performers['Attendance_Percentage'].mean():.1f}%+")
                st.write(f"• 📱 Screen Time: ≤{high_performers['Screen_Time_per_Day'].mean():.1f} hours/day")
            
            st.success("✅ Complete EDA analysis generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating EDA report: {str(e)}")

if __name__ == "__main__":
    main()