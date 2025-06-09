import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}")

def clean_data(data):
    """Clean the dataset by handling missing values and correcting data types."""
    # Make a copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = cleaned_data.dropna()
    
    # Convert columns to appropriate data types
    numeric_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                      'Screen_Time_per_Day', 'Attendance_Percentage', 'GPA']
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # Remove any rows with invalid data
    cleaned_data = cleaned_data.dropna()
    
    # Remove outliers using IQR method
    for col in numeric_columns:
        if col in cleaned_data.columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & 
                                      (cleaned_data[col] <= upper_bound)]
    
    return cleaned_data

def analyze_data(data):
    """Perform comprehensive statistical analysis on the dataset."""
    analysis_results = {
        'basic_stats': {
            'total_students': len(data),
            'mean_study_hours': data['Study_Hours_per_Week'].mean(),
            'median_study_hours': data['Study_Hours_per_Week'].median(),
            'std_study_hours': data['Study_Hours_per_Week'].std(),
            'mean_sleep_hours': data['Sleep_Hours_per_Night'].mean(),
            'median_sleep_hours': data['Sleep_Hours_per_Night'].median(),
            'std_sleep_hours': data['Sleep_Hours_per_Night'].std(),
            'mean_screen_time': data['Screen_Time_per_Day'].mean(),
            'median_screen_time': data['Screen_Time_per_Day'].median(),
            'std_screen_time': data['Screen_Time_per_Day'].std(),
            'mean_attendance': data['Attendance_Percentage'].mean(),
            'median_attendance': data['Attendance_Percentage'].median(),
            'std_attendance': data['Attendance_Percentage'].std(),
            'mean_gpa': data['GPA'].mean(),
            'median_gpa': data['GPA'].median(),
            'std_gpa': data['GPA'].std(),
        },
        'correlations': correlation_matrix(data),
        'performance_categories': categorize_performance(data)
    }
    return analysis_results

def correlation_matrix(data):
    """Calculate the correlation matrix for the dataset."""
    numeric_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                      'Screen_Time_per_Day', 'Attendance_Percentage', 'GPA']
    return data[numeric_columns].corr()

def categorize_performance(data):
    """Categorize students based on GPA performance."""
    data_copy = data.copy()
    data_copy['Performance_Category'] = pd.cut(data_copy['GPA'], 
                                               bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                               labels=['Low', 'Medium', 'High', 'Excellent'])
    
    performance_stats = data_copy.groupby('Performance_Category').agg({
        'Study_Hours_per_Week': ['mean', 'std', 'count'],
        'Sleep_Hours_per_Night': ['mean', 'std'],
        'Screen_Time_per_Day': ['mean', 'std'],
        'Attendance_Percentage': ['mean', 'std']
    }).round(2)
    
    return performance_stats

def statistical_tests(data):
    """Perform various statistical tests."""
    results = {}
    
    # Correlation tests
    variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                'Screen_Time_per_Day', 'Attendance_Percentage']
    
    for var in variables:
        corr_coef, p_value = stats.pearsonr(data[var], data['GPA'])
        results[f'{var}_vs_GPA'] = {
            'correlation': corr_coef,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results

def identify_top_performers(data, top_percent=10):
    """Identify characteristics of top performers."""
    top_threshold = data['GPA'].quantile(1 - top_percent/100)
    top_performers = data[data['GPA'] >= top_threshold]
    
    characteristics = {
        'count': len(top_performers),
        'avg_study_hours': top_performers['Study_Hours_per_Week'].mean(),
        'avg_sleep_hours': top_performers['Sleep_Hours_per_Night'].mean(),
        'avg_screen_time': top_performers['Screen_Time_per_Day'].mean(),
        'avg_attendance': top_performers['Attendance_Percentage'].mean(),
        'min_gpa': top_performers['GPA'].min(),
        'max_gpa': top_performers['GPA'].max()
    }
    
    return characteristics

def generate_insights(data):
    """Generate key insights from the data analysis."""
    insights = []
    
    # Correlation insights
    corr_matrix = correlation_matrix(data)
    gpa_correlations = corr_matrix['GPA'].drop('GPA').sort_values(key=abs, ascending=False)
    
    strongest_positive = gpa_correlations[gpa_correlations > 0].index[0] if any(gpa_correlations > 0) else None
    strongest_negative = gpa_correlations[gpa_correlations < 0].index[0] if any(gpa_correlations < 0) else None
    
    if strongest_positive:
        insights.append(f"Strongest positive correlation with GPA: {strongest_positive} ({gpa_correlations[strongest_positive]:.3f})")
    
    if strongest_negative:
        insights.append(f"Strongest negative correlation with GPA: {strongest_negative} ({gpa_correlations[strongest_negative]:.3f})")
    
    # Performance insights
    high_performers = data[data['GPA'] >= 3.5]
    low_performers = data[data['GPA'] < 2.5]
    
    if len(high_performers) > 0 and len(low_performers) > 0:
        study_diff = high_performers['Study_Hours_per_Week'].mean() - low_performers['Study_Hours_per_Week'].mean()
        sleep_diff = high_performers['Sleep_Hours_per_Night'].mean() - low_performers['Sleep_Hours_per_Night'].mean()
        
        insights.append(f"High performers study {study_diff:.1f} more hours per week on average")
        insights.append(f"High performers sleep {sleep_diff:.1f} more hours per night on average")
    
    return insights