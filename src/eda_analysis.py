"""
Comprehensive Exploratory Data Analysis Script
This script performs all the analysis that would typically be in a Jupyter notebook
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_analysis import load_data, clean_data
import visualizations as viz

warnings.filterwarnings('ignore')

class StudentHabitsEDA:
    def __init__(self, data_path=None):
        """Initialize the EDA class"""
        self.data_path = data_path or "../data/Study_Hours_per_Week,Sleep_Hours.csv"
        self.df = None
        self.cleaned_data = None
        self.correlation_matrix = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("🔄 Loading and preparing data...")
        
        # Try multiple paths
        possible_paths = [
            self.data_path,
            "data/Study_Hours_per_Week,Sleep_Hours.csv",
            "../data/Study_Hours_per_Week,Sleep_Hours.csv",
            os.path.join(os.path.dirname(current_dir), "data", "Study_Hours_per_Week,Sleep_Hours.csv")
        ]
        
        for path in possible_paths:
            try:
                self.df = pd.read_csv(path)
                print(f"✅ Data loaded successfully from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if self.df is None:
            raise FileNotFoundError("❌ Dataset not found in any expected location!")
        
        self.cleaned_data = clean_data(self.df)
        self.correlation_matrix = self.cleaned_data.corr()
        
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("📊 BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"\n🎯 Total students: {len(self.df)}")
        print(f"📈 Variables: {len(self.df.columns)}")
        
        print("\n📋 First 5 rows:")
        print(self.df.head())
        
        print("\n📊 Dataset Info:")
        print(self.df.info())
        
        print("\n📈 Summary Statistics:")
        print(self.df.describe())
        
        # Missing values
        missing_values = self.df.isnull().sum()
        print(f"\n🔍 Missing values: {missing_values.sum()}")
        if missing_values.sum() > 0:
            print(missing_values)
        
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Display correlations with GPA
        gpa_correlations = self.correlation_matrix['GPA'].sort_values(ascending=False)
        print("\n🎯 Correlations with GPA:")
        print("-" * 40)
        
        for var, corr in gpa_correlations.items():
            if var != 'GPA':
                strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                direction = "Positive" if corr > 0 else "Negative"
                print(f"{var:25s}: {corr:6.3f} ({strength} {direction})")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    square=True, center=0, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap of Student Habits and Academic Performance', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def distribution_analysis(self):
        """Analyze distributions of all variables"""
        print("\n" + "="*60)
        print("📊 DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        variables = self.df.columns
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
        
        for i, var in enumerate(variables):
            row = i // 3
            col = i % 3
            
            sns.histplot(self.df[var], kde=True, ax=axes[row, col], 
                        color=colors[i], alpha=0.7, bins=15)
            axes[row, col].set_title(f'Distribution of {var}', fontsize=14)
            axes[row, col].set_xlabel(var, fontsize=12)
            axes[row, col].grid(alpha=0.3)
            
            # Add mean and median lines
            mean_val = self.df[var].mean()
            median_val = self.df[var].median()
            axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                                  alpha=0.8, label=f'Mean: {mean_val:.2f}')
            axes[row, col].axvline(median_val, color='green', linestyle='--', 
                                  alpha=0.8, label=f'Median: {median_val:.2f}')
            axes[row, col].legend()
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        plt.suptitle('Distribution Analysis of All Variables', fontsize=16)
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def performance_analysis(self):
        """Analyze performance categories"""
        print("\n" + "="*60)
        print("🎯 PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Create performance categories
        df_copy = self.df.copy()
        df_copy['Performance_Category'] = pd.cut(df_copy['GPA'], 
                                                bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                                labels=['Low', 'Medium', 'High', 'Excellent'])
        
        performance_counts = df_copy['Performance_Category'].value_counts()
        performance_pct = (performance_counts / len(df_copy) * 100).round(1)
        
        print("\n📊 Performance Category Distribution:")
        print("-" * 35)
        for category, count in performance_counts.items():
            pct = performance_pct[category]
            print(f"{category:12s}: {count:3d} students ({pct:5.1f}%)")
        
        # Performance comparison
        high_performers = self.df[self.df['GPA'] >= 3.5]
        low_performers = self.df[self.df['GPA'] < 2.5]
        
        if len(high_performers) > 0 and len(low_performers) > 0:
            print("\n📋 Average Habits Comparison (High vs Low Performers):")
            print("-" * 55)
            
            variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                        'Screen_Time_per_Day', 'Attendance_Percentage']
            for var in variables:
                high_avg = high_performers[var].mean()
                low_avg = low_performers[var].mean()
                diff = high_avg - low_avg
                print(f"{var:25s}: High: {high_avg:5.1f} | Low: {low_avg:5.1f} | Diff: {diff:+5.1f}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors_pie = ['lightcoral', 'gold', 'lightgreen', 'skyblue']
        ax1.pie(performance_counts.values, labels=performance_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax1.set_title('Performance Category Distribution', fontsize=14)
        
        # Bar chart
        performance_counts.plot(kind='bar', ax=ax2, color=colors_pie)
        ax2.set_title('Performance Category Counts', fontsize=14)
        ax2.set_xlabel('Performance Category')
        ax2.set_ylabel('Number of Students')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def statistical_testing(self):
        """Perform statistical significance tests"""
        print("\n" + "="*60)
        print("🧪 STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        variables = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 
                    'Screen_Time_per_Day', 'Attendance_Percentage']
        
        # Correlation significance tests
        print("\n📊 Correlation Significance Tests:")
        print("-" * 35)
        
        for var in variables:
            corr_coef, p_value = stats.pearsonr(self.df[var], self.df['GPA'])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"{var:25s}: r={corr_coef:6.3f}, p={p_value:.3f} ({significance})")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # Normality tests
        print("\n🔍 Normality Tests (Shapiro-Wilk):")
        print("-" * 32)
        for var in variables + ['GPA']:
            stat, p_value = stats.shapiro(self.df[var])
            normal = "Normal" if p_value > 0.05 else "Not Normal"
            print(f"{var:25s}: W={stat:.3f}, p={p_value:.3f} ({normal})")
    
    def generate_insights(self):
        """Generate key insights and recommendations"""
        print("\n" + "="*70)
        print("🎯 KEY INSIGHTS AND RECOMMENDATIONS")
        print("="*70)
        
        # Find strongest relationships
        gpa_corr_abs = self.correlation_matrix['GPA'].drop('GPA').abs().sort_values(ascending=False)
        strongest_factor = gpa_corr_abs.index[0]
        strongest_corr = self.correlation_matrix['GPA'][strongest_factor]
        
        print("\n📋 Key Findings:")
        print("-" * 15)
        print(f"1. 🏆 {strongest_factor} shows the strongest relationship with GPA (r = {strongest_corr:.3f})")
        
        # Specific insights
        screen_corr = self.correlation_matrix['GPA']['Screen_Time_per_Day']
        if screen_corr < -0.2:
            print(f"2. 📱 Screen time negatively correlates with GPA (r = {screen_corr:.3f})")
        
        sleep_corr = self.correlation_matrix['GPA']['Sleep_Hours_per_Night']
        if sleep_corr > 0.2:
            print(f"3. 😴 Sleep hours positively correlate with GPA (r = {sleep_corr:.3f})")
        
        # Recommendations
        print("\n💡 Evidence-Based Recommendations:")
        print("-" * 35)
        
        high_performers = self.df[self.df['GPA'] >= 3.5]
        if len(high_performers) > 0:
            print("• 📖 Increase weekly study hours for better performance")
            print("• 🎯 Maintain consistent class attendance (>90%)")
            print("• 😴 Prioritize adequate sleep (7-8 hours)")
            print("• 📱 Limit recreational screen time")
            
            print(f"\n🏆 Success Formula (based on high performers):")
            print(f"   Study Hours: {high_performers['Study_Hours_per_Week'].mean():.1f}+ hours/week")
            print(f"   Sleep: {high_performers['Sleep_Hours_per_Night'].mean():.1f}+ hours/night")
            print(f"   Attendance: {high_performers['Attendance_Percentage'].mean():.1f}%+")
            print(f"   Screen Time: ≤{high_performers['Screen_Time_per_Day'].mean():.1f} hours/day")
    
    def run_complete_analysis(self):
        """Run the complete EDA pipeline"""
        print("🚀 Starting Comprehensive Student Habits Analysis")
        print("="*60)
        
        try:
            self.load_and_prepare_data()
            self.basic_info()
            self.correlation_analysis()
            self.distribution_analysis()
            self.performance_analysis()
            self.statistical_testing()
            self.generate_insights()
            
            print("\n✅ Analysis Complete!")
            print("📊 Generated visualizations:")
            print("   - correlation_heatmap.png")
            print("   - distributions.png") 
            print("   - performance_analysis.png")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return False
        
        return True

# Main execution
if __name__ == "__main__":
    # Run the complete analysis
    eda = StudentHabitsEDA()
    success = eda.run_complete_analysis()
    
    if success:
        print("\n🎯 Ready for presentation!")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")