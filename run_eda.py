"""
Standalone script to run complete EDA without Jupyter notebook
Run this script to generate a complete analysis report
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.eda_analysis import StudentHabitsEDA

def main():
    print("🚀 Student Habits Analysis - Standalone EDA")
    print("=" * 50)
    
    # Run the complete analysis
    eda = StudentHabitsEDA()
    success = eda.run_complete_analysis()
    
    if success:
        print("\n🎯 Analysis completed successfully!")
        print("📊 Check the generated PNG files for visualizations")
        print("🔗 Run 'streamlit run src/app.py' for interactive dashboard")
    else:
        print("\n❌ Analysis failed. Please check the error messages.")

if __name__ == "__main__":
    main()