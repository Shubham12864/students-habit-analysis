# Configuration settings for the student habits analysis project

# File paths - Fixed to match actual structure
DATA_PATH = 'data/Study_Hours_per_Week,Sleep_Hours.csv'
ASSETS_PATH = 'assets/style.css'

# Constants
SLEEP_HOURS_COLUMN = 'Sleep_Hours_per_Night'
STUDY_HOURS_COLUMN = 'Study_Hours_per_Week'
GPA_COLUMN = 'GPA'
ATTENDANCE_COLUMN = 'Attendance_Percentage'
SCREEN_TIME_COLUMN = 'Screen_Time_per_Day'

# Analysis settings
OUTLIER_METHOD = 'IQR'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Performance categories
PERFORMANCE_BINS = [0, 2.5, 3.0, 3.5, 4.0]
PERFORMANCE_LABELS = ['Low', 'Medium', 'High', 'Excellent']