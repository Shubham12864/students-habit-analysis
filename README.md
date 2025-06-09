# Student Habits Analysis

This project analyzes the relationship between student habits and academic performance using a dataset that includes study hours, sleep hours, screen time, attendance percentage, and GPA.

## Project Structure

```
student-habits-analysis
├── src
│   ├── app.py                # Main entry point for the Streamlit web application
│   ├── data_analysis.py      # Functions for loading, cleaning, and analyzing data
│   ├── visualizations.py      # Functions for creating visualizations
│   └── utils.py              # Utility functions for data manipulation
├── data
│   └── raw                   # Directory for raw dataset files
├── notebooks
│   └── exploratory_analysis.ipynb  # Jupyter notebook for exploratory data analysis
├── assets
│   └── style.css             # Custom CSS for the Streamlit app
├── requirements.txt          # List of required Python libraries
├── config.py                 # Configuration settings for the project
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd student-habits-analysis
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data/raw` directory.

## Usage

To run the Streamlit web application, execute the following command in your terminal:
```
streamlit run src/app.py
```

This will launch the application in your default web browser, where you can explore the analysis and visualizations of student habits versus academic performance.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.