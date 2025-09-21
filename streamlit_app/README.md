# Employee Churn Prediction Dashboard

A comprehensive Streamlit dashboard for predicting employee churn risk and providing actionable recommendations.

## Features

### 🏠 Overview & Insights Page
- Model performance metrics (F2-Score: 0.789)
- Feature importance analysis
- Risk distribution visualization
- Business insights and recommendations
- Model bias analysis
- Sample data download

### 🔮 Employee Churn Predictor Page
- CSV file upload functionality
- Data validation and preview
- Real-time churn predictions
- Risk categorization (Loyal/At Risk/Potential Churn)
- Personalized recommendations
- Results export (CSV download)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Data Requirements
The predictor expects CSV files with the following columns:
- `age` (numeric)
- `gender` (Male/Female)
- `education` (High School/Diploma/Bachelor)
- `experience_years` (numeric)
- `monthly_target` (numeric)
- `target_achievement` (numeric, 0-2)
- `working_hours_per_week` (numeric)
- `overtime_hours_per_week` (numeric)
- `salary` (numeric)
- `commission_rate` (numeric)
- `job_satisfaction` (numeric, 1-10)
- `work_location` (Urban/Suburban/Rural)
- `manager_support_score` (numeric, 1-10)
- `company_tenure_years` (numeric)
- `marital_status` (Single/Married)
- `distance_to_office_km` (numeric)

### Sample Data
Download sample data from the Overview page or use the "Load Sample Data" button in the Predictor page.

## Model Information

- **Algorithm**: RandomForest Classifier
- **Performance**: F2-Score 0.789, Recall 100%, Precision 68%
- **Features**: 22 features (including 5 engineered features)
- **Threshold**: 0.31 (optimized for F2-Score)

### Engineered Features
1. `target_gap`: Difference between target and actual achievement
2. `overwork_ratio`: Overtime to regular hours ratio
3. `tenure_per_age`: Company loyalty indicator
4. `income_per_hour`: Compensation efficiency
5. `experience_to_tenure`: Experience vs company tenure ratio

## Risk Categories

- **Loyal** (probability < 0.31): Low churn risk
- **At Risk** (0.31 ≤ probability < 0.60): Medium churn risk
- **Potential Churn** (probability ≥ 0.60): High churn risk

## Recommendations Engine

The system provides personalized recommendations based on:
- Risk level and probability
- Top contributing factors
- Business rules and best practices
- Priority levels (High/Medium/Low)

## File Structure

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── pages/
│   ├── __init__.py
│   ├── overview.py        # Overview & insights page
│   └── predictor.py       # Prediction page
├── utils/
│   ├── __init__.py
│   ├── prediction.py      # Prediction utilities
│   └── recommendations.py # Recommendation engine
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technical Details

- Built with Streamlit for interactive web interface
- Uses scikit-learn for machine learning
- Plotly for interactive visualizations
- Imbalanced-learn for handling class imbalance
- SHAP for model explainability

## Business Value

This dashboard helps HR teams:
- Identify employees at risk of churning
- Understand key factors driving churn
- Get actionable recommendations for retention
- Monitor churn risk trends
- Make data-driven retention decisions

## Model Performance

The model achieves:
- **100% Recall**: Catches all potential churners
- **68% Precision**: Reasonable false positive rate
- **71% Accuracy**: Good overall performance
- **F2-Score 0.789**: Optimized for recall (critical for HR)

## Future Enhancements

- Real-time model updates
- Advanced visualization options
- Integration with HR systems
- Automated intervention workflows
- A/B testing for retention strategies
