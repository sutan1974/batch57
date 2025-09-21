import pandas as pd
import numpy as np

class SimpleChurnPredictor:
    """Simple Employee Churn Prediction Model for Demo"""
    
    def __init__(self):
        self.threshold = 0.31  # Optimal threshold from analysis
        
    def predict(self, df):
        """Make predictions on new data using simple rules"""
        
        n_samples = len(df)
        
        # Create mock probabilities based on some simple rules
        probabilities = []
        predictions = []
        risk_categories = []
        
        for idx, row in df.iterrows():
            # Simple rule-based probability calculation for demo
            prob = 0.3  # Base probability
            
            # Adjust based on some features
            if row.get('job_satisfaction', 5) < 5:
                prob += 0.2
            if row.get('manager_support_score', 5) < 5:
                prob += 0.15
            if row.get('target_achievement', 0.8) < 0.7:
                prob += 0.2
            if row.get('distance_to_office_km', 20) > 30:
                prob += 0.1
            if row.get('company_tenure_years', 3) < 2:
                prob += 0.15
            
            # Add some randomness
            prob += np.random.normal(0, 0.1)
            prob = max(0, min(1, prob))  # Clamp between 0 and 1
            
            probabilities.append(prob)
            predictions.append(1 if prob > self.threshold else 0)
            
            # Categorize risk levels
            if prob < self.threshold:
                risk_categories.append("Loyal")
            elif prob < 0.60:
                risk_categories.append("At Risk")
            else:
                risk_categories.append("Potential Churn")
        
        return {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'risk_categories': risk_categories
        }
    
    def get_feature_importance(self):
        """Get mock feature importance for demo"""
        
        feature_importance = {
            'Feature': [
                'target_achievement', 'distance_to_office_km', 'target_gap',
                'job_satisfaction', 'company_tenure_years', 'tenure_per_age',
                'working_hours_per_week', 'manager_support_score', 'monthly_target',
                'income_per_hour', 'overwork_ratio', 'experience_to_tenure',
                'salary', 'commission_rate', 'age', 'experience_years',
                'overtime_hours_per_week', 'education', 'work_location_Urban',
                'work_location_Suburban', 'gender_Male', 'marital_status_Single'
            ],
            'Importance': [0.1085, 0.0820, 0.0713, 0.0655, 0.0628, 0.0621, 
                          0.0599, 0.0590, 0.0464, 0.0440, 0.0402, 0.0397,
                          0.0404, 0.0336, 0.0392, 0.0272, 0.0348, 0.0168,
                          0.0101, 0.0097, 0.0086, 0.0383]
        }
        
        importance_df = pd.DataFrame(feature_importance).sort_values('Importance', ascending=False)
        
        return importance_df

def validate_input_data(df):
    """Validate input data format and requirements"""
    
    required_columns = [
        'age', 'gender', 'education', 'experience_years', 'monthly_target',
        'target_achievement', 'working_hours_per_week', 'overtime_hours_per_week',
        'salary', 'commission_rate', 'job_satisfaction', 'work_location',
        'manager_support_score', 'company_tenure_years', 'marital_status',
        'distance_to_office_km'
    ]
    
    errors = []
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {list(missing_cols)}")
    
    # Check data types
    numeric_cols = ['age', 'experience_years', 'monthly_target', 'target_achievement',
                   'working_hours_per_week', 'overtime_hours_per_week', 'salary',
                   'commission_rate', 'job_satisfaction', 'manager_support_score',
                   'company_tenure_years', 'distance_to_office_km']
    
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric")
    
    # Check categorical values
    categorical_checks = {
        'gender': ['Male', 'Female'],
        'education': ['High School', 'Diploma', 'Bachelor'],
        'work_location': ['Urban', 'Suburban', 'Rural'],
        'marital_status': ['Single', 'Married']
    }
    
    for col, valid_values in categorical_checks.items():
        if col in df.columns:
            invalid_values = set(df[col].unique()) - set(valid_values)
            if invalid_values:
                errors.append(f"Column '{col}' contains invalid values: {list(invalid_values)}")
    
    # Check value ranges
    range_checks = {
        'age': (18, 65),
        'job_satisfaction': (1, 10),
        'manager_support_score': (1, 10),
        'target_achievement': (0, 2)
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            if df[col].min() < min_val or df[col].max() > max_val:
                errors.append(f"Column '{col}' values must be between {min_val} and {max_val}")
    
    return errors

def create_simple_predictor():
    """Create a simple predictor for demo purposes"""
    return SimpleChurnPredictor()
