import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

class ChurnPredictor:
    """Employee Churn Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.threshold = 0.31  # Optimal threshold from analysis
        
    def create_model(self):
        """Create and train a RandomForest model based on the notebook analysis"""
        
        # Load the original dataset (you'll need to provide this)
        # For now, we'll create a mock model structure
        # In production, you would load your trained model from the notebook
        
        # Define preprocessing pipeline (same as in notebook)
        ordinal_col = ["education"]
        ordinal_categories = [["High School", "Diploma", "Bachelor"]]
        nominal_cols = ["gender", "work_location", "marital_status"]
        
        # Get numeric columns (all except categorical and target)
        all_cols = ['age', 'gender', 'education', 'experience_years', 'monthly_target',
                   'target_achievement', 'working_hours_per_week', 'overtime_hours_per_week',
                   'salary', 'commission_rate', 'job_satisfaction', 'work_location',
                   'manager_support_score', 'company_tenure_years', 'marital_status',
                   'distance_to_office_km', 'target_gap', 'overwork_ratio',
                   'tenure_per_age', 'income_per_hour', 'experience_to_tenure']
        
        numeric_cols = [col for col in all_cols if col not in ordinal_col + nominal_cols]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("edu", OrdinalEncoder(categories=ordinal_categories), ordinal_col),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nominal_cols),
                ("num", StandardScaler(), numeric_cols)
            ]
        )
        
        # Create RandomForest model with best parameters from notebook
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Create pipeline with SMOTE
        self.model = Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", rf_model)
        ])
        
        return self.model
    
    def preprocess_data(self, df):
        """Preprocess input data"""
        
        # Feature engineering (same as in notebook)
        df = df.copy()
        
        # Create engineered features
        df["target_gap"] = df["monthly_target"] - (df["target_achievement"] * df["monthly_target"])
        df["overwork_ratio"] = df["overtime_hours_per_week"] / df["working_hours_per_week"].replace(0, np.nan)
        df["tenure_per_age"] = df["company_tenure_years"] / df["age"].replace(0, np.nan)
        df["income_per_hour"] = df["salary"] / (df["working_hours_per_week"].replace(0, np.nan) * 4.3)
        df["experience_to_tenure"] = df["experience_years"] / (df["company_tenure_years"] + 1)
        
        # Handle NaN values more carefully
        # First, identify numeric and categorical columns
        numeric_columns = []
        categorical_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        # Fill numeric columns with median
        for col in numeric_columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Fill categorical columns with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def predict(self, df):
        """Make predictions on new data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call create_model() first.")
        
        # For demo purposes, create simple mock predictions
        # In production, you would use the actual preprocessing and model
        
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
        """Get feature importance from the trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # For demo purposes, return mock feature importance
        # In production, you would get this from the actual trained model
        
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

def create_mock_model():
    """Create a mock model for demonstration purposes"""
    # This is a simplified version for the dashboard demo
    # In production, you would load your actual trained model
    
    predictor = ChurnPredictor()
    
    # Create a simple mock model for demonstration
    # In reality, you would load your trained RandomForest model here
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate some mock data to train a simple model
    X_mock, y_mock = make_classification(
        n_samples=1000, 
        n_features=22,  # Number of features after preprocessing
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    # Create and train a simple RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Create simple pipeline without SMOTE for demo
    from sklearn.pipeline import Pipeline
    predictor.model = Pipeline([
        ("clf", rf_model)
    ])
    
    # Train the model
    predictor.model.fit(X_mock, y_mock)
    
    # Create a simple preprocessor for demo
    predictor.preprocessor = None  # Skip complex preprocessing for demo
    
    return predictor
