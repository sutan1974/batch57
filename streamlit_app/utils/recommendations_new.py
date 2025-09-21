import pandas as pd
import numpy as np

class SimpleRecommendationEngine:
    """Simple recommendation engine for demo purposes"""
    
    def __init__(self):
        self.recommendation_templates = {
            'target_achievement': {
                'high_risk': [
                    "Set more realistic and achievable targets",
                    "Provide additional training and support for performance improvement",
                    "Schedule regular check-ins to monitor progress"
                ],
                'medium_risk': [
                    "Monitor target achievement closely",
                    "Provide coaching and mentoring support"
                ],
                'low_risk': [
                    "Maintain current performance levels",
                    "Continue regular performance reviews"
                ]
            },
            'distance_to_office': {
                'high_risk': [
                    "Consider remote work arrangements",
                    "Provide relocation assistance or transportation benefits",
                    "Explore flexible working hours to reduce commute stress"
                ],
                'medium_risk': [
                    "Discuss flexible working arrangements",
                    "Provide transportation benefits or carpooling options"
                ],
                'low_risk': [
                    "Monitor commute satisfaction",
                    "Maintain current arrangements"
                ]
            },
            'job_satisfaction': {
                'high_risk': [
                    "Schedule immediate one-on-one meeting to address concerns",
                    "Conduct detailed satisfaction survey",
                    "Review workload and work-life balance"
                ],
                'medium_risk': [
                    "Schedule regular check-ins",
                    "Provide additional support and resources"
                ],
                'low_risk': [
                    "Continue current engagement strategies",
                    "Maintain regular communication"
                ]
            },
            'manager_support_score': {
                'high_risk': [
                    "Provide manager training on employee support",
                    "Implement mentorship program",
                    "Review management practices and feedback"
                ],
                'medium_risk': [
                    "Provide additional management training",
                    "Implement regular feedback sessions"
                ],
                'low_risk': [
                    "Maintain current management practices",
                    "Continue regular team meetings"
                ]
            },
            'company_tenure_years': {
                'high_risk': [
                    "Provide career advancement opportunities",
                    "Offer additional responsibilities and challenges",
                    "Implement retention bonus or benefits"
                ],
                'medium_risk': [
                    "Discuss career goals and opportunities",
                    "Provide additional training and development"
                ],
                'low_risk': [
                    "Continue current development programs",
                    "Maintain engagement initiatives"
                ]
            },
            'working_hours_per_week': {
                'high_risk': [
                    "Review and optimize workload distribution",
                    "Implement flexible working hours",
                    "Provide additional resources or support"
                ],
                'medium_risk': [
                    "Monitor workload and stress levels",
                    "Provide additional support when needed"
                ],
                'low_risk': [
                    "Maintain current work arrangements",
                    "Continue monitoring workload"
                ]
            }
        }
    
    def get_top_risk_factors(self, row, feature_importance):
        """Get top 3 risk factors for an employee based on their data"""
        
        # Define risk thresholds for each feature
        risk_thresholds = {
            'target_achievement': {'low': 0.8, 'high': 0.6},
            'distance_to_office_km': {'low': 20, 'high': 50},
            'job_satisfaction': {'low': 7, 'high': 4},
            'manager_support_score': {'low': 7, 'high': 4},
            'company_tenure_years': {'low': 5, 'high': 2},
            'working_hours_per_week': {'low': 40, 'high': 50}
        }
        
        risk_factors = []
        
        # Get top 10 features from importance
        top_features = feature_importance.head(10)
        
        for idx, feature_row in top_features.iterrows():
            feature_name = feature_row['Feature']
            
            if feature_name in risk_thresholds:
                value = row.get(feature_name, 0)
                thresholds = risk_thresholds[feature_name]
                
                # Determine risk level
                if value <= thresholds['high']:
                    risk_level = 'high_risk'
                elif value <= thresholds['low']:
                    risk_level = 'medium_risk'
                else:
                    risk_level = 'low_risk'
                
                # Only include high and medium risk factors
                if risk_level in ['high_risk', 'medium_risk']:
                    risk_factors.append({
                        'factor': feature_name,
                        'value': value,
                        'risk_level': risk_level,
                        'importance': feature_row['Importance']
                    })
        
        # Sort by importance and return top 3
        risk_factors.sort(key=lambda x: x['importance'], reverse=True)
        return risk_factors[:3]
    
    def generate_recommendations(self, employee_data, predictions, feature_importance):
        """Generate personalized recommendations for each employee"""
        
        recommendations = []
        
        for idx, (_, row) in enumerate(employee_data.iterrows()):
            prob = predictions['probabilities'][idx]
            risk_category = predictions['risk_categories'][idx]
            
            # Get top risk factors
            top_factors = self.get_top_risk_factors(row, feature_importance)
            
            # Generate recommendations based on risk factors
            employee_recommendations = []
            priority = "Low"
            
            if risk_category == "Potential Churn":
                priority = "High"
            elif risk_category == "At Risk":
                priority = "Medium"
            
            # Add recommendations based on top risk factors
            for factor in top_factors:
                factor_name = factor['factor']
                risk_level = factor['risk_level']
                
                if factor_name in self.recommendation_templates:
                    if risk_level in self.recommendation_templates[factor_name]:
                        factor_recommendations = self.recommendation_templates[factor_name][risk_level]
                        employee_recommendations.extend(factor_recommendations[:2])  # Take top 2
                    else:
                        # Fallback to medium_risk
                        if 'medium_risk' in self.recommendation_templates[factor_name]:
                            factor_recommendations = self.recommendation_templates[factor_name]['medium_risk']
                            employee_recommendations.extend(factor_recommendations[:2])
                        else:
                            employee_recommendations.append(f"Review {factor_name} for this employee")
                else:
                    employee_recommendations.append(f"Review {factor_name} for this employee")
            
            # Add general recommendations based on risk category
            if risk_category == "Potential Churn":
                employee_recommendations.extend([
                    "Schedule immediate HR meeting",
                    "Implement retention intervention plan",
                    "Consider salary review or benefits enhancement"
                ])
            elif risk_category == "At Risk":
                employee_recommendations.extend([
                    "Increase monitoring frequency",
                    "Provide additional support and resources"
                ])
            else:
                employee_recommendations.extend([
                    "Continue current engagement strategies",
                    "Maintain regular performance reviews"
                ])
            
            # Remove duplicates and limit to 5 recommendations
            unique_recommendations = list(dict.fromkeys(employee_recommendations))[:5]
            
            recommendations.append({
                'employee_id': idx + 1,
                'churn_probability': prob,
                'risk_category': risk_category,
                'priority': priority,
                'top_risk_factors': [f['factor'] for f in top_factors],
                'recommendations': unique_recommendations
            })
        
        return recommendations
    
    def create_recommendations_dataframe(self, recommendations):
        """Convert recommendations to DataFrame for display"""
        
        data = []
        for rec in recommendations:
            for i, recommendation in enumerate(rec['recommendations']):
                data.append({
                    'Employee ID': rec['employee_id'],
                    'Risk Category': rec['risk_category'],
                    'Priority': rec['priority'],
                    'Churn Probability': f"{rec['churn_probability']:.3f}",
                    'Top Risk Factors': ', '.join(rec['top_risk_factors']),
                    'Recommendation': recommendation,
                    'Action Order': i + 1
                })
        
        return pd.DataFrame(data)

def get_priority_color(priority):
    """Get color for priority level"""
    colors = {
        'High': '#d62728',    # Red
        'Medium': '#ff7f0e',  # Orange
        'Low': '#2ca02c'      # Green
    }
    return colors.get(priority, '#1f77b4')

def get_risk_category_color(category):
    """Get color for risk category"""
    colors = {
        'Potential Churn': '#d62728',  # Red
        'At Risk': '#ff7f0e',          # Orange
        'Loyal': '#2ca02c'             # Green
    }
    return colors.get(category, '#1f77b4')
