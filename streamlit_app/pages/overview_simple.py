import streamlit as st
import pandas as pd
import numpy as np

def show():
    st.markdown("## 📈 Model Performance & Business Insights")
    
    # Key metrics from the analysis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 F2-Score",
            value="0.789",
            help="Optimized for recall to catch potential churners"
        )
    
    with col2:
        st.metric(
            label="📊 Total Employees",
            value="1,000",
            help="Dataset size used for training"
        )
    
    with col3:
        st.metric(
            label="⚠️ Churn Rate",
            value="62.9%",
            help="Percentage of employees who churned"
        )
    
    with col4:
        st.metric(
            label="🎯 Optimal Threshold",
            value="0.31",
            help="Probability threshold for risk categorization"
        )
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🔍 Top 10 Most Important Features")
    
    # Feature importance data from RandomForest analysis
    feature_importance = {
        'Feature': [
            'Target Achievement', 'Distance to Office (km)', 'Target Gap',
            'Job Satisfaction', 'Company Tenure (years)', 'Tenure per Age',
            'Working Hours/Week', 'Manager Support Score', 'Monthly Target',
            'Income per Hour'
        ],
        'Importance': [0.1085, 0.0820, 0.0713, 0.0655, 0.0628, 0.0621, 
                      0.0599, 0.0590, 0.0464, 0.0440]
    }
    
    df_importance = pd.DataFrame(feature_importance)
    
    # Display as a table instead of chart
    st.dataframe(df_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Distribution")
        
        # Risk categories based on threshold analysis
        risk_data = {
            'Category': ['Loyal', 'At Risk', 'Potential Churn'],
            'Count': [3, 8, 4],  # Based on test set results
            'Percentage': [20, 53, 27],
        }
        
        df_risk = pd.DataFrame(risk_data)
        st.dataframe(df_risk, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Model Performance Metrics")
        
        # Performance metrics
        metrics_data = {
            'Metric': ['F2-Score', 'F1-Score', 'Recall', 'Precision', 'Accuracy'],
            'Value': [0.789, 0.795, 1.000, 0.685, 0.710]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    st.markdown("---")
    
    # Key Business Insights
    st.markdown("### 💡 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Factors that INCREASE Churn Risk:")
        st.markdown("""
        - **Low Target Achievement** (10.85% importance)
        - **Long Distance to Office** (8.20% importance)
        - **Large Target Gap** (7.13% importance)
        - **Low Job Satisfaction** (6.55% importance)
        - **Short Company Tenure** (6.28% importance)
        - **High Working Hours** (5.99% importance)
        - **Low Manager Support** (5.90% importance)
        """)
    
    with col2:
        st.markdown("#### 🟢 Factors that DECREASE Churn Risk:")
        st.markdown("""
        - **High Target Achievement**
        - **Short Distance to Office**
        - **High Job Satisfaction**
        - **Long Company Tenure**
        - **Strong Manager Support**
        - **Good Work-Life Balance**
        - **Competitive Compensation**
        """)
    
    st.markdown("---")
    
    # Model Bias Analysis
    st.markdown("### ⚖️ Model Bias Analysis")
    
    bias_data = {
        'Demographic': ['Female', 'Male', 'Bachelor', 'Diploma', 'High School'],
        'F2-Score': [0.956, 0.896, 0.904, 0.923, 0.915],
        'Accuracy': [0.833, 0.657, 0.667, 0.732, 0.716]
    }
    
    df_bias = pd.DataFrame(bias_data)
    st.dataframe(df_bias, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    
    st.markdown("""
    #### 🚨 Immediate Actions:
    1. **Focus on Target Achievement**: Implement support programs for employees struggling with targets
    2. **Address Commute Issues**: Consider remote work options or relocation assistance
    3. **Improve Manager Support**: Train managers on employee engagement and support techniques
    
    #### 📈 Medium-term Initiatives:
    1. **Job Satisfaction Surveys**: Regular pulse checks to identify issues early
    2. **Career Development**: Clear progression paths to improve retention
    3. **Work-Life Balance**: Flexible working arrangements and workload management
    
    #### 🔄 Long-term Strategy:
    1. **Predictive Analytics**: Use this model for proactive retention efforts
    2. **Continuous Monitoring**: Regular model updates with new data
    3. **Intervention Programs**: Targeted retention strategies based on risk levels
    """)
    
    # Sample data download
    st.markdown("---")
    st.markdown("### 📥 Sample Data for Testing")
    
    if st.button("📥 Download Sample Employee Data"):
        # Generate sample data for testing
        sample_data = generate_sample_data()
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sample_employee_data.csv",
            mime="text/csv"
        )

def generate_sample_data():
    """Generate sample employee data for testing"""
    np.random.seed(42)
    n_samples = 20
    
    data = {
        'age': np.random.randint(22, 60, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'education': np.random.choice(['High School', 'Diploma', 'Bachelor'], n_samples),
        'experience_years': np.random.randint(1, 20, n_samples),
        'monthly_target': np.random.randint(50, 200, n_samples),
        'target_achievement': np.random.uniform(0.3, 1.2, n_samples),
        'working_hours_per_week': np.random.randint(35, 60, n_samples),
        'overtime_hours_per_week': np.random.randint(0, 20, n_samples),
        'salary': np.random.randint(30000, 120000, n_samples),
        'commission_rate': np.random.uniform(0.01, 0.15, n_samples),
        'job_satisfaction': np.random.randint(1, 10, n_samples),
        'work_location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'manager_support_score': np.random.randint(1, 10, n_samples),
        'company_tenure_years': np.random.uniform(0.5, 15, n_samples),
        'marital_status': np.random.choice(['Single', 'Married'], n_samples),
        'distance_to_office_km': np.random.randint(5, 100, n_samples)
    }
    
    return pd.DataFrame(data)
