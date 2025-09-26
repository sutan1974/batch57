import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
from utils.prediction_new import SimpleChurnPredictor, validate_input_data, create_simple_predictor
from utils.recommendations_new import SimpleRecommendationEngine, get_priority_color, get_risk_category_color

def show():
    st.markdown("## 🔮 Employee Churn Predictor")
    st.markdown("Upload employee data to predict churn risk and get personalized recommendations.")
    
    # Clear cache button for development
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear Cache & Reload"):
            st.cache_data.clear()
            st.session_state.predictor = None
            st.session_state.predictions = None
            st.session_state.recommendations = None
            st.rerun()
    
    with col2:
        if st.button("🆕 Force Refresh"):
            import importlib
            import sys
            # Force reload the modules
            if 'utils.prediction_new' in sys.modules:
                importlib.reload(sys.modules['utils.prediction_new'])
            if 'utils.recommendations_new' in sys.modules:
                importlib.reload(sys.modules['utils.recommendations_new'])
            st.session_state.predictor = None
            st.session_state.predictions = None
            st.session_state.recommendations = None
            st.rerun()
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    # Model info section
    st.markdown("### 📊 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "RandomForest")
    with col2:
        st.metric("F2-Score", "0.789")
    with col3:
        st.metric("Threshold", "0.31")
    with col4:
        st.metric("Features", "22")
    
    # Required columns info
    with st.expander("📋 Required Columns"):
        required_cols = [
            "age", "gender", "education", "experience_years",
            "monthly_target", "target_achievement", "working_hours_per_week",
            "overtime_hours_per_week", "salary", "commission_rate",
            "job_satisfaction", "work_location", "manager_support_score",
            "company_tenure_years", "marital_status", "distance_to_office_km"
        ]
        
        col1, col2, col3 = st.columns(3)
        for i, col in enumerate(required_cols):
            if i < 6:
                col1.markdown(f"• {col}")
            elif i < 12:
                col2.markdown(f"• {col}")
            else:
                col3.markdown(f"• {col}")
    
    # File upload section
    st.markdown("### 📁 Upload Employee Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with employee data. Download sample data from the Overview page."
        )
    
    with col2:
        st.markdown("**Or use sample data:**")
        if st.button("📥 Load Sample Data"):
            sample_data = generate_sample_data()
            st.session_state.uploaded_data = sample_data
            st.success("Sample data loaded!")
            st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.success(f"✅ File uploaded successfully! {len(df)} employees loaded.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return
    
    # Display uploaded data
    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data validation
        st.markdown("### ✅ Data Validation")
        validation_errors = validate_input_data(df)
        
        if validation_errors:
            st.error("❌ Data validation failed:")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            st.success("✅ Data validation passed!")
            
            # Prediction section
            st.markdown("### 🔮 Make Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Predictions", type="primary"):
                    with st.spinner("Running predictions..."):
                        # Initialize predictor if not already done
                        if st.session_state.predictor is None:
                            st.session_state.predictor = create_simple_predictor()
                        
                        # Make predictions
                        predictions = st.session_state.predictor.predict(df)
                        st.session_state.predictions = predictions
                        
                        # Generate recommendations
                        feature_importance = st.session_state.predictor.get_feature_importance()
                        rec_engine = SimpleRecommendationEngine()
                        recommendations = rec_engine.generate_recommendations(
                            df, predictions, feature_importance
                        )
                        st.session_state.recommendations = recommendations
                        
                        st.success("✅ Predictions completed!")
                        st.rerun()
            
            with col2:
                if st.button("📊 View Results"):
                    if st.session_state.predictions is not None:
                        st.rerun()
            
            with col3:
                if st.button("🔄 Reset"):
                    st.session_state.predictions = None
                    st.session_state.recommendations = None
                    st.rerun()
    
    # Display results
    if st.session_state.predictions is not None and st.session_state.recommendations is not None:
        display_results(st.session_state.predictions, st.session_state.recommendations, df)

def display_results(predictions, recommendations, original_data):
    """Display prediction results and recommendations"""
    
    st.markdown("---")
    st.markdown("## 📈 Prediction Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        loyal_count = sum(1 for cat in predictions['risk_categories'] if cat == 'Loyal')
        st.metric("Loyal Employees", loyal_count)
    
    with col2:
        at_risk_count = sum(1 for cat in predictions['risk_categories'] if cat == 'At Risk')
        st.metric("At Risk", at_risk_count)
    
    with col3:
        churn_count = sum(1 for cat in predictions['risk_categories'] if cat == 'Potential Churn')
        st.metric("Potential Churn", churn_count)
    
    with col4:
        avg_prob = np.mean(predictions['probabilities'])
        st.metric("Avg Churn Probability", f"{avg_prob:.3f}")
    
    # Risk distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Distribution")
        risk_counts = pd.Series(predictions['risk_categories']).value_counts()
        
        # Create pie chart using matplotlib
        plt.figure(figsize=(8, 6))
        colors = ['#2ca02c', '#ff7f0e', '#d62728']
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Employee Risk Distribution')
        st.pyplot(plt)
    
    with col2:
        st.markdown("### 📈 Probability Distribution")
        # Create histogram using matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(predictions['probabilities'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(x=0.31, color='red', linestyle='--', linewidth=2, label='Threshold: 0.31')
        plt.xlabel('Churn Probability')
        plt.ylabel('Count')
        plt.title('Churn Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    # Create results DataFrame
    results_data = []
    for idx, (_, row) in enumerate(original_data.iterrows()):
        results_data.append({
            'Employee ID': idx + 1,
            'Age': row.get('age', 'N/A'),
            'Gender': row.get('gender', 'N/A'),
            'Education': row.get('education', 'N/A'),
            'Job Satisfaction': row.get('job_satisfaction', 'N/A'),
            'Company Tenure (years)': row.get('company_tenure_years', 'N/A'),
            'Churn Probability': f"{predictions['probabilities'][idx]:.3f}",
            'Risk Category': predictions['risk_categories'][idx],
            'Prediction': 'Churn' if predictions['predictions'][idx] == 1 else 'Stay'
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Add color coding for risk categories
    def color_risk_category(val):
        colors = {
            'Loyal': 'background-color: #d4edda',
            'At Risk': 'background-color: #fff3cd',
            'Potential Churn': 'background-color: #f8d7da'
        }
        return colors.get(val, '')
    
    styled_df = results_df.style.applymap(color_risk_category, subset=['Risk Category'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
    
    # Recommendations section - HIDDEN as requested
    # st.markdown("---")
    # st.markdown("## 💡 Actionable Recommendations")
    # 
    # # Create recommendations DataFrame
    # rec_engine = SimpleRecommendationEngine()
    # rec_df = rec_engine.create_recommendations_dataframe(recommendations)
    # 
    # # Filter by priority
    # priority_filter = st.selectbox(
    #     "Filter by Priority:",
    #     ["All", "High", "Medium", "Low"]
    # )
    # 
    # if priority_filter != "All":
    #     rec_df_filtered = rec_df[rec_df['Priority'] == priority_filter]
    # else:
    #     rec_df_filtered = rec_df
    # 
    # # Display recommendations
    # st.dataframe(rec_df_filtered, use_container_width=True)
    # 
    # # Summary by employee
    # st.markdown("### 👥 Employee Summary")
    # 
    # for rec in recommendations[:5]:  # Show first 5 employees
    #     with st.expander(f"Employee {rec['employee_id']} - {rec['risk_category']} (Priority: {rec['priority']})"):
    #         st.markdown(f"**Churn Probability:** {rec['churn_probability']:.3f}")
    #         st.markdown(f"**Top Risk Factors:** {', '.join(rec['top_risk_factors'])}")
    #         st.markdown("**Recommendations:**")
    #         for i, recommendation in enumerate(rec['recommendations'], 1):
    #             st.markdown(f"{i}. {recommendation}")
    # 
    # # Download recommendations
    # rec_csv = rec_df.to_csv(index=False)
    # st.download_button(
    #     label="📥 Download Recommendations as CSV",
    #     data=rec_csv,
    #     file_name="churn_recommendations.csv",
    #     mime="text/csv"
    # )

def generate_sample_data():
    """Generate sample employee data for testing"""
    np.random.seed(42)
    n_samples = 15
    
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
