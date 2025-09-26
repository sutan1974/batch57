import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Try to import with matplotlib, fallback to simple version
try:
    from pages import overview, predictor
except ImportError:
    from pages import overview_simple as overview, predictor

# Page configuration
st.set_page_config(
    page_title="Employee Churn Prediction Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">👥 Employee Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Navigation tabs
tab1, tab2 = st.tabs(["🏠 Overview & Insights", "🔮 Employee Churn Predictor"])

with tab1:
    overview.show()

with tab2:
    predictor.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance:**")
st.sidebar.metric("F2-Score", "0.789")
st.sidebar.metric("Recall", "100%")
st.sidebar.metric("Precision", "68%")
st.sidebar.metric("Accuracy", "71%")
