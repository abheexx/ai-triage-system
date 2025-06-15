import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_triage_engine import LLMTriageEngine
from src.triage_model import TriageModel
from src.database import DatabaseManager

# Initialize session state
if 'current_case' not in st.session_state:
    st.session_state.current_case = None
if 'follow_up_answers' not in st.session_state:
    st.session_state.follow_up_answers = ""

# Initialize components
llm_engine = LLMTriageEngine()
triage_model = TriageModel()
db_manager = DatabaseManager()

def main():
    st.set_page_config(
        page_title="AI-Powered Patient Triage System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 AI-Powered Patient Triage System")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Patient Intake", "Analytics Dashboard"]
    )
    
    if page == "Patient Intake":
        show_patient_intake()
    else:
        show_analytics_dashboard()

def show_patient_intake():
    st.header("Patient Intake Form")
    
    # Initial symptoms input
    symptoms = st.text_area(
        "Please describe the patient's symptoms:",
        height=150,
        help="Enter the patient's main symptoms and any relevant details"
    )
    
    if st.button("Analyze Symptoms"):
        if symptoms:
            with st.spinner("Analyzing symptoms..."):
                # Process the case
                case_result = llm_engine.process_case(symptoms)
                
                # Get triage prediction
                prediction = triage_model.predict(case_result["symptom_summary"])
                
                # Store in session state
                st.session_state.current_case = {
                    "symptoms": symptoms,
                    "follow_up_questions": case_result["follow_up_questions"],
                    "symptom_summary": case_result["symptom_summary"],
                    "prediction": prediction,
                    "processing_time": case_result["processing_time"]
                }
                
                # Save to database
                case_id = db_manager.add_case(
                    symptoms=symptoms,
                    follow_up_questions=case_result["follow_up_questions"],
                    predicted_triage=prediction["triage_level"],
                    confidence_score=prediction["confidence_score"],
                    processing_time=case_result["processing_time"]
                )
                
                st.session_state.current_case["id"] = case_id
        else:
            st.warning("Please enter the patient's symptoms.")
    
    # Display current case if available
    if st.session_state.current_case:
        st.subheader("Analysis Results")
        
        # Display follow-up questions
        st.markdown("### Follow-up Questions")
        st.write(st.session_state.current_case["follow_up_questions"])
        
        # Follow-up answers input
        st.session_state.follow_up_answers = st.text_area(
            "Enter follow-up answers:",
            value=st.session_state.follow_up_answers,
            height=100
        )
        
        # Display symptom summary
        st.markdown("### Symptom Summary")
        st.write(st.session_state.current_case["symptom_summary"])
        
        # Display triage prediction
        st.markdown("### Triage Prediction")
        prediction = st.session_state.current_case["prediction"]
        
        # Color-coded triage level
        triage_colors = {
            "Urgent": "red",
            "Moderate": "orange",
            "Low": "green"
        }
        
        st.markdown(
            f"<h3 style='color: {triage_colors[prediction['triage_level']]};'>"
            f"Triage Level: {prediction['triage_level']}</h3>",
            unsafe_allow_html=True
        )
        
        # Display confidence score
        st.progress(prediction["confidence_score"])
        st.write(f"Confidence: {prediction['confidence_score']:.2%}")
        
        # Display processing time
        st.write(f"Processing Time: {st.session_state.current_case['processing_time']:.2f} seconds")
        
        # Actual triage input (for training data)
        actual_triage = st.selectbox(
            "Actual Triage Level (for training):",
            ["Urgent", "Moderate", "Low"]
        )
        
        if st.button("Update Actual Triage"):
            db_manager.update_actual_triage(
                st.session_state.current_case["id"],
                actual_triage
            )
            st.success("Actual triage level updated!")

def show_analytics_dashboard():
    st.header("Analytics Dashboard")
    
    # Get metrics
    accuracy_metrics = db_manager.get_accuracy_metrics()
    wait_time_stats = db_manager.get_wait_time_stats()
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Accuracy",
            f"{accuracy_metrics['accuracy']:.2%}",
            f"Total Cases: {accuracy_metrics['total_cases']}"
        )
    
    with col2:
        st.metric(
            "Average Processing Time",
            f"{wait_time_stats['avg_wait_time']:.2f}s",
            f"Total Cases: {wait_time_stats['total_cases']}"
        )
    
    # Get all cases
    cases = db_manager.get_all_cases()
    
    if cases:
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': case.timestamp,
            'triage_level': case.predicted_triage,
            'confidence': case.confidence_score,
            'processing_time': case.processing_time
        } for case in cases])
        
        # Triage level distribution
        st.subheader("Triage Level Distribution")
        fig = px.pie(
            df,
            names='triage_level',
            title='Distribution of Triage Levels'
        )
        st.plotly_chart(fig)
        
        # Processing time over time
        st.subheader("Processing Time Trends")
        fig = px.line(
            df,
            x='timestamp',
            y='processing_time',
            title='Processing Time Over Time'
        )
        st.plotly_chart(fig)
        
        # Confidence scores by triage level
        st.subheader("Confidence Scores by Triage Level")
        fig = px.box(
            df,
            x='triage_level',
            y='confidence',
            title='Confidence Score Distribution by Triage Level'
        )
        st.plotly_chart(fig)
    else:
        st.info("No cases available for analysis yet.")

if __name__ == "__main__":
    main() 