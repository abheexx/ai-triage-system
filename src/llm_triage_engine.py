import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

class LLMTriageEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = """You are an experienced medical triage assistant. Your role is to:
1. Analyze patient symptoms
2. Ask relevant follow-up questions
3. Help determine the urgency level of the case
4. Maintain a professional and empathetic tone

Focus on gathering key information about:
- Severity of symptoms
- Duration of symptoms
- Any relevant medical history
- Current medications
- Vital signs (if available)"""

    def analyze_symptoms(self, initial_symptoms):
        """Analyze initial symptoms and generate follow-up questions."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Patient reports the following symptoms: {initial_symptoms}\n\nPlease analyze these symptoms and provide 2-3 relevant follow-up questions to better assess the situation."}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing symptoms: {str(e)}"

    def process_follow_up(self, symptoms, follow_up_answers):
        """Process follow-up answers and generate a symptom summary."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Initial symptoms: {symptoms}\nFollow-up answers: {follow_up_answers}\n\nPlease provide a concise summary of the patient's condition, focusing on key symptoms and their severity."}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing follow-up: {str(e)}"

    def generate_triage_summary(self, symptom_summary):
        """Generate a structured summary for the ML model."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical triage expert. Extract key information from the symptom summary and format it for ML processing."},
                    {"role": "user", "content": f"Symptom summary: {symptom_summary}\n\nPlease extract and format the following information:\n1. Main symptoms\n2. Severity indicators\n3. Risk factors\n4. Time sensitivity"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating triage summary: {str(e)}"

    def process_case(self, initial_symptoms):
        """Process a complete case from initial symptoms to final summary."""
        start_time = time.time()
        
        # Step 1: Analyze initial symptoms and get follow-up questions
        follow_up_questions = self.analyze_symptoms(initial_symptoms)
        
        # Step 2: Process follow-up answers (in a real system, this would be interactive)
        # For demo purposes, we'll use a placeholder
        follow_up_answers = "Patient reports symptoms have been present for 2 days. No relevant medical history. Not currently taking any medications."
        
        # Step 3: Generate symptom summary
        symptom_summary = self.process_follow_up(initial_symptoms, follow_up_answers)
        
        # Step 4: Generate structured summary for ML
        ml_summary = self.generate_triage_summary(symptom_summary)
        
        processing_time = time.time() - start_time
        
        return {
            "follow_up_questions": follow_up_questions,
            "symptom_summary": symptom_summary,
            "ml_summary": ml_summary,
            "processing_time": processing_time
        } 