from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class PatientCase(Base):
    __tablename__ = 'patient_cases'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(Text)
    follow_up_questions = Column(Text)
    predicted_triage = Column(String(20))
    actual_triage = Column(String(20), nullable=True)
    confidence_score = Column(Float)
    processing_time = Column(Float)  # in seconds
    
    def __repr__(self):
        return f"<PatientCase(id={self.id}, triage={self.predicted_triage})>"

class DatabaseManager:
    def __init__(self, db_path="data/triage.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_case(self, symptoms, follow_up_questions, predicted_triage, confidence_score, processing_time):
        case = PatientCase(
            symptoms=symptoms,
            follow_up_questions=follow_up_questions,
            predicted_triage=predicted_triage,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
        self.session.add(case)
        self.session.commit()
        return case.id
    
    def update_actual_triage(self, case_id, actual_triage):
        case = self.session.query(PatientCase).get(case_id)
        if case:
            case.actual_triage = actual_triage
            self.session.commit()
            return True
        return False
    
    def get_case(self, case_id):
        return self.session.query(PatientCase).get(case_id)
    
    def get_all_cases(self):
        return self.session.query(PatientCase).all()
    
    def get_cases_by_triage(self, triage_level):
        return self.session.query(PatientCase).filter_by(predicted_triage=triage_level).all()
    
    def get_accuracy_metrics(self):
        cases = self.session.query(PatientCase).filter(PatientCase.actual_triage.isnot(None)).all()
        total = len(cases)
        if total == 0:
            return {"accuracy": 0, "total_cases": 0}
        
        correct = sum(1 for case in cases if case.predicted_triage == case.actual_triage)
        return {
            "accuracy": correct / total,
            "total_cases": total
        }
    
    def get_wait_time_stats(self):
        cases = self.session.query(PatientCase).all()
        if not cases:
            return {"avg_wait_time": 0, "total_cases": 0}
        
        avg_wait_time = sum(case.processing_time for case in cases) / len(cases)
        return {
            "avg_wait_time": avg_wait_time,
            "total_cases": len(cases)
        }
    
    def close(self):
        self.session.close() 