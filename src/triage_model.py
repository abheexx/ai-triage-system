import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class TriageModel:
    def __init__(self, model_path="models/triage_model.joblib"):
        self.model_path = model_path
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["Urgent", "Moderate", "Low"])
        
        # Initialize the model pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        # Load model if it exists
        if os.path.exists(model_path):
            self.load_model()
    
    def preprocess_text(self, text):
        """Preprocess the text for model input."""
        # In a real system, this would include more sophisticated preprocessing
        return text.lower()
    
    def predict(self, symptom_summary):
        """Predict triage level from symptom summary."""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(symptom_summary)
            
            # Get prediction probabilities
            probas = self.model.predict_proba([processed_text])[0]
            
            # Get the predicted class
            predicted_class = self.model.predict([processed_text])[0]
            
            # Get confidence score (probability of predicted class)
            confidence_score = probas[np.argmax(probas)]
            
            return {
                "triage_level": predicted_class,
                "confidence_score": float(confidence_score),
                "probabilities": {
                    level: float(prob) for level, prob in zip(
                        self.label_encoder.classes_,
                        probas
                    )
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "triage_level": "Moderate",  # Default to moderate in case of error
                "confidence_score": 0.0
            }
    
    def train(self, X, y):
        """Train the model on new data."""
        try:
            # Encode labels
            y_encoded = self.label_encoder.transform(y)
            
            # Train the model
            self.model.fit(X, y_encoded)
            
            # Save the model
            self.save_model()
            
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def load_model(self):
        """Load the model from disk."""
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        try:
            y_pred = self.model.predict(X_test)
            y_true = self.label_encoder.transform(y_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_true)
            
            return {
                "accuracy": accuracy,
                "predictions": self.label_encoder.inverse_transform(y_pred),
                "true_labels": y_test
            }
        except Exception as e:
            return {"error": str(e)} 