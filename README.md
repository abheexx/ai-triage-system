# AI-Powered Patient Intake Triage System

An intelligent patient triage system that combines Large Language Models (LLMs) with machine learning to provide accurate and efficient patient assessment and prioritization.

## Overview

This system streamlines the patient intake process by:
1. Collecting patient symptoms through an intuitive web interface
2. Using LLMs to interpret symptoms and gather relevant follow-up information
3. Employing machine learning to predict triage levels (Urgent/Moderate/Low)
4. Maintaining a comprehensive database of cases for analysis
5. Providing real-time analytics and performance metrics

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM Integration**: OpenAI GPT-4
- **Machine Learning**: Scikit-learn (Random Forest)
- **Database**: SQLite
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Streamlit

## Project Structure

```
ai_triage_system/
├── src/
│   ├── llm_triage_engine.py    # LLM interaction and symptom analysis
│   ├── triage_model.py         # ML model for triage prediction
│   ├── database.py            # Database operations
│   ├── app.py                 # Main Streamlit application
│   └── utils.py              # Utility functions
├── tests/                    # Unit tests
├── data/                     # Training data and database
├── models/                   # Saved ML models
├── static/                   # Static assets
└── templates/               # HTML templates
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-triage-system.git
cd ai-triage-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

5. Run the application:
```bash
streamlit run src/app.py
```

## Sample Output

The system provides:
- Real-time triage level predictions
- Detailed symptom analysis
- Historical performance metrics
- Wait time analytics
- Accuracy reports

## Inspiration

This project was inspired by the challenges faced in hospital emergency departments where efficient patient triage is crucial for:
- Reducing wait times
- Prioritizing critical cases
- Optimizing resource allocation
- Improving patient outcomes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 