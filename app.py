"""
Flask backend API for MAILLY - Live Spam Email Classification System.
Provides endpoints for email classification using trained neural network models.
"""

import os
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Import our modules
from preprocessing.text_cleaning import TextPreprocessor
from models.rnn_model import SimpleRNNModel
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
MODEL_DIR = 'saved_models'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for loaded models and preprocessor
loaded_models = {}
preprocessor = None
best_model_info = None


def load_best_model():
    """Load all model and identify best performing model."""
    global loaded_models, preprocessor, best_model_info
    
    try:
        # Load best model info
        best_model_info_path = os.path.join(MODEL_DIR, 'best_model_info.pkl')
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, 'rb') as f:
                best_model_info = pickle.load(f)
            
            best_model_name = best_model_info['best_model_name']
            
            print(f"Loading best model: {best_model_name}")
            
            # Load preprocessor
            preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                preprocessor = TextPreprocessor()
                preprocessor.load(preprocessor_path)
                print("Preprocessor loaded successfully")
            
            # Load all three models
            model_configs = {
                'vocab_size': len(preprocessor.word2idx) if preprocessor else 10000,
                'embedding_dim': preprocessor.embedding_dim if preprocessor else 100,
                'max_sequence_length': preprocessor.max_sequence_length if preprocessor else 100,
                'embedding_matrix': preprocessor.create_embedding_matrix() if preprocessor else None
            }
            
            # Load RNN
            try:
                rnn_path = os.path.join(MODEL_DIR, 'rnn_best.h5')
                if os.path.exists(rnn_path):
                    rnn_model = SimpleRNNModel(**model_configs)
                    rnn_model.load(rnn_path)
                    loaded_models['RNN'] = rnn_model
                    print("RNN model loaded successfully")
            except Exception as e:
                print(f"Could not load RNN model: {e}")
            
            # Load LSTM
            try:
                lstm_path = os.path.join(MODEL_DIR, 'lstm_best.h5')
                if os.path.exists(lstm_path):
                    lstm_model = LSTMModel(**model_configs)
                    lstm_model.load(lstm_path)
                    loaded_models['LSTM'] = lstm_model
                    print("LSTM model loaded successfully")
            except Exception as e:
                print(f"Could not load LSTM model: {e}")
            
            # Load GRU
            try:
                gru_path = os.path.join(MODEL_DIR, 'gru_best.h5')
                if os.path.exists(gru_path):
                    gru_model = GRUModel(**model_configs)
                    gru_model.load(gru_path)
                    loaded_models['GRU'] = gru_model
                    print("GRU model loaded successfully")
            except Exception as e:
                print(f"Could not load GRU model: {e}")
            
            print(f"Available models: {list(loaded_models.keys())}")
            return True
            
        else:
            print("No best model info found. Please train models first.")
            return False
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def classify_email(email_text, model_name=None):
    """
    Classify an email as spam or not spam.
    
    Args:
        email_text: Raw email text
        model_name: Optional model name (RNN, LSTM, GRU). If not specified, uses best model.
        
    Returns:
        Dictionary with prediction and confidence
    """
    global loaded_models, preprocessor, best_model_info
    
    if not loaded_models or not preprocessor:
        return {
            'error': 'Models not loaded. Please train models first.',
            'prediction': None,
            'confidence': None,
            'model_used': None
        }
    
    try:
        # Transform text using preprocessor
        processed_text = preprocessor.transform_single_text(email_text)
        
        # If model_name not specified, use best model
        if not model_name:
            model_name = best_model_info['best_model_name']
        
        # Check if requested model exists
        if model_name not in loaded_models:
            return {
                'error': f'Model {model_name} not available',
                'prediction': None,
                'confidence': None,
                'model_used': None,
                'available_models': list(loaded_models.keys())
            }
        
        model = loaded_models[model_name]
        
        # Make prediction with adjusted threshold for better spam detection
        # Using 0.45 threshold instead of 0.5 to be more sensitive to spam
        prediction_proba = model.predict(processed_text)[0]
        prediction = int(prediction_proba > 0.45)
        
        # Convert to human readable format
        result = {
            'prediction': 'spam' if prediction == 1 else 'not spam',
            'confidence': float(prediction_proba),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {
            'error': f'Error during classification: {str(e)}',
            'prediction': None,
            'confidence': None,
            'model_used': None
        }


@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        'message': 'MAILLY - Live Spam Email Classification API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Classify email as spam or not spam',
            '/models': 'GET - Get information about loaded models',
            '/health': 'GET - Check API health status'
        },
        'status': 'running'
    })


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(loaded_models) > 0,
        'preprocessor_loaded': preprocessor is not None
    })


@app.route('/models')
def get_models_info():
    """Get information about loaded models."""
    global best_model_info
    
    if best_model_info:
        return jsonify({
            'best_model': best_model_info['best_model_name'],
            'results': best_model_info['results'],
            'models_available': list(loaded_models.keys()),
            'preprocessor_loaded': preprocessor is not None
        })
    else:
        return jsonify({
            'error': 'No model information available',
            'message': 'Please train models first'
        }), 404


@app.route('/predict', methods=['POST'])
def predict_spam():
    """
    Predict if an email is spam or not spam.
    
    Expected JSON format:
    {
        "email": "Your email text here"
    }
    
    Returns:
    {
        "prediction": "spam" or "not spam",
        "confidence": 0.0 to 1.0,
        "model_used": "RNN/LSTM/GRU",
        "timestamp": "ISO timestamp"
    }
    """
    try:
        # Get email text from request
        data = request.get_json()
        
        if not data or 'email' not in data:
            return jsonify({
                'error': 'Please provide email text in JSON format',
                'example': {
                    'email': 'Your email text here'
                }
            }), 400
        
        email_text = data['email'].strip()
        model_name = data.get('model', None)  # Optional model parameter
        
        if not email_text:
            return jsonify({
                'error': 'Email text cannot be empty'
            }), 400
        
        # Classify email
        result = classify_email(email_text, model_name)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple emails at once.
    
    Expected JSON format:
    {
        "emails": ["email1 text", "email2 text", ...]
    }
    
    Returns:
    {
        "predictions": [
            {
                "prediction": "spam" or "not spam",
                "confidence": 0.0 to 1.0,
                "model_used": "RNN/LSTM/GRU",
                "timestamp": "ISO timestamp"
            },
            ...
        ]
    }
    """
    try:
        # Get emails from request
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({
                'error': 'Please provide emails list in JSON format',
                'example': {
                    'emails': ['email1 text', 'email2 text']
                }
            }), 400
        
        emails = data['emails']
        
        if not emails:
            return jsonify({
                'error': 'Emails list cannot be empty'
            }), 400
        
        # Process each email
        predictions = []
        for email_text in emails:
            email_text = email_text.strip()
            if email_text:
                result = classify_email(email_text)
                predictions.append(result)
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback for model predictions.
    
    Expected JSON format:
    {
        "email": "email text",
        "prediction": "spam" or "not spam",
        "actual": "spam" or "not spam",
        "confidence": 0.0 to 1.0
    }
    
    Returns:
    {
        "message": "Feedback received",
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Please provide feedback data in JSON format'
            }), 400
        
        # Log feedback (in a real application, this would be stored in a database)
        feedback_data = {
            'email': data.get('email', ''),
            'prediction': data.get('prediction', ''),
            'actual': data.get('actual', ''),
            'confidence': data.get('confidence', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        # In a production system, you would:
        # 1. Store this feedback in a database
        # 2. Use it to retrain models periodically
        # 3. Monitor model performance over time
        
        print(f"Feedback received: {feedback_data}")
        
        return jsonify({
            'message': 'Feedback received successfully',
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing feedback: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("Starting MAILLY API Server...")
    print("Loading best model...")
    
    # Load the best model on startup
    if load_best_model():
        print("Server ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please train models first.")
        print("Run: python training/train_models.py")