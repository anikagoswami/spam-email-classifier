# MAILLY - Live Spam Email Classification System

A complete end-to-end machine learning project for spam email detection using neural networks. MAILLY provides a modern web interface for classifying emails as spam or not spam using trained deep learning models.

## 🚀 Features

### Neural Network Models
- **Simple RNN**: Basic recurrent neural network for sequence classification
- **LSTM**: Long Short-Term Memory network for capturing long-term dependencies
- **GRU**: Gated Recurrent Unit for efficient sequence modeling

### Backend API
- **Flask REST API**: Fast and scalable backend for model serving
- **Real-time Classification**: Instant email classification via API endpoints
- **Model Management**: Automatic loading of best performing model
- **Health Monitoring**: API health checks and model status

### Frontend Interface
- **Modern Email Client UI**: Beautiful interface named "Mailly"
- **Live Classification**: Paste email text and get instant results
- **Visual Feedback**: Color-coded results with confidence bars
- **Analytics Dashboard**: View model performance and classification history
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Model Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score
- **Confusion Matrices**: Detailed performance visualization
- **Training Visualizations**: Loss and accuracy plots over epochs
- **Model Comparison**: Side-by-side performance comparison
- **Performance Reports**: Detailed analysis and recommendations

## 🎨 Design & Color Palette

MAILLY uses a carefully designed color palette:

- **Background**: `#E4D5B7` (Warm beige)
- **Accent Yellow**: `#FDB01C` (Vibrant yellow)
- **Spam Red**: `#990000` (Alert red)
- **Highlight Cyan**: `#71E8F0` (Bright cyan)
- **Primary Teal**: `#13545A` (Dark teal)

## 📋 Project Structure

```
Mailly-Spam-Classifier/
├── dataset/                    # Email dataset (CSV format)
├── preprocessing/              # Text preprocessing modules
│   └── text_cleaning.py       # Text cleaning and Word2Vec
├── models/                     # Neural network model implementations
│   ├── rnn_model.py           # Simple RNN model
│   ├── lstm_model.py          # LSTM model
│   └── gru_model.py           # GRU model
├── training/                   # Training pipeline
│   └── train_models.py        # Complete training script
├── evaluation/                 # Model evaluation modules
│   ├── confusion_matrix.py    # Confusion matrix generation
│   ├── metrics.py             # Detailed metrics computation
│   └── model_comparison.py    # Model comparison visualization
├── visualizations/             # Generated plots and charts
├── saved_models/               # Trained model files
├── frontend/                   # Web interface
│   ├── index.html             # Main HTML page
│   ├── style.css              # CSS styling
│   └── script.js              # JavaScript functionality
├── app.py                      # Flask backend API
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Mailly-Spam-Classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your spam email dataset in the `dataset/` folder. The dataset should be a CSV file with the following format:

```csv
text,label
"Your email text here...","spam"
"Another email text...","ham"
```

**Expected columns:**
- `text`: The email content
- `label`: Either "spam" or "ham" (not spam)

### 3. Train Models

```bash
# Train all models (RNN, LSTM, GRU)
python training/train_models.py
```

This will:
- Load and preprocess the dataset
- Train all three neural network models
- Save the best performing model
- Generate evaluation visualizations
- Create performance reports

## 🚀 Usage

### 1. Start the Backend API

```bash
# Start the Flask server
python app.py
```

The API will be available at `http://localhost:5000`

### 2. Open the Frontend

Open `frontend/index.html` in your web browser to access the MAILLY interface.

### 3. Classify Emails

1. Navigate to the "Classify Email" section
2. Paste any email text in the input box
3. Click "Classify Email"
4. View the results with confidence scores

## 📊 API Endpoints

### Classification Endpoints

#### POST `/predict`
Classify a single email as spam or not spam.

**Request:**
```json
{
    "email": "Your email text here..."
}
```

**Response:**
```json
{
    "prediction": "spam" or "not spam",
    "confidence": 0.85,
    "model_used": "LSTM",
    "timestamp": "2023-10-15T10:30:00"
}
```

#### POST `/predict/batch`
Classify multiple emails at once.

**Request:**
```json
{
    "emails": ["email1 text", "email2 text", ...]
}
```

**Response:**
```json
{
    "predictions": [
        {
            "prediction": "spam",
            "confidence": 0.92,
            "model_used": "LSTM",
            "timestamp": "2023-10-15T10:30:00"
        }
    ],
    "count": 1
}
```

### Information Endpoints

#### GET `/models`
Get information about loaded models and their performance.

#### GET `/health`
Check API health status.

#### GET `/`
Get API information and available endpoints.

## 🎯 Model Performance

The system automatically trains and compares three neural network models:

1. **Simple RNN**: Basic recurrent network
2. **LSTM**: Long Short-Term Memory with bidirectional support
3. **GRU**: Gated Recurrent Unit with bidirectional support

After training, the system:
- Evaluates all models on a test set
- Selects the best performing model (by F1 score)
- Automatically loads the best model for API serving
- Generates comprehensive performance reports

## 📈 Evaluation & Visualization

The system generates several types of visualizations:

### Training Visualizations
- Training and validation accuracy over epochs
- Training and validation loss over epochs
- Saved in `visualizations/` directory

### Model Evaluation
- Confusion matrices for each model
- Precision-Recall curves
- ROC curves
- Classification reports

### Model Comparison
- Performance comparison bar charts
- Radar charts showing all metrics
- Detailed performance analysis
- Recommendations for model selection

## 🔧 Customization

### Adding New Models
1. Create a new model class in the `models/` directory
2. Follow the existing pattern (build_model, train, predict methods)
3. Add the model to the training pipeline in `training/train_models.py`

### Customizing Frontend
- Modify `frontend/style.css` for styling changes
- Update `frontend/script.js` for new functionality
- Edit `frontend/index.html` for layout changes

### Model Configuration
Adjust model parameters in the training script:
- Vocabulary size
- Embedding dimensions
- Sequence length
- Training epochs
- Batch size

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure your dataset is in the `dataset/` folder
   - Check that the file is named correctly in the training script

2. **Memory Issues**
   - Reduce vocabulary size in `text_cleaning.py`
   - Decrease batch size in training parameters
   - Use smaller embedding dimensions

3. **API Not Responding**
   - Ensure models are trained before starting the API
   - Check that the Flask server is running
   - Verify port 5000 is not in use

4. **Frontend Not Loading**
   - Open `index.html` directly in browser (no server needed)
   - Ensure API is running on `localhost:5000`
   - Check browser console for errors

### Performance Optimization

- Use GPU for training (TensorFlow will auto-detect)
- Adjust model complexity based on your dataset size
- Consider using pre-trained embeddings for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Uses TensorFlow/Keras for neural network implementations
- NLTK for text preprocessing
- Scikit-learn for evaluation metrics
- Seaborn and Matplotlib for visualizations
- Modern CSS Grid and Flexbox for responsive design

## 📞 Support

For questions, issues, or feature requests:

1. Check the [Issues](../../issues) section
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

---

**MAILLY** - Smart Email Security for the Modern Web