"""
Text preprocessing module for spam email classification.
Handles text cleaning, tokenization, and Word2Vec embedding generation.
"""

import re
import numpy as np
import pandas as pd
import nltk
import os
import pickle
from scipy.stats import fit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Handles all text preprocessing operations for email classification."""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100, embedding_dim=100):
        """
        Initialize the preprocessor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_sequence_length: Maximum length of padded sequences
            embedding_dim: Dimension of Word2Vec embeddings
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.stop_words = set(stopwords.words('english'))
        self.word2idx = {}
        self.idx2word = {}
        
        
    def clean_text(self, text):
        """
        Clean and preprocess a single email text.
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned and tokenized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """
        Tokenize text and remove stopwords.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of tokens without stopwords
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of email texts
        """
        word_count = {}
        
        for text in texts:
            tokens = self.tokenize_and_remove_stopwords(self.clean_text(text))
            for token in tokens:
                word_count[token] = word_count.get(token, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_count, key=word_count.get, reverse=True)
        vocab_words = sorted_words[:self.max_vocab_size - 1]  # -1 for padding token
        
        # Create mappings
        self.word2idx = {'<PAD>': 0}
        for idx, word in enumerate(vocab_words, 1):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of indices.
        
        Args:
            texts: List of email texts
            
        Returns:
            List of sequences (each sequence is a list of indices)
        """
        sequences = []
        
        for text in texts:
            tokens = self.tokenize_and_remove_stopwords(self.clean_text(text))
            sequence = [self.word2idx.get(token, 0) for token in tokens]  # 0 for unknown words
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences_data(self, sequences):
        """
        Pad sequences to have the same length.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Padded sequences as numpy array
        """
        return pad_sequences(sequences, maxlen=self.max_sequence_length, 
                           padding='post', truncating='post')
    
   
    def create_embedding_matrix(self):
        """
        Create simple random embedding matrix 
        
        """
        vocab_size = len(self.word2idx)
        embedding_matrix = np.random.normal(
        scale=0.6,
        size=(vocab_size, self.embedding_dim)
          )

        embedding_matrix[0] = np.zeros(self.embedding_dim)
        
        return embedding_matrix
    
    def preprocess_data(self, texts, labels=None, fit=True):
        """
        Complete preprocessing pipeline.
        """

        if fit:
            self.build_vocabulary(texts)

        sequences = self.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences_data(sequences)

        if labels is not None:
            return padded_sequences, np.array(labels)

        return padded_sequences
    
    def save(self, filepath):
        """Save preprocessor state to file."""
        state = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'max_vocab_size': self.max_vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.word2idx = state['word2idx']
        self.idx2word = state['idx2word']
        self.max_vocab_size = state['max_vocab_size']
        self.max_sequence_length = state['max_sequence_length']
        self.embedding_dim = state['embedding_dim']
        
        
        print(f"Preprocessor loaded from {filepath}")
    
    def transform_single_text(self, text):
        """
        Transform a single text for prediction.
        
        Args:
            text: Raw email text
            
        Returns:
            Padded sequence ready for model prediction
        """
        tokens = self.tokenize_and_remove_stopwords(self.clean_text(text))
        sequence = [self.word2idx.get(token, 0) for token in tokens]
        padded_sequence = pad_sequences([sequence], maxlen=self.max_sequence_length, 
                                       padding='post', truncating='post')
        return padded_sequence


def load_dataset(dataset_path):
    """
    Load spam email dataset from CSV file.
    
    Expected format: CSV with 'text' and 'label' columns
    where label is 'spam' or 'ham'
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        texts: List of email texts
        labels: List of labels (0 for ham, 1 for spam)
    """
    df = pd.read_csv(dataset_path, encoding="latin-1", sep=",",on_bad_lines="skip")
    
    # Convert labels to binary
    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower().map({'ham': 0, 'spam': 1})
    elif 'Label' in df.columns:
        df['label'] = df['Label'].str.strip().str.lower().map({'ham': 0, 'spam': 1})

    # Get text column
    text_column = 'text' if 'text' in df.columns else 'Text' if 'Text' in df.columns else df.columns[0]
    
    texts = df[text_column].tolist()
    labels = df['label'].tolist()
    
    print(f"Dataset loaded: {len(texts)} emails")
    print(f"Spam: {sum(labels)}, Ham: {len(labels) - sum(labels)}")
    
    return texts, labels


if __name__ == "__main__":
    # Example usage
    print("Testing TextPreprocessor...")
    
    # Sample texts for testing
    sample_texts = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Please verify your information immediately.",
        "Thanks for your help with the project. Let's catch up soon!",
        "FREE VIAGRA! Best prices online. Order now!"
    ]
    
    sample_labels = [1, 0, 1, 0, 1]  # 1=spam, 0=ham
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=1000, max_sequence_length=20, embedding_dim=50)
    
    # Preprocess data
    X, y = preprocessor.preprocess_data(sample_texts, sample_labels, fit=True)
    
    print(f"\nProcessed sequences shape: {X.shape}")
    print(f"Labels: {y}")
    
    # Test single text transformation
    test_text = "Win a free iPhone now! Click here!"
    transformed = preprocessor.transform_single_text(test_text)
    print(f"\nTest text: '{test_text}'")
