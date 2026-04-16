"""
Training script for all neural network models.
Trains RNN, LSTM, and GRU models and saves the best performing ones.
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import tensorflow as tf

# Import our modules
from preprocessing.text_cleaning import TextPreprocessor, load_dataset
from models.rnn_model import SimpleRNNModel
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ModelTrainer:
    """Handles training of all neural network models."""

    def __init__(self, dataset_path, output_dir='saved_models', visualization_dir='visualizations'):

        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)

        self.model_configs = {
            'vocab_size': 10000,
            'embedding_dim': 100,
            'max_sequence_length': 100
        }

        self.training_params = {
            'epochs': 50,
            'batch_size': 32
        }

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.preprocessor = None
        self.embedding_matrix = None

    def load_and_preprocess_data(self):
        """Load dataset and preprocess it for training."""

        print("Loading and preprocessing dataset...")

        texts, labels = load_dataset(self.dataset_path)

        self.preprocessor = TextPreprocessor(
            max_vocab_size=self.model_configs['vocab_size'],
            max_sequence_length=self.model_configs['max_sequence_length'],
            embedding_dim=self.model_configs['embedding_dim']
        )

        X, y = self.preprocessor.preprocess_data(texts, labels, fit=True)
        self.model_configs['vocab_size'] = len(self.preprocessor.word2idx)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        self.embedding_matrix = self.preprocessor.create_embedding_matrix()

        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_rnn_model(self):

        print("\n" + "="*50)
        print("TRAINING SIMPLE RNN MODEL")
        print("="*50)

        rnn_model = SimpleRNNModel(
            vocab_size=len(self.preprocessor.word2idx),
            embedding_dim=self.preprocessor.embedding_dim,
            max_sequence_length=self.preprocessor.max_sequence_length,
            embedding_matrix=self.embedding_matrix
        )

        rnn_model.build_model()

        model_path = os.path.join(self.output_dir, 'rnn_best.h5')

        history = rnn_model.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=self.training_params['epochs'],
            batch_size=self.training_params['batch_size'],
            model_path=model_path
        )

        self.plot_training_history(history, "RNN", "rnn_training")

        return rnn_model, history

    def train_lstm_model(self):

        print("\n" + "="*50)
        print("TRAINING LSTM MODEL")
        print("="*50)

        print("vocab_size:", len(self.preprocessor.word2idx))
        print("embedding_matrix shape:", self.embedding_matrix.shape)

        assert self.embedding_matrix is None or self.embedding_matrix.shape[0] == len(self.preprocessor.word2idx), \
            "Embedding matrix vocab size mismatch!"

        lstm_model = LSTMModel(
        vocab_size=len(self.preprocessor.word2idx),
        embedding_dim=100,
        max_sequence_length=100,
        embedding_matrix=self.embedding_matrix
        )

        lstm_model.build_model()

        model_path = os.path.join(self.output_dir, 'lstm_best.h5')

        history = lstm_model.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=self.training_params['epochs'],
            batch_size=self.training_params['batch_size'],
            model_path=model_path
        )

        self.plot_training_history(history, "LSTM", "lstm_training")

        return lstm_model, history

    def train_gru_model(self):

        print("\n" + "="*50)
        print("TRAINING GRU MODEL")
        print("="*50)

        gru_model = GRUModel(
            vocab_size=self.model_configs['vocab_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            max_sequence_length=self.model_configs['max_sequence_length'],
            embedding_matrix=self.embedding_matrix,
            bidirectional=True
        )

        gru_model.build_model()

        model_path = os.path.join(self.output_dir, 'gru_best.h5')

        history = gru_model.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=self.training_params['epochs'],
            batch_size=self.training_params['batch_size'],
            model_path=model_path
        )

        self.plot_training_history(history, "GRU", "gru_training")

        return gru_model, history

    def plot_training_history(self, history, model_name, filename):

        acc = history.history.get('accuracy') or history.history.get('acc')
        val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0,0].plot(acc)
        axes[0,0].plot(val_acc)
        axes[0,0].set_title(f"{model_name} Accuracy")

        axes[0,1].plot(history.history['loss'])
        axes[0,1].plot(history.history['val_loss'])
        axes[0,1].set_title(f"{model_name} Loss")

        axes[1,0].plot(acc)
        axes[1,0].set_title("Training Accuracy")

        axes[1,1].plot(val_acc)
        axes[1,1].set_title("Validation Accuracy")

        plt.tight_layout()

        path = os.path.join(self.visualization_dir, f"{filename}.png")
        plt.savefig(path)
        plt.close()

        print(f"Training plots saved to {path}")

    def plot_confusion_matrix(self, cm, model_name, filename):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['Not Spam', 'Spam']
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=classes,
            yticklabels=classes,
            title=f'{model_name} Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        thresh = cm.max() / 2. if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        
        fig.tight_layout()
        path = os.path.join(self.visualization_dir, f"{filename}.png")
        plt.savefig(path)
        plt.close()
        print(f"Confusion matrix saved to {path}")

    def evaluate_models(self, models):

        print("\nMODEL EVALUATION")

        results = {}

        for name, model in models.items():

            y_pred_proba = model.predict(self.X_test).reshape(-1,)
            y_pred = (y_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }

            print(name)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1:", f1)
            print("Confusion Matrix:\n", cm)

            self.plot_confusion_matrix(cm, name, f"{name.lower()}_confusion_matrix")

        return results

    def find_best_model(self, results):

        best = max(results, key=lambda x: results[x]['f1_score'])

        print("Best Model:", best)

        return best

    def train_all_models(self):

        self.load_and_preprocess_data()

        # Save preprocessor after preprocessing
        self.preprocessor.save(os.path.join(self.output_dir, 'preprocessor.pkl'))

        models = {}

        rnn, _ = self.train_rnn_model()
        models["RNN"] = rnn

        lstm, _ = self.train_lstm_model()
        models["LSTM"] = lstm

        gru, _ = self.train_gru_model()
        models["GRU"] = gru

        results = self.evaluate_models(models)

        best_model = self.find_best_model(results)

        # Save best model info
        best_model_info = {
            'best_model_name': best_model,
            'best_model_path': os.path.join(self.output_dir, f'{best_model.lower()}_best.h5')
        }
        with open(os.path.join(self.output_dir, 'best_model_info.pkl'), 'wb') as f:
            pickle.dump(best_model_info, f)

        with open(os.path.join(self.output_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)

        return models, results, best_model


def main():

    DATASET_PATH = "dataset/spam_dataset.csv"

    if not os.path.exists(DATASET_PATH):
        print("Dataset not found:", DATASET_PATH)
        return

    trainer = ModelTrainer(DATASET_PATH)

    models, results, best_model = trainer.train_all_models()

    print("\nTraining completed.")
    print("Best model:", best_model)


if __name__ == "__main__":
    main()