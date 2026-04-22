"""
Simple RNN model for spam email classification.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


class SimpleRNNModel:
    """Simple RNN model for binary spam classification."""

    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix):
        """
        Initialize the RNN model.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.model = None

    def build_model(self):
        """Build the Simple RNN model architecture."""

        self.model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                weights=[self.embedding_matrix],
                trainable=False,
                name="embedding"
            ),

            # RNN layer
            SimpleRNN(
                units=64,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3,
                name="rnn"
            ),

            Dense(32, activation="relu"),
            Dropout(0.5),

            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Simple RNN model built successfully!")
        self.model.summary()

        return self.model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=32, model_path=None):

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_path if model_path else "saved_models/rnn_best.h5",
                monitor="val_accuracy",
                save_best_only=True
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.predict(X, verbose=0).flatten()

    def predict_classes(self, X):
        return (self.predict(X) > 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.evaluate(X_test, y_test, verbose=0)

    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath)
