"""
GRU model for spam email classification.
"""

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class GRUModel:
    """GRU model for binary spam classification."""

    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix=None, bidirectional=True):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.bidirectional = bidirectional
        self.model = None

    def build_model(self):
        """Build the GRU model architecture."""

        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            weights=[self.embedding_matrix] if self.embedding_matrix is not None else None,
            trainable=False if self.embedding_matrix is not None else True,
            name="embedding"
        )

        if self.bidirectional:
            gru_layer = Bidirectional(
                GRU(
                    units=64,
                    return_sequences=False,
                    dropout=0.3,
                    recurrent_dropout=0.3
                )
            )
        else:
            gru_layer = GRU(
                units=128,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3
            )

        self.model = Sequential([
            embedding_layer,
            gru_layer,

            Dense(64, activation="relu"),
            Dropout(0.5),

            Dense(32, activation="relu"),
            Dropout(0.3),

            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("GRU model built successfully!")
        self.model.summary()

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_path=None):

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_path if model_path else "saved_models/gru_best.h5",
                monitor="val_accuracy",
                save_best_only=True
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X):

        if self.model is None:
            raise ValueError("Model not built.")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def predict_classes(self, X):
        probabilities = self.predict(X)
        return (probabilities > 0.5).astype(int)

    def evaluate(self, X_test, y_test):

        if self.model is None:
            raise ValueError("Model not built.")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    def save(self, filepath):

        if self.model is None:
            raise ValueError("Model not built.")

        self.model.save(filepath)
        print(f"GRU model saved to {filepath}")

    def load(self, filepath):

        self.model = load_model(filepath)
        print(f"GRU model loaded from {filepath}")


class StackedGRUModel:
    """Stacked GRU model with multiple GRU layers."""

    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix=None):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.model = None

    def build_model(self):

        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                weights=[self.embedding_matrix] if self.embedding_matrix is not None else None,
                trainable=False if self.embedding_matrix is not None else True
            ),

            GRU(
                units=128,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3
            ),

            GRU(
                units=64,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3
            ),

            Dense(64, activation="relu"),
            Dropout(0.5),

            Dense(32, activation="relu"),
            Dropout(0.3),

            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Stacked GRU model built successfully!")
        self.model.summary()

        return self.model


class FastGRUModel:
    """Faster GRU model with fewer parameters."""

    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix=None):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.model = None

    def build_model(self):

        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                weights=[self.embedding_matrix] if self.embedding_matrix is not None else None,
                trainable=False if self.embedding_matrix is not None else True
            ),

            GRU(
                units=64,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2
            ),

            Dense(32, activation="relu"),
            Dropout(0.3),

            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Fast GRU model built successfully!")
        self.model.summary()

        return self.model
    