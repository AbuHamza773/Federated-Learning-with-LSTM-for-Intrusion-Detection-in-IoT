
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

def train_local_model(model, X_local, y_local, epochs=10, batch_size=32):
    """Train a local model on the given data."""
    model.fit(
        X_local, y_local,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model.get_weights()

def prepare_data(X, y, num_classes):
    """Prepare data by splitting and one-hot encoding labels if needed."""
    y = to_categorical(y, num_classes) if num_classes > 2 else y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
