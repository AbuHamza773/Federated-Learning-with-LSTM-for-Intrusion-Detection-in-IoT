
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, num_classes):
    """Create an LSTM model for intrusion detection."""
    model = Sequential()
    
    # LSTM Layer 1
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    
    # LSTM Layer 2
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
    else:
        model.add(Dense(num_classes, activation='softmax'))  # Multiclass classification
    
    model.compile(
        loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the LSTM model."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

if __name__ == "__main__":
    # Example usage (Replace with real data)
    input_shape = (23, 1)  # Example input shape for WSN-DS dataset
    num_classes = 2  # Binary classification example

    # Create model
    model = create_lstm_model(input_shape, num_classes)
    model.summary()

    # Example data for training (Replace with actual preprocessed data)
    import numpy as np
    X_train = np.random.rand(1000, 23, 1)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.rand(200, 23, 1)
    y_val = np.random.randint(0, 2, 200)

    # Train model
    train_model(model, X_train, y_train, X_val, y_val)
