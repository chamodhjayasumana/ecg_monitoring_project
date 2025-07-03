
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def build_model(input_shape):
    model = Sequential([
        Conv1D(64, 11, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 7, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# This code defines a function to build a CNN-LSTM model for ECG signal classification.
# The model consists of convolutional layers for feature extraction, followed by LSTM layers for sequence
# learning, and dense layers for classification. It uses Batch Normalization and Dropout for regularization.
# The model is compiled with the Adam optimizer and categorical crossentropy loss function, suitable for
# multi-class classification tasks. The input shape is expected to be a 1D sequence, typically representing ECG signals.