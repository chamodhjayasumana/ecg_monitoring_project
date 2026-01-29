
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional


def build_model(input_shape, num_classes=2):
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
        Dense(64, activation='relu')
    ])

    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model
