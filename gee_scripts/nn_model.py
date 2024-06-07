from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def get_nn_model() -> Model:
    # LSTM Sequence Input
    # 3 time steps, 1 feature (sm value)
    sequence_input_1 = Input(shape=(3, 1), name="sequence_input_1")
    lstm_out_1 = LSTM(50)(sequence_input_1)  # 50 LSTM units, can be tuned

    # 3 time steps, 1 feature (sm value)
    sequence_input_2 = Input(shape=(3, 1), name="sequence_input_2")
    # 50 LSTM units, can be tuned
    lstm_out_2 = LSTM(50)(sequence_input_2)

    # Dense Input for non-sequential data
    # 26 other explanatory variables
    dense_input = Input(shape=(22,), name="dense_input")
    dense_out = Dense(50, activation="relu")(dense_input)  # 50 units, can be tuned

    # Combine LSTM and Dense outputs
    merged = Concatenate()([lstm_out_1, lstm_out_2, dense_out])

    # Add further dense layers if needed
    dense_merged_1 = Dense(100, activation="relu")(merged)
    dense_merged_2 = Dense(50, activation="relu")(dense_merged_1)

    # Regression output
    output = Dense(1, activation="linear")(dense_merged_2)

    # Compile the model
    model = Model(
        inputs=[sequence_input_1, sequence_input_2, dense_input], outputs=output
    )

    return model


def get_nn_model_(input_dim: int, hidden_units: int = 64) -> Sequential:
    """
    Create a simple deep learning model using only Dense layers.

    Parameters:
    - input_dim: The number of input features.
    - hidden_units: Number of units for the hidden layer.

    Returns:
    - A compiled Keras model.
    """
    model = Sequential(
        [
            Dense(hidden_units, activation="relu", input_shape=(input_dim,)),
            Dense(hidden_units, activation="relu"),
            Dense(1),
        ]
    )
    return model
