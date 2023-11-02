from typing import List, Tuple
from tensorflow.keras.layers import concatenate, Input, LSTM, Dense
from tensorflow.keras.models import Model


def create_lstm_branch(input_shape: Tuple[int, int], units: int) -> LSTM:
    """Create an LSTM branch."""
    input_layer = Input(shape=input_shape)
    return input_layer, LSTM(units)(input_layer)


def create_dense_branch(input_shape: int, units: int) -> Dense:
    """Create a dense branch."""
    input_layer = Input(shape=(input_shape,))
    return input_layer, Dense(units, activation="relu")(input_layer)


def create_hybrid_model(lstm_units: int = 50, dense_units: int = 64) -> Model:
    """
    Create a hybrid deep learning model combining LSTM and Dense layers.

    Parameters:
    - lstm_units: Number of units for the LSTM layers
    - dense_units: Number of units for the Dense layer

    Returns:
    - A compiled Keras model
    """
    # LSTM branches
    input_shapes_time_series = [(3, 1), (7, 1), (30, 1), (3, 1), (7, 1), (30, 1)]
    lstm_inputs_and_layers = [
        create_lstm_branch(shape, lstm_units) for shape in input_shapes_time_series
    ]

    # Dense branch for static features
    dense_input, dense_layer = create_dense_branch(26, dense_units)

    # Concatenation
    all_inputs = [inp for inp, _ in lstm_inputs_and_layers] + [dense_input]
    all_layers = [layer for _, layer in lstm_inputs_and_layers] + [dense_layer]
    merged = concatenate(all_layers)

    # Output layer
    output = Dense(1)(merged)

    model = Model(inputs=all_inputs, outputs=output)
    return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_simplified_model(input_dim: int, hidden_units: int = 64) -> Sequential:
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
