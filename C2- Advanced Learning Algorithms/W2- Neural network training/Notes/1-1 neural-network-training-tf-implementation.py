import tensorflow as tf
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore

X = [[]]
Y = [[]]

#! STEP 1/
model = Sequential(
    [
        Dense(units=25, activation="signmoid"),
        Dense(units=15, activation="signmoid"),
        Dense(units=1, activation="signmoid"),
    ]
)

#! STEP 2/
model.compile(loss=BinaryCrossentropy)

#! STEP 3/
model.fit(X, Y, epochs=100)
