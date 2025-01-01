import tensorflow as tf
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

from tensorflow.keras.losses import SparseCategoricalCrossEntropy  # type: ignore

X = [[]]
Y = [[]]

# ! VERSION 1:
model = Sequential(
    [
        Dense(units=25, activation="relu"),
        Dense(units=15, activation="relu"),
        Dense(units=10, activation="softmax"),
    ]
)
model.compile(loss=SparseCategoricalCrossEntropy)
model.fit(X, Y, epochs=100)

# ! RECOMMENDED VERSION:
# * IMPROVE NUMERICAL ROUNDOFF ERRORS
model = Sequential(
    [
        Dense(units=25, activation="relu"),
        Dense(units=15, activation="relu"),
        Dense(units=10, activation="linear"),
    ]
)
model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
model.fit(X, Y, epochs=100)

"""
--> In the recommended version, the output layer doesn't give the probability for each label.
--> It rather outputs the values z1, z2, z3, ...
--> To get the labels probabilities, add this:
    > logits = model(X)
    > f_x = tf.nn.softmax(logits)
"""
