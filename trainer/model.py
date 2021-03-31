import tensorflow as tf
from tensorflow.keras import layers


def create_model(input_dim: int, class_dim: int) -> tf.keras.Sequential:
    model: tf.keras.Sequential = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(input_dim, activation=tf.nn.relu),
        layers.Dense(300, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu),
        layers.Dense(class_dim, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model
