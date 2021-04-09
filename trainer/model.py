import tensorflow as tf
from tensorflow.keras import layers


def create_model(class_dim: int, learning_rate: float) -> tf.keras.Sequential:
    # Model based on Tensorflow Image Classification Tutorial
    # model: tf.keras.Sequential = tf.keras.Sequential([
    #     layers.experimental.preprocessing.Rescaling(1. / 255),
    #     layers.Conv2D(32, 3, activation=tf.nn.relu),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, activation=tf.nn.relu),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, activation=tf.nn.relu),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(input_dim, activation=tf.nn.relu),
    #     layers.Dense(300, activation=tf.nn.relu),
    #     layers.Dense(200, activation=tf.nn.relu),
    #     layers.Dense(100, activation=tf.nn.relu),
    #     layers.Dense(class_dim, activation=tf.nn.softmax)
    # ])
    # Based on AlexNet CNN Architecture
    model: tf.keras.Sequential = tf.keras.Sequential([
        layers.Conv2D(
            filters=96,
            kernel_size=11,
            strides=4,
            padding="valid",
            activation=tf.nn.relu,
            input_shape=(227, 227, 3)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=3, strides=2, padding="valid"),
        layers.Conv2D(
            filters=256,
            kernel_size=5,
            strides=1,
            padding="same",
            activation=tf.nn.relu),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=3, strides=2, padding="valid"),
        layers.Conv2D(
            filters=384,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=384,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu
        ),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=3, strides=2, padding="valid"),
        layers.Flatten(),
        layers.Dense(4096, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(4096, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(class_dim, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    return model
