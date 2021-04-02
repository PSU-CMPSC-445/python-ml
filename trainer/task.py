import tensorflow as tf
from trainer.model import create_model
from google.cloud import storage
import argparse
import zipfile
import numpy as np


# On proper packaging for use with Google AI
# https://cloud.google.com/ai-platform/training/docs/packaging-trainer

image_dir: str = "./dataset"
dataset_file = "dataset.zip"
model_file = "legoModel.h5"
labels_file = "labels.csv"
client = storage.Client()


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        default="../models",
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--image-height',
        default=400,
        type=int
    )
    parser.add_argument(
        '--image-width',
        default=400,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--num-epochs',
        default=1,
        type=int
    )
    args, _ = parser.parse_known_args()
    return args


def load_training_and_validation_data(args):
    bucket = client.bucket('cmpsc445-bucket')
    blob = bucket.blob(dataset_file)
    blob.download_to_filename(dataset_file)

    with zipfile.ZipFile(dataset_file) as zip_ref:
        zip_ref.extractall()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size
    )

    return train_ds, validation_ds


def train(args):
    AUTOTUNE = tf.data.AUTOTUNE
    # Load Data for training and validation
    train_ds, validation_ds = load_training_and_validation_data(args)
    np.savetxt(labels_file, [train_ds.class_names], delimiter=",", fmt="%s")
    num_classes = len(train_ds.class_names)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Create, train and log model
    # To view logs during training tensorboard --logdir=gs://cmpsc445-models --port=8080
    model: tf.keras.Sequential = create_model(args.image_height, num_classes)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"{args.job_dir}logs/",
        histogram_freq=0,
        write_graph=True,
        write_images=True)
    model.fit(train_ds, validation_data=validation_ds, epochs=args.num_epochs, callbacks=[tensorboard])

    # Save model as hdf5 format
    model.save(model_file)
    print(model.summary())
    bucket = client.bucket("cmpsc445-models")
    model_blob = bucket.blob(model_file)
    model_blob.upload_from_filename(model_file)

    # Save label names to GCS
    labels_blob = bucket.blob(labels_file)
    labels_blob.upload_from_filename(labels_file)


if __name__ == "__main__":
    arguments = get_args()
    train(arguments)
