import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from trainer.model import create_model
from google.cloud import storage
from typing import Tuple
from typing import Dict
import argparse
import zipfile
import datetime
import numpy as np


# On proper packaging for use with Google AI
# https://cloud.google.com/ai-platform/training/docs/packaging-trainer

time_string: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
image_dir: str = "./dataset"
dataset_file = "dataset.zip"
model_file = f"legoModel_{time_string}.h5"
labels_file = f"labels_{time_string}.csv"
client = storage.Client()


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        default="./models",
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--time-id',
        default=time_string,
        type=str
    )
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning-rate',
        default=0.001,
        type=float
    ),
    parser.add_argument(
        '--num-epochs',
        default=10,
        type=int
    )
    args, _ = parser.parse_known_args()
    return args


def load_training_and_validation_data(args) -> Tuple[ImageDataGenerator]:
    bucket = client.bucket('cmpsc445-bucket')
    blob = bucket.blob(dataset_file)
    blob.download_to_filename(dataset_file)

    with zipfile.ZipFile(dataset_file) as zip_ref:
        zip_ref.extractall()

    datagen: ImageDataGenerator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    train_ds: DirectoryIterator = datagen.flow_from_directory(
        image_dir,
        subset="training",
        seed=123,
        batch_size=args.batch_size,
        class_mode="categorical"
    )

    validation_ds: DirectoryIterator = datagen.flow_from_directory(
        image_dir,
        subset="validation",
        seed=123,
        batch_size=args.batch_size,
        class_mode="categorical"
    )

    return train_ds, validation_ds


def train(args):
    # Load Data for training and validation
    train_ds, validation_ds = load_training_and_validation_data(args)
    classes: Dict[int] = train_ds.class_indices
    class_labels = list(classes.keys())
    np.savetxt(labels_file, class_labels, delimiter=",", fmt="%s")

    # Create, train and log model
    # To view logs during training enter the line below into the command line
    # tensorboard --logdir=gs://cmpsc445-models/logs --port=8080
    model: tf.keras.Sequential = create_model(train_ds.num_classes, args.learning_rate)

    log_dir = f"{args.job_dir}logs/{args.time_id}"
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True)

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.num_epochs,
        callbacks=[tensorboard])

    model = tf.keras.models.load_model("legoModel_20210413-070348.h5")
    Y_pred = model.predict(validation_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    y_pred = [class_labels[pred] for pred in y_pred]
    y_true = [class_labels[true] for true in validation_ds.classes]
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    print(cm)
    print('Report')
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Save model as hdf5 format
    model.save(model_file)

    bucket = client.bucket("cmpsc445-models")
    model_blob = bucket.blob(model_file)
    model_blob.upload_from_filename(model_file)

    # Save label names to GCS
    labels_blob = bucket.blob(labels_file)
    labels_blob.upload_from_filename(labels_file)

if __name__ == "__main__":
    arguments = get_args()
    train(arguments)
