import googleapiclient.discovery
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from google.cloud import storage
import tempfile
import csv
import json
import numpy as np


# Create the AI Platform service object.
# To authenticate set the environment variable
# GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
service = googleapiclient.discovery.build('ml', 'v1')
storage_client = storage.Client()

model = None
labels = None


def predict(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    global model
    global labels

    if model is None:
        print("loading model...")
        _, model_tmp_filename = tempfile.mkstemp()
        model_blob = storage_client.bucket("cmpsc445-models").get_blob("legoModel_20210413-060050.h5")
        model_blob.download_to_filename(model_tmp_filename)
        model = tf.keras.models.load_model(model_tmp_filename)
        model = tf.keras.models.load_model("../trainer/legoModel_20210413-070348.h5")

    if labels is None:
        print("loading labels...")
        _, labels_tmp_filename = tempfile.mkstemp()
        labels_blob = storage_client.bucket("cmpsc445-models").get_blob("labels_20210413-060050.csv")
        labels_blob.download_to_filename(labels_tmp_filename)
        labels = np.loadtxt(labels_tmp_filename, dtype='str', delimiter=',')
        labels = np.loadtxt("../trainer/labels_20210413-070348.csv", dtype='str', delimiter=',')

    request_json = request.get_json()
    image_filename = request_json['filename']
    _, img_tmp_filename = tempfile.mkstemp()
    img_blob = storage_client.bucket(request_json['bucket']).get_blob(image_filename)
    img_blob.download_to_filename(img_tmp_filename)

    test_img = tf.keras.preprocessing.image.load_img(img_tmp_filename)
    img_array = tf.keras.preprocessing.image.img_to_array(test_img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions, axis=1)[0]

    return json.dumps({
        "label": labels[pred_index],
        "confidence": str(predictions.item(pred_index))
    }), 200, {'Content-Type': 'application/json'}


if __name__ == "__main__":
    response = predict({
        "bucket": "cmpsc445-uploads",
        "filename": "2420-7B4EE615-7B53-4A25-BA24-64318B00EB88.jpeg"
    })
    print(type(response))
    print(response)

