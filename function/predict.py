import googleapiclient.discovery
import tensorflow as tf
from google.cloud import storage
import tempfile
import csv
import json


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
        model_blob = storage_client.bucket("cmpsc445-models").get_blob("legoModel.h5")
        model_blob.download_to_filename(model_tmp_filename)
        model = tf.keras.models.load_model(model_tmp_filename)
        print(model.summary())

    if labels is None:
        print("loading labels...")
        _, labels_tmp_filename = tempfile.mkstemp()
        labels_blob = storage_client.bucket("cmpsc445-models").get_blob("labels.csv")
        labels_blob.download_to_filename(labels_tmp_filename)
        with open(labels_tmp_filename) as f:
            reader = csv.reader(f)
            labels = list(reader)
            labels = labels[0]

    request_json = request # request.get_json()
    image_filename = request_json['filename']
    _, img_tmp_filename = tempfile.mkstemp()
    img_blob = storage_client.bucket(request_json['bucket']).get_blob(image_filename)
    img_blob.download_to_filename(img_tmp_filename)

    test_img = tf.keras.preprocessing.image.load_img(
        img_tmp_filename, target_size=(400, 400)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(test_img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_index = tf.argmax(score)
    return json.dumps({
        "label": labels[predicted_index],
        "confidence": str(score[predicted_index].numpy())
    }), 200, {'Content-Type': 'application/json'}


if __name__ == "__main__":
    response = predict({
        "bucket": "cmpsc445-uploads",
        "filename": "3001-BBA14BB7-DA8A-4BCD-BF88-0F4CEA5C1ADC.jpeg"
    })
    print(type(response))
    print(response)

