import tensorflow as tf

save_dir: str = "./models/"
img_path: str = "./archive/test-images/3001-BBA14BB7-DA8A-4BCD-BF88-0F4CEA5C1ADC.jpeg"
img_height: int = 3024
img_width: int = 3024

# my_model = tf.keras.models.load_model(save_dir + "my_model.h5")

my_model = tf.keras.models.load_model("gs://cmpsc445-models/model-26475b83-6383-4df2-988e-391849a148f8")

my_model.summary()

test_img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=(400, 400)
)

img_array = tf.keras.preprocessing.image.img_to_array(test_img)
img_array = tf.expand_dims(img_array, 0)    # Create a batch

predictions = my_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
