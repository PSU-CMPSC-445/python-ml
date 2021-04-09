# python-ml
Repo to house python code to build out the machine learning model for the semester project

Setup
======
1. [Install GCloud command line tools](https://cloud.google.com/sdk/docs/quickstart)
2. Download Key for python-ml@ardent-strategy-308404.iam.gserviceaccount.com service account
    * From Google Cloud Platform, choose IAM & Admin
    * Select Service Accounts from the pop up list
    * Select the python-ml@ardent-strategy-308404.iam.gserviceaccount.com service account
    * Choose Keys
    * Add Key
    * Save your key's json file somewhere safe on your local machine!
3. Create a System Environment variable called `GOOGLE_APPLICATION_CREDENTIALS` that points to the location of 
your service account access key file.
    * For me, this was in my bash profile
4. From the command line, run `gcloud init`
    * Our project is called **ardent-strategy-308404**
5. Download project libraries `pip install -r requirements.txt`


Project Structure
==================

The following is the general flow of our system from training to prediction by the client:
1. Create a training job in AI-Platform to create a new model
2. The newly created model is saved to a cloud bucket along with the class labels
3. The user uploads an image through the client
4. The client sends the image to a cloud bucket at `cmpsc445-uploads`
5. The client posts to `https://us-central1-ardent-strategy-308404.cloudfunctions.net/get-prediction`
with the following json (where the filename is the name of the file just uploaded to our bucket)

        {
            "bucket": "cmpsc445-uploads",
            "filename": "3001-BBA14BB7-DA8A-4BCD-BF88-0F4CEA5C1ADC.jpeg"
        }
        
6. The request initiates a Cloud Function
7. The Cloud Function (`get-prediction`) loads the previously saved model and labels from the cloud bucket, 
downloads the user uploaded image from its cloud bucket, and then makes a prediction with the iamge and model.
8. The results of the prediction are returned to the client from the Cloud Function
9. The client displays the results of the prediction to the user.

AI-Platform
-------------
- Google's AI Platform is used to train the model remotely
- To create a new training job, run the `submint_training.sh` script
    * Make sure to change the number of the job name, each job must have a unique name
    * A training job can be found through the AI-Platform section in Google Cloud under the jobs sub-section
   
- The file structure of the model training code follows proper packaging format laid out 
[here](https://cloud.google.com/ai-platform/training/docs/packaging-trainer)
   
       +--trainer
       |
       |    +--model.py
       |     
       |    +--task.py
       |
       +--setup.py

 **model.py**
  - responsible for defining and compiling the Sequential ML Model
  
 **task.py**
  - Main entry point of the ai-platform package.
  - Downloads a zip of the image dataset to the local server where it is running and 
  unzips the files.
  - loads all images with tensorflow
  - creates the model, trains the model, and saves it as an .h5 file a google bucket 
  - saves a .csv file with the label class names to the same bucket.
  
**setup.py**
 - Lists out requirements so Google AI-Platform can download them
 - sets other project level settings
 
 Cloud Function
 --------------
 - The cloud function `get-prediction` receives incoming requests from the client to make
 a prediction on a specific image already uploaded to a cloud bucket.
 - The cloud function will load the already created ml model, use the model to make a prediction
 and then return the prediction to the client.
 - Here is the file strucutre of the function:
 
        +--predict.py
        +--requirements.txt
        
 **predict.py**
 - This is where all of the magic happens 
    * model and image loading
    * image pre-processing
    * etc
 
 **requirements.txt**
 - The function gets its own requirements file since it lives on its on in the cloud.
 These are only the requirements needed to make the Cloud Function work.
 
 Resources
 ==========
 - [Training an image classifier with Tensorflow](https://www.tensorflow.org/tutorials/images/classification)
 - [Packaging code to submit a job to AI-Platform](https://cloud.google.com/ai-platform/training/docs/packaging-trainer)
 - [Flask Requests (Cloud Function)](https://flask.palletsprojects.com/en/1.1.x/reqcontext/)
 - [Http Cloud Function](https://cloud.google.com/functions/docs/writing/http)
 - [Downloading files from Bucket with Python](https://medium.com/@sandeepsinh/multiple-file-download-form-google-cloud-storage-using-python-and-gcs-api-1dbcab23c44)
 - [Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-on-google-cloud-platform-with-scikit-learn-7d6155195efb)
 - [Google Hyperparameter Docs](https://cloud.google.com/ai-platform/training/docs/using-hyperparameter-tuning)
 - [Image Data Augmentation](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)


Next Steps
==========
 - Setup our AI-Platorm job to automatically tune hyperparameters
 - Add any additional desired image pre-processing
 - Create variations for each real life lego image to bolster that dataset
 - Prepare full dataset by grouping images into folder classes
 - Fix Tensorboard to show training progress
 - Train full model
 