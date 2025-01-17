## Task 1
**( classify image )**

This code employs a pre-trained deep learning model to categorize images. It utilizes the TensorFlow library to import the model and handle the input images. The picture is sorted into specific categories like "cat," "dog," or "rabbit." If the image does not fit into any of these categories, it is categorized as "not an animal." The aim of this code is to offer a simple and precise method for recognizing the content of images through artificial intelligence. 

***This animals category:***
- Processes the image with the model to create predictions.
- Obtains the anticipated class index and confidence level.
- Associates the index with a class name by utilizing the labels file.
- Displays the expected category and level of certainty.

**Primary Reasoning:**

Step 1: Verifies whether the input image is present.

Step 2: Loads the model that has been trained (keras_model.h5).

Step 3: Retrieves the class labels from labels.txt.

Step 4: Uses predict_class to categorize the image and displays the outcome.

**When an image (e.g., ccc.jpg) is supplied, *the script:***

1- Uploads the picture.

2- Prepares it to align with the model.

3- Processes it with the model to create predictions.

4- Provides the anticipated class label and the confidence level.
