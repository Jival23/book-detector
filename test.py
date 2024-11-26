import tensorflow as tf
import cv2
from keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

validation_file = "validation"
model_file = "models/book_detection_v5.keras"
model = tf.keras.models.load_model(model_file)

'''validation_augmentation = ImageDataGenerator(
    rescale=1.0/255
)

validation_data = validation_augmentation.flow_from_directory (
    validation_file,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(validation_data)'''


def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(224, 224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult


classes = ['Book', 'Not_a_book']
testImagePath = "validation/Book/20240927_201739.jpg"

imgForModel = prepareImage(testImagePath)

resultsArray = model(imgForModel)
print(resultsArray)

answer = np.argmax(resultsArray, axis=1)
print(answer)

index = answer[0]
print(classes[index])
