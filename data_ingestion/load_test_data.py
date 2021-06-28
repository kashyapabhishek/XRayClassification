from enums.image_info import ImageInfo
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import os
import tensorflow as tf
import numpy as np


class LoadTestData(object):

    def __init__(self):
        self.url = os.path.join(os.getcwd(), ImageInfo.IMAGE_TEST_URL.value)
        self.generator = None

    @property
    def data(self):
        return image_dataset_from_directory(
            self.url,
            shuffle=True,
            batch_size=ImageInfo.BATCH_SIZE.value,
            image_size=ImageInfo.IMAGE_SIZE.value)

    def image_predict(self, url):
        image_load = image.load_img(url, target_size=ImageInfo.IMAGE_SIZE.value)
        input_arr = image.img_to_array(image_load)
        img_array = tf.expand_dims(input_arr, 0)  # Create a batc
        return img_array
