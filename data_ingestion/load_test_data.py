from enums.image_info import ImageInfo
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import os
import tensorflow as tf
from application_logging.logger import Logger


class LoadTestData(object):

    def __init__(self, file_object):
        self.url = os.path.join(os.getcwd(), ImageInfo.IMAGE_TEST_URL.value)
        self.generator = None
        self.logger = file_object
        self.logger = Logger()

    @property
    def data(self):
        try:
            return image_dataset_from_directory(
                self.url,
                shuffle=True,
                batch_size=ImageInfo.BATCH_SIZE.value,
                image_size=ImageInfo.IMAGE_SIZE.value)
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in data getter class LoadTestData {e}')

    def image_predict(self, url):
        try:
            image_load = image.load_img(url, target_size=ImageInfo.IMAGE_SIZE.value)
            input_arr = image.img_to_array(image_load)
            img_array = tf.expand_dims(input_arr, 0)  # Create a batc
            return img_array
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in image_predict methord class LoadTestData {e}')
