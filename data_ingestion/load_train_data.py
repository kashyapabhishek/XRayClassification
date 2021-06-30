from enums.image_info import ImageInfo
from tensorflow.keras.preprocessing import image_dataset_from_directory
from application_logging.logger import Logger
import os


class LoadTrainData(object):

    def __init__(self, file_object):
        self.url = os.path.join(os.getcwd(), ImageInfo.IMAGE_TRAIN_URL.value)
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
            self.logger.log(self.file_obj, f'Exception in data getter class LoadTrainData {e}')

