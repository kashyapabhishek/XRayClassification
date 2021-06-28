from enums.image_info import ImageInfo
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


class LoadTrainData(object):

    def __init__(self):
        self.url = os.path.join(os.getcwd(), ImageInfo.IMAGE_TRAIN_URL.value)
        self.generator = None

    @property
    def data(self):
        return image_dataset_from_directory(
            self.url,
            shuffle=True,
            batch_size=ImageInfo.BATCH_SIZE.value,
            image_size=ImageInfo.IMAGE_SIZE.value)

