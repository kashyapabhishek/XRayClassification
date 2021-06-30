from tensorflow.keras.layers.experimental import preprocessing
from application_logging.logger import Logger


class Preprocessing(object):

    def __init__(self, file_object):
        self.data_augmentation = None
        self.logger = file_object
        self.logger = Logger()

    def augmentation(self):
        try:
            self.data_augmentation = [
                preprocessing.RandomFlip('horizontal'),
                preprocessing.RandomRotation(0.2),
                preprocessing.Rescaling(1./125)
            ]
            return self.data_augmentation
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in augmentation function class Preprocessing {e}')
