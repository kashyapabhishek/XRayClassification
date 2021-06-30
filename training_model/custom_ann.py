import os
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import BinaryCrossentropy
from data_ingestion.load_train_data import LoadTrainData
from data_ingestion.load_val_data import LoadValData
from enums.image_info import ImageInfo
from application_logging.logger import Logger


class CustomAnn(object):

    def __init__(self, file_object):
        self.obj_train_data = LoadTrainData()
        self.obj_val_data = LoadValData()
        self.train_images = self.obj_train_data.data
        self.val_images = self.obj_val_data.data
        self.model = models.Sequential()
        self.summary = None
        self.logger = file_object
        self.logger = Logger()

    def create_model(self):
        try:
            self.model.add(Input(shape=ImageInfo.IMAGE_SHAPE.value))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in create model method class CustomAnn {e}')

    def compile_model(self):
        try:
            self.model.compile(
                loss=BinaryCrossentropy(from_logits=False),
                optimizer='adam',
                metrics=['accuracy']
            )
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in create model method class CustomAnn {e}')

    def fit_model(self):
        try:
            self.model.fit(
                self.train_images,
                validation_data=self.val_images,
                batch_size=ImageInfo.BATCH_SIZE.value,
                epochs=ImageInfo.EPOCHS.value
            )
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in fit model method class CustomAnn {e}')

    def train(self):
        try:
            self.logger.log(self.file_obj, f'create model start CustomCnn')
            self.create_model()
            self.logger.log(self.file_obj, f'create model done CustomCnn')
            self.logger.log(self.file_obj, f'compile model start CustomCnn')
            self.compile_model()
            self.logger.log(self.file_obj, f'compile model done CustomCnn')
            self.logger.log(self.file_obj, f'fit model start CustomCnn')
            self.fit_model()
            self.logger.log(self.file_obj, f'fit model done CustomCnn')
            self.logger.log(self.file_obj, f'save model start CustomCnn')
            self.model.save(os.path.join(os.getcwd(), 'models/ann/'))
            self.logger.log(self.file_obj, f'save model start CustomCnn')
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in train method class CustomCnn {e}')
