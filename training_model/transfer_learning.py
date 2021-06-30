from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from data_ingestion.load_train_data import LoadTrainData
from data_ingestion.load_val_data import LoadValData
from data_ingestion.load_test_data import LoadTestData
from enums.image_info import ImageInfo
from preprocessing.preprocessing import Preprocessing
from tensorflow.keras.applications import mobilenet_v2, MobileNetV2
import tensorflow as tf
import os
from tensorflow.keras.layers.experimental import preprocessing
from application_logging.logger import Logger
import PIL


class TransferLearning(object):

    def __init__(self, file_obj=None):
        self.model = models.Sequential()
        self.obj_train_data = LoadTrainData()
        self.obj_val_data = LoadValData()
        self.obj_test_data = LoadTestData()
        self.train_images = self.obj_train_data.data
        self.test_images = self.obj_test_data.data
        self.val_images = self.obj_val_data.data
        self.data_augmentation = None
        self.preprocessing = Preprocessing()
        self.logger = Logger()
        self.file_obj = file_obj

    def autotune_model(self):
        try:
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            self.train_images = self.train_images.prefetch(buffer_size=AUTOTUNE)
            self.val_images = self.val_images.prefetch(buffer_size=AUTOTUNE)
            self.test_images = self.test_images.prefetch(buffer_size=AUTOTUNE)
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in autotune_model method class TransferLearning {e}')

    def create_model(self):
        try:
            inputs = Input(shape=ImageInfo.IMAGE_SHAPE.value)
            data_augmentation = tf.keras.Sequential([
                preprocessing.Rescaling(1. / 125)
            ])
            x = data_augmentation(inputs)
            x = mobilenet_v2.preprocess_input(x)
            x = MobileNetV2(
                input_shape=ImageInfo.IMAGE_SHAPE.value,
                include_top=False,
                weights='imagenet')(x)
            x.trainable = True
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            self.model = tf.keras.Model(inputs, outputs)
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in create_model method class TransferLearning {e}')

    def compile_model(self):
        try:
            base_learning_rate = 0.0001
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in compile_model method class TransferLearning {e}')

    def fit_model(self):
        try:
            self.model.fit(
                self.train_images,
                validation_data=self.val_images,
                epochs=ImageInfo.EPOCHS.value
            )
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in fit_model method class TransferLearning {e}')

    def train(self):
        try:
            self.logger.log(self.file_obj, f'Training function started')
            self.autotune_model()
            self.logger.log(self.file_obj, f'autotune_model function complete')
            self.create_model()
            self.logger.log(self.file_obj, f'create_model function complete')
            self.compile_model()
            self.logger.log(self.file_obj, f'compile_model function complete')
            self.fit_model()
            self.logger.log(self.file_obj, f'fit_model function complete')
            self.model.save(os.path.join(os.getcwd(), 'models/MobileNetV2/'))
            self.logger.log(self.file_obj, f'mode save function complete, mode saved in models/MobileNetV2/')
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in train method class TransferLearning {e}')

    def load_model(self):
        try:
            self.model = models.load_model(os.path.join(os.getcwd(), 'models/MobileNetV2/'))
            print('model loaded')
            self.logger.log(self.file_obj, f'mode load function complete')
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in train method class TransferLearning {e}')

    def evaluate(self):
        try:
            self.load_model()
            self.model.evaluate(self.test_images, verbose=2)
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in train method class TransferLearning {e}')

    def predict(self, url):
        try:
            image = self.obj_test_data.image_predict(url)
            self.load_model()
            predictions = self.model.predict(image)
            score = tf.nn.sigmoid(predictions[0])
            print(score)
            return score
        except Exception as e:
            self.logger.log(self.file_obj, f'Exception in train method class TransferLearning {e}')